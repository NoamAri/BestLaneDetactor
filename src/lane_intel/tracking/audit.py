"""Visual audit of stitch joins: is the vehicle after the join the same vehicle as before it?

Draws a random sample of joins (fixed seed) from one variant of a tracking experiment and renders
one card per join: the clearest view of piece A in its last 1.5 s next to the clearest view of
piece B in its first 1.5 s, cropped from the full-resolution video. A person then fills the
``verdict`` column of the CSV with ``same`` or ``different``; ``--score`` summarises it.

Usage
    python -m lane_intel.tracking.audit --cfg configs/T8_D1.json \
        --experiment data/out/tracking/T8_D1_30-30s_step3 --variant s3_stitch_5s \
        --n 20 --seed 1 --out notes/audits/T8_D1_30-30s_s3_stitch_5s_seed1
    # writes <out>.jpg (cards) and <out>.csv (one row per join, verdict column empty)
    python -m lane_intel.tracking.audit --score notes/audits/T8_D1_30-30s_s3_stitch_5s_seed1.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CARD_H = 170          # height of each vehicle crop on a card, px
LOOK_FRAMES = 45      # search the clearest view within 1.5 s of the join
MARGIN = 25           # context around the box, full-resolution px
CROP_Y = (1040, 1660) # rows decoded from the video (the road strip, as in the viewer)


def sample_links(exp_dir: Path, variant: str, n: int, seed: int) -> pd.DataFrame:
    links = pd.read_csv(exp_dir / variant / "links.csv")
    jumps = pd.read_csv(exp_dir / variant / "jumps.csv")
    pts = pd.read_parquet(exp_dir / f"points_{variant}.parquet")
    s = links.sample(n=min(n, len(links)), random_state=seed).reset_index(drop=True)
    vehicle = pts.groupby("raw_track_id").track_id.first()
    flagged = set(zip(jumps.track_id, jumps.frame_idx))
    s["vehicle"] = s.from_track.map(vehicle)
    s["flagged"] = [(v, f) in flagged for v, f in zip(s.vehicle, s.frame_start)]
    s.insert(0, "card", range(1, len(s) + 1))
    s["verdict"] = ""
    return s


def clearest_views(pts: pd.DataFrame, s: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    pts = pts.assign(area=(pts.x2 - pts.x1) * (pts.y2 - pts.y1))
    out = []
    for r in s.itertuples():
        a = pts[(pts.raw_track_id == r.from_track) & (pts.frame_idx > r.frame_end - LOOK_FRAMES)]
        b = pts[(pts.raw_track_id == r.to_track) & (pts.frame_idx < r.frame_start + LOOK_FRAMES)]
        out.append((a.nlargest(1, "area").iloc[0], b.nlargest(1, "area").iloc[0]))
    return out


def read_frames(video: str, frames: list[int], crop_y: tuple[int, int], width: int) -> dict[int, np.ndarray]:
    """Decode only the listed frames (one ffmpeg pass), cropped to rows crop_y."""
    y0, y1 = crop_y
    h = y1 - y0
    sel = "+".join(f"eq(n\\,{f})" for f in frames)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"select={sel},crop={width}:{h}:0:{y0}",
                          "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    size = width * h * 3
    return {f: np.frombuffer(raw[i * size:(i + 1) * size], np.uint8).reshape(h, width, 3) for i, f in enumerate(frames)}


def render(cfg: dict, exp_dir: Path, variant: str, s: pd.DataFrame, out_jpg: Path) -> None:
    pts = pd.read_parquet(exp_dir / f"points_{variant}.parquet")
    views = clearest_views(pts, s)
    y0, y1 = CROP_Y
    frames = sorted({int(v.frame_idx) for pair in views for v in pair})
    imgs = read_frames(cfg["video"], frames, (y0, y1), cfg["frame_size"][0])
    W, H = cfg["frame_size"][0], y1 - y0

    def crop(v):
        x1, x2 = int(max(0, v.x1 - MARGIN)), int(min(W, v.x2 + MARGIN))
        ya, yb = int(max(0, v.y1 - y0 - MARGIN)), int(min(H, v.y2 - y0 + MARGIN))
        c = imgs[int(v.frame_idx)][ya:yb, x1:x2]
        return cv2.resize(c, (int(c.shape[1] * CARD_H / c.shape[0]), CARD_H))[:, :310]

    cards = []
    for r, (a, b) in zip(s.itertuples(), views):
        card = np.full((CARD_H + 40, 640, 3), 30, np.uint8)
        ca, cb = crop(a), crop(b)
        card[35:35 + CARD_H, 5:5 + ca.shape[1]] = ca
        card[35:35 + CARD_H, 325:325 + cb.shape[1]] = cb
        cv2.putText(card, f"#{r.card}  A (gap {r.gap_s:.2f}s)", (8, 24), 0, 0.65, (255, 255, 255), 2)
        cv2.putText(card, "B" + ("  [flagged]" if r.flagged else ""), (330, 24), 0, 0.65, (255, 255, 255), 2)
        cards.append(card)
    blank = np.zeros_like(cards[0])
    rows = [np.hstack((cards[i:i + 3] + [blank] * 3)[:3]) for i in range(0, len(cards), 3)]
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def score(csv: Path) -> dict:
    s = pd.read_csv(csv)
    v = s.verdict.fillna("").str.strip().str.lower()
    if not v.isin(["same", "different"]).all():
        raise ValueError(f"{csv}: every verdict must be 'same' or 'different'")
    same = int((v == "same").sum())
    res = {"joins": len(s), "same": same, "different": len(s) - same, "precision": round(same / len(s), 3),
           "flagged_same": int(((v == "same") & s.flagged).sum()), "flagged_different": int(((v != "same") & s.flagged).sum()),
           "wrong_cards": s.card[v != "same"].tolist(), "wrong_gaps_s": s.gap_s[v != "same"].tolist()}
    print(json.dumps(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", help="summarise a filled-in audit CSV and exit")
    ap.add_argument("--cfg")
    ap.add_argument("--experiment")
    ap.add_argument("--variant")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.score:
        score(Path(a.score))
        return
    cfg = json.load(open(a.cfg))
    exp = Path(a.experiment)
    s = sample_links(exp, a.variant, a.n, a.seed)
    out = Path(a.out)
    render(cfg, exp, a.variant, s, out.with_suffix(".jpg"))
    s.to_csv(out.with_suffix(".csv"), index=False)
    print(f"{len(s)} joins -> {out.with_suffix('.jpg')} and {out.with_suffix('.csv')} (fill the verdict column)")


if __name__ == "__main__":
    main()
