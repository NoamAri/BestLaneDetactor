"""Bundle a browser-friendly video + per-frame tags for app/viewer/index.html.

One bundle can carry several tracker variants of the same window so the viewer can switch
between them (for example the baseline and the tuned tracker).

Writes into --out-dir:
    video.mp4            road-strip proxy (short GOP so frame stepping is snappy)
    tags_<variant>.json  one array, one record per line:
                         [frame, track_id, cls_idx, conf_pct, x1, y1, x2, y2]  (full-frame px)
    meta.json            fps, frame range, crop/scale mapping, class names, and per variant:
                         label, note, quality summary, one summary row per track
    index.html           the viewer, wrapped in a document skeleton for local serving

Usage
    # variants from a tracking experiment (points_<name>.parquet + report.json in --experiment)
    python -m lane_intel.export_viewer --cfg configs/T8_D1.json --duration 30 \
        --experiment data/out/tracking/T8_D1_0-30s --variants baseline,s1_lowconf,s2_bt_buf4s \
        --out-dir data/out/viewer
    # or a single points file
    python -m lane_intel.export_viewer --cfg configs/T8_D1.json --duration 30 \
        --points data/interim/T8_D1_0-30s_points.parquet --out-dir data/out/viewer

Serve locally with:  python -m lane_intel.serve <out-dir> --port 8000  (needs HTTP Range for seeking)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lane_intel.tracking.quality import QualityParams, track_table

VIEWER_SRC = Path(__file__).resolve().parents[2] / "app" / "viewer" / "index.html"


@dataclass
class ViewerVariant:
    name: str
    points: pd.DataFrame
    label: str = ""
    note: str = ""
    quality: dict = field(default_factory=dict)


def encode_proxy(video: str, out: Path, start_s: float, duration_s: float,
                 crop: tuple[int, int, int, int], out_w: int, crf: int) -> tuple[int, int]:
    x0, y0, w, h = crop
    out_h = int(round(h * out_w / w / 2)) * 2
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start_s:
        cmd += ["-ss", f"{start_s:.6f}"]
    cmd += ["-i", video, "-t", f"{duration_s:.6f}",
            "-vf", f"crop={w}:{h}:{x0}:{y0},scale={out_w}:{out_h}",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-g", "10",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out_w, out_h


def track_summary(df: pd.DataFrame, fps: float, frame_range: tuple[int, int] | None = None,
                  width: int = 3840, params: QualityParams = QualityParams()) -> list[dict]:
    """One record per track for the viewer's register and start/end markers.

    Start/end flags come from tracking.quality.track_table, so the viewer shows exactly the
    breaks the quality metrics count. px_per_s is the average horizontal image motion over
    the whole track (not km/h: that needs calibration).
    """
    if df.empty:
        return []
    if frame_range is None:
        frame_range = (int(df.frame_idx.min()), int(df.frame_idx.max()))
    tr = track_table(df, fps, frame_range, width, params)
    conf = df.groupby("track_id").conf.mean()
    span = tr.span_s.to_numpy()
    px_per_s = np.where(span > 0, (tr.x_last - tr.x_first) / np.where(span > 0, span, 1), 0.0)
    out = pd.DataFrame({
        "track_id": tr.track_id, "cls": tr.cls, "first": tr["first"], "last": tr["last"], "n": tr.n,
        "conf": conf.loc[tr.track_id].round(3).to_numpy(), "span_s": tr.span_s.round(2),
        "px_per_s": np.round(px_per_s, 1),
        "born_mid": tr.born_mid.astype(bool), "died_mid": tr.died_mid.astype(bool),
        "x_first": tr.x_first.round().astype(int), "y_first": tr.y_first.round().astype(int),
        "x_last": tr.x_last.round().astype(int), "y_last": tr.y_last.round().astype(int),
    })
    return json.loads(out.to_json(orient="records"))


def write_tags(df: pd.DataFrame, path: Path, cidx: dict[str, int]) -> None:
    lines = [json.dumps([int(r.frame_idx), int(r.track_id), cidx[r.cls], int(round(r.conf * 100)),
                         int(round(r.x1)), int(round(r.y1)), int(round(r.x2)), int(round(r.y2))],
                        separators=(",", ":"))
             for r in df.sort_values(["frame_idx", "track_id"]).itertuples()]
    path.write_text("[\n" + ",\n".join(lines) + "\n]\n")


def wrap_page(fragment: str) -> str:
    """Page fragment (title/meta/links, then <style>, markup, script) -> complete document."""
    head, sep, rest = fragment.partition("<style>")
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + head.strip() + "\n</head><body>\n" + sep + rest + "\n</body></html>\n")


def build_bundle(cfg: dict, variants: list[ViewerVariant], out_dir: str | Path, start: float,
                 duration: float, crop_y: tuple[int, int], width: int, crf: int = 32) -> dict:
    """Write video.mp4, tags_<variant>.json, meta.json and index.html; return meta.

    The first variant is the one the viewer opens with.
    """
    if not variants:
        raise ValueError("need at least one variant")
    fps = cfg["native_fps"]
    W = cfg["frame_size"][0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("tags_*.json"):
        old.unlink()

    crop = (0, crop_y[0], W, crop_y[1] - crop_y[0])
    vw, vh = encode_proxy(cfg["video"], out / "video.mp4", start, duration, crop, width, crf)

    f0 = round(start * fps)
    f1 = f0 + round(duration * fps)
    windowed = {v.name: v.points[(v.points.frame_idx >= f0) & (v.points.frame_idx < f1)] for v in variants}
    classes = sorted(set().union(*(set(p.cls.unique()) for p in windowed.values())))
    cidx = {c: i for i, c in enumerate(classes)}

    meta_variants = []
    for v in variants:
        df = windowed[v.name]
        fname = f"tags_{v.name}.json"
        write_tags(df, out / fname, cidx)
        meta_variants.append({
            "name": v.name, "label": v.label or v.name, "note": v.note, "file": fname,
            "quality": v.quality, "tags": int(len(df)),
            "tracks": track_summary(df, fps, (f0, f1 - 1), W),
        })
        print(f"  {v.name}: {len(df)} tags, {df.track_id.nunique()} tracks -> {fname}")

    det = cfg.get("detector", {})
    meta = {
        "flight": cfg["flight"], "drone": cfg["drone"],
        "fps": fps, "clip_offset_s": cfg["clip_offset_s"],
        "frame_first": f0, "frame_count": f1 - f0,
        "frame_size": cfg["frame_size"],
        "crop": {"x": crop[0], "y": crop[1], "w": crop[2], "h": crop[3]},
        "video_size": [vw, vh],
        "classes": classes,
        "model": Path(det.get("model", "?")).name, "imgsz": det.get("imgsz"),
        "edge_px": QualityParams().edge_px,
        "variants": meta_variants,
    }
    (out / "meta.json").write_text(json.dumps(meta))

    if VIEWER_SRC.exists():
        (out / "index.html").write_text(wrap_page(VIEWER_SRC.read_text()))
    print(f"viewer bundle -> {out}  ({len(variants)} variant(s), video {vw}x{vh})")
    return meta


def variants_from_experiment(exp_dir: str | Path, names: list[str]) -> list[ViewerVariant]:
    """Load points_<name>.parquet and labels/quality from an experiment's report.json.

    The name '@selected' stands for the variant the experiment's selection rule picked.
    """
    exp_dir = Path(exp_dir)
    report = json.loads((exp_dir / "report.json").read_text())
    out, seen = [], set()
    for n in names:
        n = report["selected"] if n == "@selected" else n
        if n is None or n in seen:
            continue
        seen.add(n)
        info = report["variants"][n]
        out.append(ViewerVariant(name=n, points=pd.read_parquet(exp_dir / f"points_{n}.parquet"),
                                 label=info["config"]["label"], note=info["config"]["note"],
                                 quality={**info["summary"], "selected": n == report["selected"],
                                          "guard_failures": info["guard_failures"]}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--points", help="a single points parquet")
    src.add_argument("--experiment", help="a tracking experiment folder (report.json + points_*.parquet)")
    ap.add_argument("--variants", default="", help="with --experiment: comma-separated names, '@selected' allowed")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--crop-y", type=int, nargs=2, default=[1040, 1660])
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--crf", type=int, default=32)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    if a.points:
        variants = [ViewerVariant(name="points", points=pd.read_parquet(a.points), label=Path(a.points).stem)]
    else:
        names = [n.strip() for n in a.variants.split(",") if n.strip()] or ["@selected"]
        variants = variants_from_experiment(a.experiment, names)
    build_bundle(cfg, variants, a.out_dir, a.start, a.duration, tuple(a.crop_y), a.width, a.crf)


if __name__ == "__main__":
    main()
