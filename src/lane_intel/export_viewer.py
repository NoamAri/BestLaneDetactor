"""Bundle a browser-friendly video + per-frame tags for app/viewer/index.html.

Writes into --out-dir:
    video.mp4        road-strip proxy (short GOP so frame stepping is snappy)
    detections.json  one array, one record per line:
                     [frame, track_id, cls_idx, conf_pct, x1, y1, x2, y2]  (full-frame px)
    meta.json        fps, frame range, crop/scale mapping, class names, track summary
    index.html       the viewer, wrapped in a document skeleton for local serving

Serve locally with:  python -m http.server -d <out-dir> 8000
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

VIEWER_SRC = Path(__file__).resolve().parents[2] / "app" / "viewer" / "index.html"


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


def track_summary(df: pd.DataFrame, fps: float) -> list[dict]:
    g = df.groupby("track_id")
    s = pd.DataFrame({
        "first": g.frame_idx.min(), "last": g.frame_idx.max(), "n": g.size(),
        "cls": g.cls.agg(lambda c: c.value_counts().index[0]),
        "conf": g.conf.mean().round(3),
        "x_first": g.x_px.first(), "x_last": g.x_px.last(),
        "y_first": g.y_px.first(), "y_last": g.y_px.last(),
    }).reset_index()
    span = (s["last"] - s["first"]) / fps
    s["span_s"] = span.round(2)
    dx = s.x_last - s.x_first
    s["px_per_s"] = np.where(span > 0, dx / span.replace(0, np.nan), 0.0).round(1)
    return s.drop(columns=["x_first", "x_last", "y_first", "y_last"]).to_dict("records")


def build_bundle(cfg: dict, df: pd.DataFrame, out_dir: str | Path, start: float, duration: float,
                 crop_y: tuple[int, int], width: int, crf: int = 32) -> dict:
    """Write video.mp4, detections.json, meta.json and index.html into out_dir; return meta."""
    fps = cfg["native_fps"]
    W = cfg["frame_size"][0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    crop = (0, crop_y[0], W, crop_y[1] - crop_y[0])
    vw, vh = encode_proxy(cfg["video"], out / "video.mp4", start, duration, crop, width, crf)

    f0 = round(start * fps)
    f1 = f0 + round(duration * fps)
    df = df[(df.frame_idx >= f0) & (df.frame_idx < f1)].copy()
    classes = sorted(df.cls.unique())
    cidx = {c: i for i, c in enumerate(classes)}

    with open(out / "detections.json", "w") as fh:
        fh.write("[\n")
        recs = df.sort_values(["frame_idx", "track_id"])
        lines = [json.dumps([int(r.frame_idx), int(r.track_id), cidx[r.cls], int(round(r.conf * 100)),
                             int(round(r.x1)), int(round(r.y1)), int(round(r.x2)), int(round(r.y2))],
                            separators=(",", ":"))
                 for r in recs.itertuples()]
        fh.write(",\n".join(lines))
        fh.write("\n]\n")

    meta = {
        "flight": cfg["flight"], "drone": cfg["drone"],
        "fps": fps, "clip_offset_s": cfg["clip_offset_s"],
        "frame_first": f0, "frame_count": f1 - f0,
        "frame_size": cfg["frame_size"],
        "crop": {"x": crop[0], "y": crop[1], "w": crop[2], "h": crop[3]},
        "video_size": [vw, vh],
        "classes": classes,
        "model": Path(cfg["model"]).name, "imgsz": cfg["imgsz"], "conf_min": cfg["conf"],
        "tracks": track_summary(df, fps),
    }
    json.dump(meta, open(out / "meta.json", "w"))

    if VIEWER_SRC.exists():
        body = VIEWER_SRC.read_text()
        (out / "index.html").write_text(
            "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
            "</head><body>\n" + body + "\n</body></html>\n")
    print(f"viewer bundle -> {out}  ({len(df)} tags, {df.track_id.nunique()} tracks, video {vw}x{vh})")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--crop-y", type=int, nargs=2, default=[1040, 1660])
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--crf", type=int, default=32)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    build_bundle(cfg, pd.read_parquet(a.points), a.out_dir, a.start, a.duration,
                 tuple(a.crop_y), a.width, a.crf)


if __name__ == "__main__":
    main()
