"""Stage 1 of the pipeline: detect vehicles in every frame. No tracking happens here.

The detector runs once per clip window and writes every box above a deliberately low
confidence floor (default 0.05). Tracking (lane_intel.tracking) then reads this file and
decides which boxes to use, so tracker experiments take seconds instead of rerunning YOLO.

Outputs
    <out>.parquet     one row per box: t, frame_idx, cls, cls_id, conf, x1, y1, x2, y2
                      (coordinates in full-frame pixels of the original video)
    <out>.meta.json   provenance: video, band, model, thresholds, versions, runtime

Usage
    python -m lane_intel.detect --cfg configs/T8_D1.json --start 0 --duration 30 \
        --out data/interim/T8_D1_0-30s_detections.parquet
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from lane_intel.video import iter_frames

COLUMNS = ["t", "frame_idx", "cls", "cls_id", "conf", "x1", "y1", "x2", "y2"]


def pick_device() -> str:
    """cuda if an NVIDIA GPU is usable, mps on Apple Silicon, otherwise cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def meta_path(out: str | Path) -> Path:
    """Sidecar provenance file that sits next to a detections parquet."""
    out = Path(out)
    return out.with_name(out.stem + ".meta.json")


def run(cfg: dict, start_s: float, duration_s: float | None, out: str | Path) -> pd.DataFrame:
    det = cfg["detector"]
    fps = cfg["native_fps"]
    every = det.get("process_every", 1)
    y0, y1 = cfg["band_y"]
    width = cfg["frame_size"][0]
    device = pick_device()

    model = YOLO(det["model"])
    names = model.names
    first = round(start_s * fps)

    rows: list[tuple] = []
    t_start = time.perf_counter()
    n = 0
    for i, crop in iter_frames(cfg["video"], (y0, y1), width, start_s, duration_s):
        if i % every:
            continue
        frame_idx = first + i
        t = frame_idx / fps
        r = model.predict(crop, imgsz=det["imgsz"], classes=det["classes"], conf=det["conf_min"],
                          max_det=det.get("max_det", 1000), device=device, verbose=False)[0]
        boxes = r.boxes
        for (x1, yy1, x2, yy2), cid, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(),
                                                 boxes.conf.cpu().numpy()):
            rows.append((t, frame_idx, names[int(cid)], int(cid), float(conf),
                         float(x1), float(yy1 + y0), float(x2), float(yy2 + y0)))
        n += 1
        if n % 50 == 0:
            el = time.perf_counter() - t_start
            print(f"frame {frame_idx}  t={t:7.2f}s  {n / el:.2f} fps  boxes so far={len(rows)}", flush=True)

    df = pd.DataFrame(rows, columns=COLUMNS)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    elapsed = time.perf_counter() - t_start
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": cfg["video"], "start_s": start_s, "duration_s": duration_s,
        "frame_first": first, "frames": n, "fps": fps, "band_y": [y0, y1],
        "detector": det, "device": device, "seconds": round(elapsed, 1),
        "boxes": len(df), "boxes_conf_ge_0_25": int((df.conf >= 0.25).sum()),
        "versions": {p: metadata.version(p) for p in ("ultralytics", "torch")},
        "python": platform.python_version(),
    }
    meta_path(out).write_text(json.dumps(provenance, indent=2))
    print(f"done: {n} frames in {elapsed:.0f}s ({n / elapsed:.2f} fps), {len(df)} boxes "
          f"({provenance['boxes_conf_ge_0_25']} with conf >= 0.25) -> {out}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(json.load(open(a.cfg)), a.start, a.duration, a.out)


if __name__ == "__main__":
    main()
