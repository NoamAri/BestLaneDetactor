"""Detect + track vehicles in the road band, write points.parquet.

Usage:
    python -m lane_intel.extract --cfg configs/T8_D1.json --start 0 --duration 30 \
        --out data/interim/T8_D1_points.parquet
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from lane_intel.video import iter_frames


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(cfg: dict, start_s: float, duration_s: float | None, out: str) -> pd.DataFrame:
    fps = cfg["native_fps"]
    every = cfg.get("process_every", 1)
    proc_fps = fps / every
    y0, y1 = cfg["band_y"]
    width = cfg["frame_size"][0]
    device = pick_device()

    model = YOLO(cfg["model"])
    names = model.names
    tracker = ByteTrackTracker(
        lost_track_buffer=max(1, round(cfg["track_lost_buffer_s"] * 30)),  # scaled by frame_rate/30 inside
        frame_rate=proc_fps,
        track_activation_threshold=cfg.get("track_activation_threshold", 0.5),
        high_conf_det_threshold=cfg.get("track_high_conf", 0.5),
    )

    first = round(start_s * fps)
    rows = []
    t0 = time.perf_counter()
    n = 0
    for i, crop in iter_frames(cfg["video"], (y0, y1), width, start_s, duration_s):
        if i % every:
            continue
        frame_idx = first + i
        t = frame_idx / fps
        r = model.predict(crop, imgsz=cfg["imgsz"], classes=cfg["classes"], conf=cfg["conf"],
                          device=device, verbose=False)[0]
        det = sv.Detections.from_ultralytics(r)
        det = tracker.update(det, timestamp=t)
        det = det[det.tracker_id >= 0]
        for (x1, yy1, x2, yy2), tid, cid, conf in zip(det.xyxy, det.tracker_id, det.class_id, det.confidence):
            yy1 += y0
            yy2 += y0
            rows.append((t, frame_idx, int(tid), names[int(cid)], int(cid), float(conf),
                         float(x1), float(yy1), float(x2), float(yy2),
                         float((x1 + x2) / 2), float(yy2)))
        n += 1
        if n % 50 == 0:
            el = time.perf_counter() - t0
            print(f"frame {frame_idx}  t={t:7.2f}s  {n / el:.2f} fps  tracks so far={len({r[2] for r in rows})}",
                  flush=True)

    df = pd.DataFrame(rows, columns=["t", "frame_idx", "track_id", "cls", "cls_id", "conf",
                                     "x1", "y1", "x2", "y2", "x_px", "y_px"])
    # placeholders filled by later stages (stabilisation, homography, lanes)
    df["x_w"] = np.nan
    df["y_w"] = np.nan
    df["lane_frame"] = -1
    df.to_parquet(out, index=False)
    el = time.perf_counter() - t0
    print(f"done: {n} frames in {el:.0f}s ({n / el:.2f} fps), {len(df)} points, "
          f"{df.track_id.nunique()} tracks -> {out}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    run(cfg, a.start, a.duration, a.out)


if __name__ == "__main__":
    main()
