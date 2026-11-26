"""Run one tracker variant over saved detections.

Input   detections DataFrame from lane_intel.detect (t, frame_idx, cls, cls_id, conf, x1..y2)
Output  points DataFrame, one row per box that carries a confirmed track ID:
        t, frame_idx, track_id, cls, cls_id, conf, x1, y1, x2, y2, x_px, y_px, x_w, y_w, lane_frame
        (same schema the viewer and later stages already read)
        plus a stats dict (boxes in, boxes fed, boxes tracked, coverage).

The tracker is updated on every frame of the window, including frames where no box passes
``min_conf``, so a lost vehicle's memory ages at the true rate.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import supervision as sv

from lane_intel.tracking.config import TrackerConfig, build_tracker

POINT_COLUMNS = ["t", "frame_idx", "track_id", "cls", "cls_id", "conf",
                 "x1", "y1", "x2", "y2", "x_px", "y_px", "x_w", "y_w", "lane_frame"]

# Boxes at or above this confidence are "clearly detected"; a good tracker should give
# almost all of them an ID. Used for the coverage guard in quality.evaluate.
COVERAGE_CONF = 0.5


def run_tracker(dets: pd.DataFrame, cfg: TrackerConfig, fps: float,
                frame_range: tuple[int, int] | None = None) -> tuple[pd.DataFrame, dict]:
    """Track ``dets`` with variant ``cfg``.

    frame_range: inclusive (first, last) frame of the window. Defaults to the span of the
    detections; pass it explicitly when the window may start or end with empty frames.
    """
    t0 = time.perf_counter()
    dets = dets.reset_index(drop=True)
    if frame_range is None:
        frame_range = (int(dets.frame_idx.min()), int(dets.frame_idx.max()))
    f_first, f_last = frame_range

    fed = dets[dets.conf >= cfg.min_conf]
    by_frame = {f: g for f, g in fed.groupby("frame_idx", sort=True)}
    tracker = build_tracker(cfg, fps)

    assigned_rows: list[np.ndarray] = []
    assigned_ids: list[np.ndarray] = []
    for f in range(f_first, f_last + 1):
        g = by_frame.get(f)
        if g is None or g.empty:
            det = sv.Detections.empty()
        else:
            det = sv.Detections(
                xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                confidence=g.conf.to_numpy(np.float32),
                class_id=g.cls_id.to_numpy(int),
                data={"row": g.index.to_numpy()},
            )
        out = tracker.update(det, timestamp=f / fps)
        if len(out):
            keep = out.tracker_id >= 0
            if keep.any():
                assigned_rows.append(np.asarray(out.data["row"])[keep])
                assigned_ids.append(out.tracker_id[keep])

    rows = np.concatenate(assigned_rows) if assigned_rows else np.array([], int)
    ids = np.concatenate(assigned_ids) if assigned_ids else np.array([], int)
    pts = dets.loc[rows].copy()
    pts.insert(2, "track_id", ids.astype(int))
    pts["x_px"] = (pts.x1 + pts.x2) / 2          # bottom-centre: where the tyres meet the road
    pts["y_px"] = pts.y2
    pts["x_w"] = np.nan                         # filled by calibration (phase 2)
    pts["y_w"] = np.nan
    pts["lane_frame"] = -1                      # filled by lane polygons (phase 2)
    pts = pts[POINT_COLUMNS].sort_values(["frame_idx", "track_id"]).reset_index(drop=True)

    clear = dets.conf >= COVERAGE_CONF
    clear_tracked = np.isin(dets.index[clear], rows).sum()
    stats = {
        "frames": f_last - f_first + 1,
        "boxes_in": int(len(dets)),
        "boxes_fed": int(len(fed)),
        "boxes_tracked": int(len(pts)),
        "clear_boxes": int(clear.sum()),
        "coverage_clear": round(float(clear_tracked / max(1, clear.sum())), 4),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    return pts, stats
