"""From boxes to clean vehicle positions on the road (notes/lane_speed_plan.md §2).

ground_point   the image point that stands for the vehicle's position: the box centre, moved into the
               stabilised reference frame. (Amendment A2: the pre-registered "bottom + half width"
               shift misplaced vehicles in the ground-truth check; see notes/lane_speed.md.) The shift
               is kept as an option for the ablation table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lane_intel import stabilize


def ground_point(pts: pd.DataFrame, stab: pd.DataFrame | None, mode: str = "box_centre",
                 shift_px: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of each box's ground point in reference-frame pixels.

    mode "box_centre"    box centre (default, amendment A2)
         "bottom_shift"  bottom-centre moved up by shift_px (half the vehicle width in pixels, per box)
    """
    x = pts.x_px.to_numpy(float)
    if mode == "box_centre":
        y = pts.y_px.to_numpy(float)
    elif mode == "bottom_shift":
        y = pts.y2.to_numpy(float) - (0.0 if shift_px is None else np.asarray(shift_px, float))
    else:
        raise ValueError(f"unknown ground point mode {mode!r}")
    if stab is None:
        return x, y
    return stabilize.apply(stab, pts.frame_idx.to_numpy(), x, y)


# --------------------------------------------------------------------------- per-camera ground table
TRACK_ID_STRIDE = 100_000      # track ids are made unique across windows: window index * stride + id


def window_tag(cfg_path: str | Path, start: float, dur: float) -> str:
    return f"{Path(cfg_path).stem}_{start:g}-{dur:g}s"


def ground_path(interim: str | Path, cfg_path: str | Path) -> Path:
    return Path(interim) / f"{Path(cfg_path).stem}_ground.parquet"


def build_ground(cfg: dict, cfg_path: str | Path, interim: str | Path = "data/interim") -> pd.DataFrame:
    """All analysis windows of a camera in one table, in the stabilised reference frame.

    Columns: those of the points files, plus window, track_local (the id within its window),
    x_raw / y_raw (box centre in the video frame). x_px / y_px become the stabilised ground point
    and track_id is unique across windows. A frame shared by two windows is kept once (the earlier).
    A vehicle crossing a window boundary gets two ids; per-lane speeds are measured so that this
    does not matter (Edie sums, notes/lane_speed_plan.md §3).
    """
    parts, last = [], -1
    for k, (start, dur) in enumerate(cfg["lane_speed"]["windows"]):
        tag = window_tag(cfg_path, start, dur)
        pts = pd.read_parquet(Path(interim) / f"{tag}_points.parquet")
        sp = Path(interim) / f"{tag}_stab.parquet"
        if not sp.exists():
            raise FileNotFoundError(f"{sp} missing: run  python run.py stabilize --cfg {cfg_path} --start {start:g} --dur {dur:g}")
        pts = pts[pts.frame_idx > last]
        if pts.empty:
            continue
        x, y = ground_point(pts, pd.read_parquet(sp))
        parts.append(pts.assign(window=tag, track_local=pts.track_id, track_id=k * TRACK_ID_STRIDE + pts.track_id,
                                x_raw=pts.x_px, y_raw=pts.y_px, x_px=x, y_px=y))
        last = int(pts.frame_idx.max())
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Build the per-camera ground-point table (all analysis windows).")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--interim", default="data/interim")
    a = ap.parse_args()
    cfg = json.loads(Path(a.cfg).read_text())
    g = build_ground(cfg, a.cfg, a.interim)
    out = ground_path(a.interim, a.cfg)
    g.to_parquet(out, index=False)
    print(f"{len(g)} boxes, {g.track_id.nunique()} tracks, {g.window.nunique()} windows, "
          f"{g.t.min():.2f}-{g.t.max():.2f} s -> {out}")


if __name__ == "__main__":
    main()
