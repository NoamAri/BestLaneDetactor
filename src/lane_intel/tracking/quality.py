"""Measure how well vehicles keep their ID, without ground truth.

Definitions (all positions are the bottom-centre anchor in full-frame pixels):

ids                 number of distinct track IDs in the window.
short_lt_1s         tracks shorter than ``short_s``: mostly flicker and duplicate boxes.
born_mid            tracks whose first observation is inside the image (more than ``edge_px``
                    from the left/right border) and after the window start. On this road
                    vehicles only enter at the image edges, so a mid-image start is a vehicle
                    that already existed under another ID: a break.
died_mid            the same for the last observation: a track that ends mid-image.
fragmentation       born_mid + died_mid. The primary metric; lower is better.
restart_pairs       one-to-one pairs (died_mid track -> later born_mid track) that look like the
                    same vehicle: gap <= ``restart_max_gap_s``, same lane band (|dy| <=
                    ``restart_dy_px``), restart where the old track's speed predicts (|dx| <=
                    ``restart_dx_base_px`` + ``restart_dx_per_s`` * gap). An estimate of the
                    breaks an offline stitcher (step 3) could repair.
identity_jumps      times a track's box jumps to a different vehicle: between two consecutive
                    observations BOTH edges of the box move, in the same direction, by more than
                    ``jump_frac`` of the box size beyond what the track's recent speed predicts
                    (plus ``jump_slack_px_per_s`` per second of gap). Requiring both edges ignores
                    a box that grows or shrinks when a vehicle is partly hidden. A guard metric:
                    settings that join tracks more eagerly must not raise it.
                    Split into identity_jumps_within (both observations belong to the same
                    tracker piece, i.e. the same raw_track_id) and junction_jumps (the step
                    crosses a stitch junction). The guard uses identity_jumps_within: the check
                    was validated on tracker swaps, and audits showed it misreads correct stitch
                    junctions (a car reappearing from behind a pole grows from half a box to a
                    whole box). Stitch junctions are judged by a visual audit (audit.py).
coverage_clear      share of clearly detected boxes (conf >= 0.5) that end up with a track ID
                    (from runner stats). A guard metric against "improving" by dropping boxes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class QualityParams:
    edge_px: float = 120.0
    window_margin_frames: int = 2
    short_s: float = 1.0
    velocity_window_s: float = 0.5
    restart_max_gap_s: float = 5.0
    restart_dy_px: float = 18.0
    restart_dx_base_px: float = 60.0
    restart_dx_per_s: float = 40.0
    jump_frac: float = 0.5
    jump_slack_px_per_s: float = 15.0

    def to_dict(self) -> dict:
        return asdict(self)


def _velocity(t: np.ndarray, x: np.ndarray) -> float:
    """Least-squares slope of x over t (px/s); 0 when fewer than 3 observations."""
    if len(t) < 3 or np.ptp(t) == 0:
        return 0.0
    return float(np.polyfit(t, x, 1)[0])


def track_table(points: pd.DataFrame, fps: float, frame_range: tuple[int, int], width: int,
                params: QualityParams = QualityParams()) -> pd.DataFrame:
    """One row per track: lifetime, start/end anchor, speed at both ends, break flags."""
    f_first, f_last = frame_range
    rows = []
    for tid, g in points.sort_values("frame_idx").groupby("track_id", sort=True):
        t = g.t.to_numpy()
        x = g.x_px.to_numpy()
        head = t <= t[0] + params.velocity_window_s
        tail = t >= t[-1] - params.velocity_window_s
        rows.append({
            "track_id": int(tid),
            "cls": g.cls.mode().iloc[0],
            "first": int(g.frame_idx.iloc[0]), "last": int(g.frame_idx.iloc[-1]), "n": len(g),
            "x_first": float(x[0]), "y_first": float(g.y_px.iloc[0]),
            "x_last": float(x[-1]), "y_last": float(g.y_px.iloc[-1]),
            "w": float((g.x2 - g.x1).median()), "h": float((g.y2 - g.y1).median()),
            "v_start": _velocity(t[head], x[head]), "v_end": _velocity(t[tail], x[tail]),
        })
    tr = pd.DataFrame(rows)
    if tr.empty:
        return tr
    tr["span_s"] = (tr["last"] - tr["first"]) / fps
    inside = lambda xs: (xs > params.edge_px) & (xs < width - params.edge_px)  # noqa: E731
    tr["born_mid"] = (tr["first"] > f_first + params.window_margin_frames) & inside(tr.x_first)
    tr["died_mid"] = (tr["last"] < f_last - params.window_margin_frames) & inside(tr.x_last)
    return tr


def restart_pairs(tracks: pd.DataFrame, fps: float, params: QualityParams = QualityParams()) -> pd.DataFrame:
    """Best one-to-one matching of mid-image track ends to later mid-image track starts."""
    cols = ["track_end", "track_start", "gap_s", "dx_err_px", "dy_px", "x", "y"]
    if tracks.empty:
        return pd.DataFrame(columns=cols)
    A = tracks[tracks.died_mid].reset_index(drop=True)
    B = tracks[tracks.born_mid].reset_index(drop=True)
    if A.empty or B.empty:
        return pd.DataFrame(columns=cols)
    gap = (B["first"].to_numpy()[None, :] - A["last"].to_numpy()[:, None]) / fps
    x_pred = A.x_last.to_numpy()[:, None] + A.v_end.to_numpy()[:, None] * gap
    dx = np.abs(B.x_first.to_numpy()[None, :] - x_pred)
    dy = np.abs(B.y_first.to_numpy()[None, :] - A.y_last.to_numpy()[:, None])
    tol = params.restart_dx_base_px + params.restart_dx_per_s * gap
    ok = (gap > 0) & (gap <= params.restart_max_gap_s) & (dy <= params.restart_dy_px) & (dx <= tol)
    BIG = 1e6
    cost = np.where(ok, dx / np.where(ok, tol, 1) + gap / params.restart_max_gap_s, BIG)
    ia, ib = linear_sum_assignment(cost)
    keep = ok[ia, ib]
    ia, ib = ia[keep], ib[keep]
    return pd.DataFrame({
        "track_end": A.track_id.to_numpy()[ia], "track_start": B.track_id.to_numpy()[ib],
        "gap_s": gap[ia, ib].round(3), "dx_err_px": dx[ia, ib].round(1), "dy_px": dy[ia, ib].round(1),
        "x": A.x_last.to_numpy()[ia].round(0), "y": A.y_last.to_numpy()[ia].round(0),
    }, columns=cols)


def identity_jumps(points: pd.DataFrame, fps: float, params: QualityParams = QualityParams()) -> pd.DataFrame:
    """Observations where a track's whole box jumps further than its motion allows."""
    events = []
    for tid, g in points.sort_values("frame_idx").groupby("track_id", sort=True):
        if len(g) < 2:
            continue
        f = g.frame_idx.to_numpy()
        t = g.t.to_numpy()
        raw = g.raw_track_id.to_numpy() if "raw_track_id" in g else np.full(len(g), tid)
        x1, x2, y1, y2 = (g[c].to_numpy() for c in ("x1", "x2", "y1", "y2"))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        starts = np.searchsorted(t, t - params.velocity_window_s)
        for i in range(1, len(g)):
            j0 = min(starts[i - 1], i - 1)            # earliest observation in the speed window
            span = t[i - 1] - t[j0]
            vx = (cx[i - 1] - cx[j0]) / span if span > 0 else 0.0
            vy = (cy[i - 1] - cy[j0]) / span if span > 0 else 0.0
            dt = t[i] - t[i - 1]
            pdx, pdy = vx * dt, vy * dt
            tol_x = params.jump_frac * (x2[i - 1] - x1[i - 1]) + params.jump_slack_px_per_s * dt
            tol_y = params.jump_frac * (y2[i - 1] - y1[i - 1]) + params.jump_slack_px_per_s * dt
            ex1, ex2 = x1[i] - x1[i - 1] - pdx, x2[i] - x2[i - 1] - pdx
            ey1, ey2 = y1[i] - y1[i - 1] - pdy, y2[i] - y2[i - 1] - pdy
            jx = np.sign(ex1) == np.sign(ex2) and min(abs(ex1), abs(ex2)) > tol_x
            jy = np.sign(ey1) == np.sign(ey2) and min(abs(ey1), abs(ey2)) > tol_y
            if jx or jy:
                events.append({"track_id": int(tid), "frame_idx": int(f[i]), "gap_frames": int(f[i] - f[i - 1]),
                               "junction": bool(raw[i] != raw[i - 1]), "axis": "x" if jx else "y", "dx_err_px": round(float((ex1 + ex2) / 2), 1),
                               "dy_err_px": round(float((ey1 + ey2) / 2), 1),
                               "x": round(float(cx[i]), 0), "y": round(float(y2[i]), 0)})
    return pd.DataFrame(events, columns=["track_id", "frame_idx", "gap_frames", "junction", "axis",
                                         "dx_err_px", "dy_err_px", "x", "y"])


def evaluate(points: pd.DataFrame, fps: float, frame_range: tuple[int, int], width: int,
             stats: dict | None = None, params: QualityParams = QualityParams()) -> tuple[dict, dict]:
    """Summary metrics plus the per-track, restart and jump tables behind them."""
    tracks = track_table(points, fps, frame_range, width, params)
    restarts = restart_pairs(tracks, fps, params)
    jumps = identity_jumps(points, fps, params)
    if tracks.empty:
        summary = {"ids": 0}
    else:
        summary = {
            "ids": int(len(tracks)),
            "ids_ge_1s": int((tracks.span_s >= params.short_s).sum()),
            "short_lt_1s": int((tracks.span_s < params.short_s).sum()),
            "born_mid": int(tracks.born_mid.sum()),
            "died_mid": int(tracks.died_mid.sum()),
            "fragmentation": int(tracks.born_mid.sum() + tracks.died_mid.sum()),
            "restart_pairs": int(len(restarts)),
            "identity_jumps": int(len(jumps)),
            "identity_jumps_within": int((~jumps.junction.astype(bool)).sum()) if len(jumps) else 0,
            "junction_jumps": int(jumps.junction.astype(bool).sum()) if len(jumps) else 0,
            "tracks_with_jumps": int(jumps.track_id.nunique()) if len(jumps) else 0,
            "median_span_s": round(float(tracks.span_s.median()), 2),
            "mean_span_s": round(float(tracks.span_s.mean()), 2),
            "tags": int(len(points)),
        }
    if stats:
        summary["coverage_clear"] = stats["coverage_clear"]
        summary["track_seconds"] = stats["seconds"]
    return summary, {"tracks": tracks, "restarts": restarts, "jumps": jumps}
