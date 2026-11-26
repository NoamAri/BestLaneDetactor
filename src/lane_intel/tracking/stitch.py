"""Step 3: offline stitching. Join a track that ends mid-image to the track that continues it.

An online tracker cannot bridge a vehicle that is fully hidden (under the trees, behind a
truck): when it reappears it gets a new ID. Because we process recorded video, we can look
forward in time and join the two pieces afterwards.

A link "track A -> track B" is allowed only when all of these hold:
    A ends mid-image and B starts mid-image   (on this road vehicles only enter and leave
                                              at the left/right edges; see quality.py)
    0 < gap <= max_gap_s                      B starts after A ends, within the allowed gap
    |x_B - x_predicted| <= x_base_frac * width_A + x_per_s * gap
                                              B starts where A's speed at its end predicts
    |top_B - top_A| <= y_px + y_per_s * gap   same lane band; the TOP edge of the box is used
                                              because trees and nearer trucks hide the lower
                                              part of a car, which moves the bottom edge
    max(width_A / width_B, width_B / width_A) <= width_ratio_max
                                              similar vehicle size
    not moving in opposite directions         when both ends move faster than speed_px_s

Each track end can join at most one track start and vice versa. The pairs are chosen together
with an optimal assignment (Hungarian method) minimising the sum of normalised errors. Links
chain naturally (A -> B -> C), and every chain becomes one vehicle whose ID is the ID of its
first piece. The original IDs stay in ``raw_track_id``. Boxes are never invented for the hidden
stretch: the vehicle simply has no box while it is hidden.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from lane_intel.tracking.quality import QualityParams, track_table

LINK_COLUMNS = ["from_track", "to_track", "frame_end", "frame_start", "gap_s", "dx_px", "tol_x_px",
                "dy_px", "tol_y_px", "width_ratio", "cost", "x", "y"]


@dataclass(frozen=True)
class StitchParams:
    max_gap_s: float = 5.0
    x_base_frac: float = 0.6
    x_per_s: float = 25.0
    y_px: float = 15.0
    y_per_s: float = 3.0
    width_ratio_max: float = 1.6
    speed_px_s: float = 30.0
    edge_px: float = 120.0
    window_margin_frames: int = 2
    ends_n: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def keys(cls) -> set[str]:
        return {f.name for f in fields(cls)}


def endpoint_table(points: pd.DataFrame, fps: float, frame_range: tuple[int, int], width: int,
                   sp: StitchParams) -> pd.DataFrame:
    """track_table plus the top edge of the box at each end (median of ``ends_n`` observations)."""
    qp = QualityParams(edge_px=sp.edge_px, window_margin_frames=sp.window_margin_frames)
    tr = track_table(points, fps, frame_range, width, qp)
    if tr.empty:
        return tr
    g = points.sort_values("frame_idx").groupby("track_id").y1
    top_first = g.apply(lambda s: s.head(sp.ends_n).median())
    top_last = g.apply(lambda s: s.tail(sp.ends_n).median())
    tr["top_first"] = top_first.loc[tr.track_id].to_numpy()
    tr["top_last"] = top_last.loc[tr.track_id].to_numpy()
    return tr


def choose_links(tr: pd.DataFrame, fps: float, sp: StitchParams) -> pd.DataFrame:
    """Feasible track-end -> track-start pairs, then the optimal one-to-one choice among them."""
    A = tr[tr.died_mid].reset_index(drop=True)
    B = tr[tr.born_mid].reset_index(drop=True)
    if A.empty or B.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)
    col = lambda df, c: df[c].to_numpy(float)  # noqa: E731
    gap = (col(B, "first")[None, :] - col(A, "last")[:, None]) / fps
    gpos = np.clip(gap, 0, None)
    x_pred = col(A, "x_last")[:, None] + col(A, "v_end")[:, None] * gpos
    dx = np.abs(col(B, "x_first")[None, :] - x_pred)
    tol_x = sp.x_base_frac * col(A, "w")[:, None] + sp.x_per_s * gpos
    dy = np.abs(col(B, "top_first")[None, :] - col(A, "top_last")[:, None])
    tol_y = sp.y_px + sp.y_per_s * gpos
    wa, wb = col(A, "w")[:, None], col(B, "w")[None, :]
    ratio = np.maximum(wa / np.maximum(wb, 1), wb / np.maximum(wa, 1))
    va, vb = col(A, "v_end")[:, None], col(B, "v_start")[None, :]
    opposite = (np.abs(va) > sp.speed_px_s) & (np.abs(vb) > sp.speed_px_s) & (np.sign(va) != np.sign(vb))
    same = col(A, "track_id")[:, None] == col(B, "track_id")[None, :]
    ok = ((gap > 0) & (gap <= sp.max_gap_s) & (dx <= tol_x) & (dy <= tol_y)
          & (ratio <= sp.width_ratio_max) & ~opposite & ~same)
    if not ok.any():
        return pd.DataFrame(columns=LINK_COLUMNS)
    cost = dx / tol_x + dy / tol_y + gap / sp.max_gap_s + np.log(ratio) / np.log(sp.width_ratio_max)
    ia, ib = linear_sum_assignment(np.where(ok, cost, 1e6))
    keep = ok[ia, ib]
    ia, ib = ia[keep], ib[keep]
    return pd.DataFrame({
        "from_track": A.track_id.to_numpy()[ia], "to_track": B.track_id.to_numpy()[ib],
        "frame_end": A["last"].to_numpy()[ia], "frame_start": B["first"].to_numpy()[ib],
        "gap_s": gap[ia, ib].round(3), "dx_px": dx[ia, ib].round(1), "tol_x_px": tol_x[ia, ib].round(1),
        "dy_px": dy[ia, ib].round(1), "tol_y_px": tol_y[ia, ib].round(1),
        "width_ratio": ratio[ia, ib].round(3), "cost": cost[ia, ib].round(3),
        "x": A.x_last.to_numpy()[ia].round(0), "y": A.y_last.to_numpy()[ia].round(0),
    }, columns=LINK_COLUMNS)


def chain_ids(links: pd.DataFrame) -> dict[int, int]:
    """raw track ID -> vehicle ID (the ID of the first piece of its chain)."""
    nxt = dict(zip(links.from_track.astype(int), links.to_track.astype(int)))
    targets = set(nxt.values())
    vehicle = {}
    for root in (t for t in nxt if t not in targets):
        cur = root
        vehicle[cur] = root
        while cur in nxt:
            cur = nxt[cur]
            vehicle[cur] = root
    return vehicle


def stitch(points: pd.DataFrame, fps: float, frame_range: tuple[int, int], width: int,
           sp: StitchParams = StitchParams()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (points with stitched track_id and raw_track_id kept, chosen links)."""
    out = points.copy()
    if "raw_track_id" not in out:
        out["raw_track_id"] = out.track_id
    tr = endpoint_table(out, fps, frame_range, width, sp)
    if tr.empty:
        return out, pd.DataFrame(columns=LINK_COLUMNS)
    links = choose_links(tr, fps, sp)
    if not links.empty:
        vehicle = chain_ids(links)
        out["track_id"] = out.track_id.map(lambda t: vehicle.get(int(t), int(t))).astype(int)
    return out.sort_values(["frame_idx", "track_id"]).reset_index(drop=True), links
