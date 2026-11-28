"""Lane membership over time: lane pieces and lane changes (notes/lane_speed_plan.md §4).

A vehicle's lane can change during its track, and its box can flicker across a lane line. So lanes
are taken per box, never one lane per track:

1. Hysteresis: within a segment (a track after plausibility cuts) a run of boxes in a new lane counts
   only if the vehicle stays there for at least ``min_dwell_s`` (time from the run's first box to the
   next run's first box). Shorter runs take the previous confirmed lane (the next one at the start).
   Boxes outside every lane ("") follow the same rule: a brief excursion is kept in the lane.
2. Lane pieces: consecutive boxes with the same smoothed lane.
3. The change is moved to where the vehicle crosses the line between the two lanes (the middle
   crossing of the midline within min_dwell_s, since hysteresis confirms it only after any flicker),
   and the step across it is split at the interpolated crossing. Each part goes to its own lane.
4. Lane-change events (from -> to, same direction, both real lanes) are recorded with their time.

``segments`` returns the trajectory as straight pieces between consecutive boxes, each in one lane:
the input of the Edie sums (lane_speed.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_lanes(t: np.ndarray, lane: np.ndarray, min_dwell_s: float) -> np.ndarray:
    """Lane per box after hysteresis (see module doc)."""
    n = len(lane)
    if n == 0:
        return lane.copy()
    starts = np.flatnonzero(np.r_[True, lane[1:] != lane[:-1]])
    ends = np.r_[starts[1:], n]                                     # exclusive
    tail = float(np.median(np.diff(t))) if n > 1 else 0.0
    dur = np.r_[t[starts[1:]] - t[starts[:-1]], t[-1] - t[starts[-1]] + tail]
    run_lane = lane[starts]
    ok = dur >= min_dwell_s
    if not ok.any():
        # nothing lasts long enough: the lane where the vehicle spent the most time
        tot = pd.Series(dur).groupby(run_lane).sum()
        return np.full(n, tot.idxmax(), dtype=lane.dtype)
    conf = run_lane.copy()
    last = None
    for k in range(len(starts)):
        if ok[k]:
            last = run_lane[k]
        elif last is not None:
            conf[k] = last
    first_ok = int(np.flatnonzero(ok)[0])
    conf[:first_ok] = run_lane[first_ok]
    return np.repeat(conf, ends - starts)


def move_to_crossings(t: np.ndarray, d: np.ndarray, lane: np.ndarray, lane_d: dict[str, float],
                      window_s: float) -> np.ndarray:
    """Move each confirmed lane change to where the vehicle crosses the line between the two lanes.

    Hysteresis confirms a change at the end of any flicker; the crossing itself is the middle one of
    the crossings of the midline (d at the midpoint of the lane centres) within window_s either side.
    """
    lane = lane.copy()
    for j in np.flatnonzero(lane[1:] != lane[:-1]) + 1:
        a, b = lane[j - 1], lane[j]
        if a == "" or b == "" or a not in lane_d or b not in lane_d:
            continue
        mid = (lane_d[a] + lane_d[b]) / 2
        lo = np.searchsorted(t, t[j] - window_s)
        hi = min(len(t), np.searchsorted(t, t[j] + window_s, "right"))
        side = np.sign(d[lo:hi] - mid) * np.sign(lane_d[b] - lane_d[a])     # +1 on the new lane's side
        xs = np.flatnonzero((side[1:] > 0) & (side[:-1] <= 0)) + lo + 1      # first box on the new side
        if len(xs) == 0:
            continue
        k = int(xs[len(xs) // 2])
        if k < j and np.all(lane[k:j] == a):
            lane[k:j] = b
        elif k > j and np.all(lane[j:k] == b):
            lane[j:k] = a
    return lane


def require_lateral(t: np.ndarray, d: np.ndarray, lane: np.ndarray, lane_d: dict[str, float],
                    min_frac: float, window_s: float, lane_change_gap_s: float = 0.0) -> np.ndarray:
    """Undo lane changes without real sideways movement (amendment A7).

    For each change a -> b: the median lateral position over window_s ending gap_s before the change and over
    window_s starting gap_s after it (so the manoeuvre itself is skipped) must differ by at least min_frac of
    |d(b) - d(a)| (the spacing of the two lane centres).
    Otherwise the run in lane b is returned to lane a. With too few boxes on either side the change stays.
    """
    lane = lane.copy()
    j = 1
    while j < len(lane):
        if lane[j] == lane[j - 1]:
            j += 1
            continue
        a, b = lane[j - 1], lane[j]
        if a == "" or b == "" or a not in lane_d or b not in lane_d:
            j += 1
            continue
        gap = lane_change_gap_s                       # skip the manoeuvre itself: compare well before with well after
        before = d[(t >= t[j] - gap - window_s) & (t < t[j] - gap)]
        after = d[(t > t[j] + gap) & (t <= t[j] + gap + window_s)]
        need = min_frac * abs(lane_d[b] - lane_d[a])
        if len(before) >= 2 and len(after) >= 2 and abs(np.median(after) - np.median(before)) < need:
            k = j
            while k < len(lane) and lane[k] == b:
                k += 1
            lane[j:k] = a                                  # no sideways movement: it never left lane a
        j += 1
    return lane


def build(road: pd.DataFrame, lane_d: dict[str, float], lane_dir: dict[str, str],
          min_dwell_s: float, lane_change: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smooth lanes per segment; return (segments, lane_changes).

    road: boxes with seg_id, t, s, d, lane ("" outside lanes), cls; sorted by seg_id, t.
    segments: one row per straight piece between consecutive boxes (or a part of one, at a lane
    change): seg_id, cls, lane, t_a, t_b, s_a, s_b.
    lane_changes: seg_id, t, s (road position of the crossing), from_lane, to_lane.
    lane_change: {"min_lateral_frac", "window_s"} applies require_lateral (None: label changes only).
    """
    road = road.sort_values(["seg_id", "t"]).reset_index(drop=True)
    sm = np.empty(len(road), dtype=object)
    t_all, l_all = road.t.to_numpy(), road.lane.to_numpy().astype(object)
    bounds = np.flatnonzero(np.r_[True, road.seg_id.to_numpy()[1:] != road.seg_id.to_numpy()[:-1], True])
    d_all = road.d.to_numpy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        ln = smooth_lanes(t_all[a:b], l_all[a:b], min_dwell_s)
        if lane_change:
            ln = require_lateral(t_all[a:b], d_all[a:b], ln, lane_d, lane_change["min_lateral_frac"], lane_change["window_s"],
                                 lane_change.get("gap_s", 0.0))
        sm[a:b] = move_to_crossings(t_all[a:b], d_all[a:b], ln, lane_d, min_dwell_s)
    road = road.assign(lane_s=sm)

    same_seg = road.seg_id.to_numpy()[1:] == road.seg_id.to_numpy()[:-1]
    ia = np.flatnonzero(same_seg)
    ib = ia + 1
    la, lb = road.lane_s.to_numpy()[ia], road.lane_s.to_numpy()[ib]
    t, s, d = road.t.to_numpy(), road.s.to_numpy(), road.d.to_numpy()
    seg, cls = road.seg_id.to_numpy(), road.cls.to_numpy()
    keep = la == lb
    segs = [pd.DataFrame({"seg_id": seg[ia[keep]], "cls": cls[ia[keep]], "lane": la[keep],
                          "t_a": t[ia[keep]], "t_b": t[ib[keep]], "s_a": s[ia[keep]], "s_b": s[ib[keep]]})]
    ch = np.flatnonzero(~keep)
    i, j = ia[ch], ib[ch]
    d_mid = np.array([(lane_d.get(x, np.nan) + lane_d.get(y, np.nan)) / 2 for x, y in zip(la[ch], lb[ch])], float)
    dd = d[j] - d[i]
    u = np.where(np.isfinite(d_mid) & (np.abs(dd) > 1e-9), (d_mid - d[i]) / np.where(np.abs(dd) > 1e-9, dd, 1), 0.5)
    u = np.clip(u, 0.0, 1.0)
    tc, sc = t[i] + u * (t[j] - t[i]), s[i] + u * (s[j] - s[i])
    segs.append(pd.DataFrame({"seg_id": seg[i], "cls": cls[i], "lane": la[ch], "t_a": t[i], "t_b": tc, "s_a": s[i], "s_b": sc}))
    segs.append(pd.DataFrame({"seg_id": seg[j], "cls": cls[j], "lane": lb[ch], "t_a": tc, "t_b": t[j], "s_a": sc, "s_b": s[j]}))
    segments = pd.concat(segs, ignore_index=True)
    segments = segments[segments.lane != ""].reset_index(drop=True)
    real = np.array([x != "" and y != "" and lane_dir.get(x) == lane_dir.get(y) for x, y in zip(la[ch], lb[ch])], bool)
    changes = pd.DataFrame({"seg_id": seg[i][real], "t": tc[real], "s": sc[real], "from_lane": la[ch][real], "to_lane": lb[ch][real]})
    return segments, changes
