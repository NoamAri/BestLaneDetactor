"""From boxes to clean vehicle positions on the road (notes/lane_speed_plan.md §2).

ground_point   the image point that stands for the vehicle's position: the box centre, moved into the
               stabilised reference frame. (Amendment A2: the pre-registered "bottom + half width"
               shift misplaced vehicles in the ground-truth check; see notes/lane_speed.md.) The shift
               is kept as an option for the ablation table.
build_ground   all analysis windows of a camera in one table (unique track ids).
road_points    road coordinates s (m along the road) and d (m across) per box, the occlusion filter
               (box height < 70% of the track's median: the lower part is hidden) and the plausibility
               cuts (an identity jump: the box landed on another vehicle). A cut track continues as a
               new segment (seg_id); nothing is thrown away except the step across the cut.

Plausibility, at every step between consecutive boxes of a track:
  * step speed |s(i+1) - s(i)| / dt > max_speed;
  * acceleration: speeds fitted (least squares) over accel_window_s before and after the step;
    |v_after - v_before| / (time between the two window centres) > max_accel. Fitted over 0.5 s
    because single-frame speeds at 30 fps are mostly position noise;
  * lateral jump: mean d over lateral_jump_window_s after minus before > max_lateral_jump_lane_frac
    of a lane width.
A limit counts as exceeded only beyond 3x the step's own measurement noise (amendment A4); for the
acceleration that noise includes the track's slow wobble of fitted speeds (amendment A5).
A jump trips several neighbouring steps; one cut per run, placed by the most precise evidence
(single step, then lateral, then acceleration).
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


# --------------------------------------------------------------------------- road coordinates and cleaning
def _window_fits(t: np.ndarray, v: np.ndarray, win: float):
    """For each step i -> i+1: least-squares fit of v over [t_i - win, t_i] and [t_{i+1}, t_{i+1} + win].

    Returns, per side, (slope, mean, mean time, n, Sxx) with n = 0 where the window spans less than
    80% of win (track ends): no estimate there. Sxx = sum (t - mean t)^2 gives the slope's noise.
    """
    n = len(t)
    c = lambda a: np.concatenate([[0.0], np.cumsum(a)])  # noqa: E731
    St, Sv, Stt, Stv, S1 = c(t), c(v), c(t * t), c(t * v), np.arange(n + 1, dtype=float)
    i = np.arange(n - 1)
    lo_l = np.searchsorted(t, t[i] - win - 1e-9, "left")
    hi_l = i + 1
    lo_r = i + 1
    hi_r = np.searchsorted(t, t[i + 1] + win + 1e-9, "right")

    def fit(lo, hi):
        m = S1[hi] - S1[lo]
        st, sv, stt, stv = St[hi] - St[lo], Sv[hi] - Sv[lo], Stt[hi] - Stt[lo], Stv[hi] - Stv[lo]
        den = m * stt - st * st
        good = den > 1e-12
        slope = np.where(good, (m * stv - st * sv) / np.where(good, den, 1), np.nan)
        sxx = np.where(good, den / np.maximum(m, 1), np.nan)
        span = t[np.maximum(hi - 1, lo)] - t[lo]
        return slope, sv / m, st / m, np.where(span >= 0.8 * win, m, 0), sxx
    return fit(lo_l, hi_l), fit(lo_r, hi_r)


def noise_sd(v: np.ndarray) -> float:
    """Robust position noise of a track: MAD of second differences (true motion is negligible at video rates)."""
    if len(v) < 5:
        return 0.0
    d2 = np.diff(v, 2)
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6))


def plausibility_cuts(t: np.ndarray, s: np.ndarray, d: np.ndarray, p: dict, lane_width_m: float,
                      z: float = 3.0) -> np.ndarray:
    """Indices i where the track is cut between box i and box i+1.

    A step is implausible when it exceeds a physical limit by more than z times its own measurement
    noise (amendment A4): the limits are those of configs/lane_speed.json, the noise is estimated per
    track from its positions, so honest driving seen through noisy boxes is not cut.
    """
    if len(t) < 2:
        return np.empty(0, int)
    dt = np.maximum(np.diff(t), 1e-6)
    sig_s, sig_d = noise_sd(s), noise_sd(d)
    vmax = p["max_speed_kmh"] / 3.6
    step = (np.abs(np.diff(s)) - z * sig_s * np.sqrt(2)) / dt / vmax
    (vl, _, tl, nl, xl), (vr, _, tr, nr, xr) = _window_fits(t, s, p["accel_window_s"])
    ok = (nl >= 3) & (nr >= 3) & np.isfinite(vl) & np.isfinite(vr)
    dv = np.where(ok, vr - vl, np.nan)
    # Speed noise from frame-to-frame jitter, and the track's own slow wobble of fitted speeds (box
    # shape changes as a vehicle passes shadows and neighbours: correlated over many frames, so it is
    # invisible to the jitter estimate). Robust spread of dv over the whole track (amendment A5).
    sig_white = sig_s * np.sqrt(1 / np.where(ok, xl, 1) + 1 / np.where(ok, xr, 1))
    dvv = dv[np.isfinite(dv)]
    sig_wobble = 1.4826 * float(np.median(np.abs(dvv - np.median(dvv)))) if len(dvv) >= 10 else 0.0
    sig_dv = np.maximum(sig_white, sig_wobble)
    sep = np.maximum(tr - tl, p["accel_window_s"] / 2)
    acc = np.where(ok, (np.abs(dv) - z * sig_dv) / sep, 0.0) / p["max_accel_ms2"]
    (_, dl, _, ml, _), (_, dr, _, mr, _) = _window_fits(t, d, p["lateral_jump_window_s"])
    okd = (ml > 0) & (mr > 0)
    sig_dd = sig_d * np.sqrt(1 / np.maximum(ml, 1) + 1 / np.maximum(mr, 1))
    lat = np.where(okd, np.abs(dr - dl) - z * sig_dd, 0.0) / (p["max_lateral_jump_lane_frac"] * lane_width_m)
    score = np.nanmax(np.vstack([step, acc, lat]), axis=0)
    bad = np.flatnonzero(score > 1)
    if len(bad) == 0:
        return bad
    def where(run):
        # most precise evidence first: a single too-fast step pins the jump exactly; windowed
        # fits straddling a jump smear it over the neighbouring steps
        for k in (step, lat, acc):
            if np.nanmax(k[run]) > 1:
                return run[int(np.nanargmax(k[run]))]
        return run[0]
    cuts, run = [], [bad[0]]
    win = max(p["accel_window_s"], p["lateral_jump_window_s"])
    for i in bad[1:]:
        if t[i] - t[run[-1]] <= win:
            run.append(i)
        else:
            cuts.append(where(np.array(run)))
            run = [i]
    cuts.append(where(np.array(run)))
    return np.array(cuts, int)


def road_points(g: pd.DataFrame, rf, lanes: list[dict], lane_d: dict[str, float], ls: dict) -> pd.DataFrame:
    """Per box: s, d (metres), lane (id or ""), travel sign, occluded flag, seg_id after plausibility cuts.

    g: the ground table with lane_frame filled (lanes.py --write-points). Occluded boxes are dropped.
    travel sign: +1 when the lane's traffic moves toward +s, -1 otherwise (from the image direction).
    """
    g = g.sort_values(["track_id", "t"]).reset_index(drop=True)
    s, d = rf.sd(g.x_px.to_numpy(), g.y_px.to_numpy())
    ids = np.array([l["id"] for l in lanes] + [""])
    out = g.assign(s=s, d=d, lane=ids[g.lane_frame.to_numpy()])
    h = out.y2 - out.y1
    out["occluded"] = h < ls["occlusion_height_frac"] * h.groupby(out.track_id).transform("median")
    out = out[~out.occluded].reset_index(drop=True)
    # direction of s for each image direction: s grows with image +x (both calibration kinds)
    lane_sign = {l["id"]: (1 if l["direction"] == "right" else -1) for l in lanes}
    out["lane_sign"] = out.lane.map(lane_sign).fillna(0).astype(int)
    spacing = np.diff(sorted(lane_d.values()))
    lane_w = float(np.median(spacing)) if len(spacing) else 3.6
    seg = np.empty(len(out), np.int64)
    pos = 0
    for tid, grp in out.groupby("track_id", sort=False):
        n = len(grp)
        cuts = (plausibility_cuts(grp.t.to_numpy(), grp.s.to_numpy(), grp.d.to_numpy(), ls["plausibility"], lane_w)
                if ls["plausibility"].get("enabled", True) else ())
        part = np.zeros(n, np.int64)
        for c in cuts:
            part[c + 1:] += 1
        seg[pos:pos + n] = tid * 100 + part
        pos += n
    out["seg_id"] = seg
    out.attrs["lane_width_m"] = lane_w
    return out
