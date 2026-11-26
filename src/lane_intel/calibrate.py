"""Calibration: from image pixels to metres along and across the road (notes/lane_speed_plan.md §1).

Speeds in pixels are biased by perspective (far lanes look slower), so lanes are compared in metres.
Two methods, chosen per camera in its config (``calibration.method``):

gt_homography   For cameras with ground truth (MiTra). Our tracked vehicles are matched, frame by
                frame, to the ground-truth vehicle positions (UTM metres, same clock), and a
                homography pixels -> metres is fitted with RANSAC. It starts from a rough similarity
                transform (road axis, direction of travel, carriageway order) and is refined by
                iterative matching with a shrinking gate. The video/ground-truth time offset is
                checked by refitting at nearby offsets.

vehicle_size    For cameras without ground truth. Lateral scale from the lane spacing (lane width in
                metres given in the config), scale along the road from the median length of cars
                (given in the config), both as smooth functions of position, so perspective is
                followed. Distances along a lane are integrated along the lane line.

Both produce a RoadFrame that gives every image point a road position: s (metres along the road,
increasing in the direction of travel of right-moving traffic) and d (metres across it), plus the
lane centres in d.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linear_sum_assignment

from lane_intel.lanes import Geometry, lane_edges, track_table, LaneParams


# --------------------------------------------------------------------------- road frame
@dataclass
class RoadFrame:
    """Maps image points to road coordinates (s along, d across, metres)."""
    kind: str
    # gt_homography
    H: np.ndarray | None = None              # pixels -> local metres (world minus origin)
    origin: np.ndarray | None = None          # world origin (UTM) of the local metres
    u_along: np.ndarray | None = None         # unit vector along the road (direction of right-moving traffic)
    u_across: np.ndarray | None = None        # unit vector across the road (image down = +d)
    # vehicle_size
    s_grid: RegularGridInterpolator | None = None
    d_of_c: tuple[np.ndarray, np.ndarray] | None = None
    geo: Geometry | None = None
    info: dict | None = None

    def world(self, x, y) -> np.ndarray:
        pts = np.stack([np.asarray(x, float), np.asarray(y, float)], -1).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def sd(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.kind == "gt_homography":
            w = self.world(x, y)
            return w @ self.u_along, w @ self.u_across
        c = self.geo.c_of(x, y)
        s = self.s_grid(np.stack([np.clip(c, *self.s_grid.grid[0][[0, -1]]),
                                  np.clip(x, *self.s_grid.grid[1][[0, -1]])], -1))
        return s, np.interp(c, *self.d_of_c)

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "info": self.info}
        if self.kind == "gt_homography":
            d.update(H=self.H.tolist(), origin=self.origin.tolist(), u_along=self.u_along.tolist(),
                     u_across=self.u_across.tolist())
        return d


# --------------------------------------------------------------------------- ground truth homography
def load_gt(csv: str | Path, t0: float, t1: float) -> pd.DataFrame:
    cols = ["Vehicle_ID", "Vehicle_type", "Time [s]", "x [m]", "y [m]", "Speed [km/h]", "Lane",
            "Angle [rad]", "Vehicle_width [m]"]
    out = []
    for chunk in pd.read_csv(csv, usecols=cols, chunksize=500_000):
        out.append(chunk[(chunk["Time [s]"] >= t0) & (chunk["Time [s]"] < t1)])
    gt = pd.concat(out).rename(columns={"Vehicle_ID": "vid", "Vehicle_type": "type", "Time [s]": "t", "x [m]": "x",
                                        "y [m]": "y", "Speed [km/h]": "v_kmh", "Lane": "lane", "Angle [rad]": "angle",
                                        "Vehicle_width [m]": "width"})
    return gt.reset_index(drop=True)


def _similarity_init(pts: pd.DataFrame, gt: pd.DataFrame, width: int) -> tuple[np.ndarray, dict]:
    """Rough pixels -> local metres map from road geometry alone (no correspondences)."""
    tr = track_table(pts, width, LaneParams())
    pts = pts.assign(dir=pts.track_id.map(tr.dir))
    # ground truth: two headings; the faster group is the free-moving carriageway
    head = np.where(np.cos(gt.angle - np.median(gt.angle)) > 0, 0, 1)
    g0, g1 = gt[head == 0], gt[head == 1]
    fast_w, slow_w = (g0, g1) if g0.v_kmh.median() > g1.v_kmh.median() else (g1, g0)
    # image: right-moving (+1) vs left-moving (-1); faster image direction = larger |px/s|
    r, l = tr[tr.dir == 1], tr[tr.dir == -1]
    fast_i_dir = 1 if r.v.abs().median() > l.v.abs().median() else -1
    fast_i = pts[pts.dir == fast_i_dir]
    slow_i = pts[pts.dir == -fast_i_dir]
    a = np.median(fast_w.angle)
    h = np.array([np.cos(a), np.sin(a)])                     # world heading of the fast carriageway
    u_x = h * fast_i_dir                                      # world direction of image +x
    sep_w = np.array([fast_w.x.median() - slow_w.x.median(), fast_w.y.median() - slow_w.y.median()])
    sep_i = fast_i.y_px.median() - slow_i.y_px.median()      # image +y separation (px)
    n = np.array([-u_x[1], u_x[0]])
    u_y = n if np.sign(sep_w @ n) == np.sign(sep_i) else -n   # world direction of image +y
    # scales: along the road from extents, across from the carriageway separation
    ax_w = np.concatenate([fast_w.x, slow_w.x]) * u_x[0] + np.concatenate([fast_w.y, slow_w.y]) * u_x[1]
    k_x = np.subtract(*np.percentile(ax_w, [95, 5])) / max(1.0, np.subtract(*np.percentile(pts.x_px, [95, 5])))
    k_y = abs(sep_w @ u_y) / max(1.0, abs(sep_i))
    cw = np.array([(fast_w.x.median() + slow_w.x.median()) / 2, (fast_w.y.median() + slow_w.y.median()) / 2])
    ci = np.array([pts.x_px.median(), (fast_i.y_px.median() + slow_i.y_px.median()) / 2])
    A = np.stack([u_x * k_x, u_y * k_y], 1)                   # 2x2: pixel offsets -> world offsets
    t = cw - A @ ci
    H = np.eye(3)
    H[:2, :2], H[:2, 2] = A, t
    return H, {"fast_image_direction": int(fast_i_dir)}


def _match(pf: pd.DataFrame, gf: pd.DataFrame, H: np.ndarray, gate: float) -> pd.DataFrame | None:
    """One-to-one matching of our boxes to ground-truth vehicles in one frame (Hungarian, metres)."""
    if pf.empty or gf.empty:
        return None
    w = cv2.perspectiveTransform(pf[["x_px", "y_px"]].to_numpy(float).reshape(-1, 1, 2), H).reshape(-1, 2)
    g = gf[["xl", "yl"]].to_numpy(float)
    dist = np.linalg.norm(w[:, None, :] - g[None, :, :], axis=2)
    ia, ig = linear_sum_assignment(np.where(dist <= gate, dist, 1e6))
    ok = dist[ia, ig] <= gate
    return pd.DataFrame({"x_px": pf.x_px.to_numpy()[ia[ok]], "y_px": pf.y_px.to_numpy()[ia[ok]],
                         "xl": g[ig[ok], 0], "yl": g[ig[ok], 1],
                         "track_id": pf.track_id.to_numpy()[ia[ok]], "vid": gf.vid.to_numpy()[ig[ok]]})


def _correspondences(pts: pd.DataFrame, gt_by: dict, pairs: pd.DataFrame, t_gt0: float, dt: float) -> pd.DataFrame:
    """Every box of each paired track against the ground truth interpolated to the same instant."""
    out = []
    by_t = dict(tuple(pts.groupby("track_id")))
    for tid, vid in pairs[["track_id", "vid"]].itertuples(index=False):
        p, g = by_t[tid], gt_by[vid]
        tt = t_gt0 + p.t.to_numpy() + dt
        m = (tt >= g.t.iat[0]) & (tt <= g.t.iat[-1])
        if not m.any():
            continue
        k = np.clip(np.searchsorted(g.t.to_numpy(), tt[m]), 0, len(g) - 1)
        out.append(pd.DataFrame({"track_id": tid, "vid": vid, "t": p.t.to_numpy()[m], "cls": p.cls.to_numpy()[m],
                                 "gt_lane": g.lane.to_numpy()[k],
                                 "x_px": p.x_px.to_numpy()[m], "y_px": p.y_px.to_numpy()[m],
                                 "X": np.interp(tt[m], g.t, g.xl), "Y": np.interp(tt[m], g.t, g.yl),
                                 "v_kmh": np.interp(tt[m], g.t, g.v_kmh)}))
    return pd.concat(out, ignore_index=True)


def _err(H: np.ndarray, c: pd.DataFrame) -> np.ndarray:
    w = cv2.perspectiveTransform(c[["x_px", "y_px"]].to_numpy(float).reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(w - c[["X", "Y"]].to_numpy(), axis=1)


def fit_gt_homography(pts: pd.DataFrame, gt: pd.DataFrame, fps: float, clip_offset_s: float, width: int,
                      frame_step: int = 5, moving_kmh: float = 20.0,
                      dt_grid=np.round(np.arange(-0.20, 0.2001, 0.01), 3)) -> tuple[RoadFrame, dict, pd.DataFrame]:
    """Homography from (stabilised) image points to ground-truth metres.

    pts: one row per box with t (clip seconds), track_id, cls, x_px, y_px (the ground point, already
    moved into the reference frame). gt: load_gt() rows covering the same time span.

    1. Rough start from road geometry (_similarity_init), refined by per-frame one-to-one matching
       with a shrinking gate (12 m -> 1.5 m) and RANSAC.
    2. Identity pairs: a track is paired with the ground-truth vehicle it matches in > 80% of its
       matched frames (at least 5). Wrong matches in dense traffic are dropped by this.
    3. Time offset: for every candidate offset (10 ms steps) every box of every pair is compared with
       the ground truth interpolated to that instant; the offset with the smallest median error on
       moving vehicles wins (standing vehicles do not constrain time).
    4. Final homography by RANSAC (1.5 m) on all those correspondences.
    """
    origin = np.array([gt.x.median(), gt.y.median()])
    gt = gt.assign(xl=gt.x - origin[0], yl=gt.y - origin[1], gframe=np.round(gt.t * fps).astype(int)).sort_values("t")
    pts = pts.assign(gframe=np.round((clip_offset_s + pts.t) * fps).astype(int))
    frames = np.sort(pts.gframe.unique())[::frame_step]
    by_p = dict(tuple(pts[pts.gframe.isin(frames)].groupby("gframe")))
    by_g = dict(tuple(gt.groupby("gframe")))

    def matches(H, gate):
        m = [_match(by_p[f], by_g.get(f, pd.DataFrame()), H, gate) for f in frames if f in by_p]
        return pd.concat([x for x in m if x is not None], ignore_index=True)

    H, init_info = _similarity_init(pts, gt.assign(x=gt.xl, y=gt.yl), width)
    for gate in (12, 8, 5, 3, 2, 1.5):
        M = matches(H, gate)
        H_new, _ = cv2.findHomography(M[["x_px", "y_px"]].to_numpy(float), M[["xl", "yl"]].to_numpy(float),
                                      cv2.RANSAC, 1.0)
        if H_new is None:
            raise RuntimeError("calibration: matching to ground truth failed")
        H = H_new
    M = matches(H, 2.0)
    cnt = M.groupby(["track_id", "vid"]).size().rename("n").reset_index()
    cnt["share"] = cnt.n / cnt.track_id.map(M.groupby("track_id").size())
    pairs = cnt[(cnt.share > 0.8) & (cnt.n >= 5)].sort_values("n").drop_duplicates("vid", keep="last")
    gt_by = dict(tuple(gt[gt.vid.isin(pairs.vid)].groupby("vid")))

    scan = {}
    for dt in dt_grid:
        c = _correspondences(pts, gt_by, pairs, clip_offset_s, dt)
        Hd, _ = cv2.findHomography(c[["x_px", "y_px"]].to_numpy(float), c[["X", "Y"]].to_numpy(float), 0)
        e = _err(Hd, c)
        mv = c.v_kmh.to_numpy() > moving_kmh
        scan[float(dt)] = float(np.median(e[mv] if mv.sum() > 100 else e))
    dt = min(scan, key=scan.get)
    c = _correspondences(pts, gt_by, pairs, clip_offset_s, dt)
    H, inl = cv2.findHomography(c[["x_px", "y_px"]].to_numpy(float), c[["X", "Y"]].to_numpy(float), cv2.RANSAC, 1.5)
    c["err_m"] = _err(H, c)

    ref_y = float(pts.y_px.median())
    q = np.array([[width / 2, ref_y], [width / 2 + 100, ref_y], [width / 2, ref_y + 100]], float)
    w = cv2.perspectiveTransform(q.reshape(-1, 1, 2), H).reshape(-1, 2)
    u_along = (w[1] - w[0]) / np.linalg.norm(w[1] - w[0])
    nrm = np.array([-u_along[1], u_along[0]])
    u_across = nrm if (w[2] - w[0]) @ nrm > 0 else -nrm
    jac = np.array([np.linalg.norm(w[1] - w[0]), np.linalg.norm(w[2] - w[0])]) / 100
    info = {"method": "gt_homography", "time_offset_s": dt,
            "time_scan_moving_median_err_m": {f"{k:+.2f}": round(v, 3) for k, v in scan.items()},
            "tracks": int(pts.track_id.nunique()), "paired_tracks": int(len(pairs)),
            "correspondences": int(len(c)), "ransac_inliers": int(inl.sum()),
            "reprojection_m": {"median": round(float(c.err_m.median()), 3),
                               "p90": round(float(c.err_m.quantile(0.9)), 3),
                               "by_class_median": c.groupby("cls").err_m.median().round(3).to_dict(),
                               "moving_median": round(float(c.err_m[c.v_kmh > moving_kmh].median()), 3)},
            "scale_m_per_px_at_centre": {"along_x": round(float(jac[0]), 4), "down_y": round(float(jac[1]), 4)},
            **init_info}
    rf = RoadFrame("gt_homography", H=H, origin=origin, u_along=u_along, u_across=u_across, info=info)
    return rf, info, c


def speed_check(rf: RoadFrame, c: pd.DataFrame, u_along: np.ndarray | None = None,
                min_span_s: float = 3.0, min_dist_m: float = 10.0) -> dict:
    """Per paired track: distance along the road over its matched span, ours vs ground truth.

    This is what lane speeds depend on: a constant position offset cancels, a wrong scale does not.
    """
    rows = []
    for (tid, vid), g in c.groupby(["track_id", "vid"]):
        g = g.sort_values("t")
        if g.t.iat[-1] - g.t.iat[0] < min_span_s:
            continue
        s, _ = rf.sd(g.x_px.to_numpy(), g.y_px.to_numpy())
        sg = g[["X", "Y"]].to_numpy() @ (rf.u_along if u_along is None else u_along)
        d_gt = sg[-1] - sg[0]
        if abs(d_gt) < min_dist_m:
            continue
        rows.append((tid, g.cls.iat[0], abs(s[-1] - s[0]) / abs(d_gt), abs(d_gt) / (g.t.iat[-1] - g.t.iat[0]) * 3.6))
    r = pd.DataFrame(rows, columns=["track_id", "cls", "ratio", "v_kmh"])
    return {"tracks": int(len(r)), "ratio_median": round(float(r.ratio.median()), 4),
            "ratio_p10_p90": [round(float(r.ratio.quantile(0.1)), 4), round(float(r.ratio.quantile(0.9)), 4)],
            "abs_rel_error_median": round(float((r.ratio - 1).abs().median()), 4),
            "ratio_by_class_median": r.groupby("cls").ratio.median().round(4).to_dict()}


# --------------------------------------------------------------------------- vehicle-size scale field
def lateral_nodes(lanes: list[dict], lane_width_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear map c (px) -> d (metres) through the lane centres.

    Adjacent lanes of one carriageway are one lane width apart by assumption. A gap between
    carriageways is converted with the mean spacing of the lanes next to it, so it keeps its
    proportion. Beyond the outer lanes the outer spacing is extended.
    """
    ls = sorted(lanes, key=lambda l: l["c"])
    cs = np.array([l["c"] for l in ls], float)
    dirs = [l["direction"] for l in ls]
    gaps = np.diff(cs)
    same = np.array([dirs[i] == dirs[i + 1] for i in range(len(ls) - 1)])
    d = [0.0]
    for i, g in enumerate(gaps):
        if same[i]:
            d.append(d[-1] + lane_width_m)
        else:
            near = [gaps[j] for j in (i - 1, i + 1) if 0 <= j < len(gaps) and same[j]]
            d.append(d[-1] + lane_width_m * g / (np.mean(near) if near else np.median(gaps)))
    first = gaps[0] if len(gaps) else 40.0
    last = gaps[-1] if len(gaps) else 40.0
    c_nodes = np.concatenate([[cs[0] - 3 * first], cs, [cs[-1] + 3 * last]])
    d_nodes = np.concatenate([[d[0] - 3 * lane_width_m], d, [d[-1] + 3 * lane_width_m]])
    return c_nodes, d_nodes


def fit_vehicle_size(pts: pd.DataFrame, geo: Geometry, lanes: list[dict], width: int, lane_width_m: float,
                     car_length_m: float, occlusion_frac: float = 0.7,
                     n_c: int = 80, n_x: int = 241) -> tuple[RoadFrame, dict]:
    """Scale field for a camera without ground truth.

    Across the road: lane spacing = lane_width_m (lateral_nodes).
    Along the road: a car's box length, measured along its lane line, is car_length_m. Metres per pixel
    k(x, c) is modelled as exp(a + b*x' + e*c') (smooth perspective change across the image), fitted
    to the median of log k in position bins (weights sqrt(n)). Only unoccluded car boxes fully inside
    the frame are used. s(c, x) = integral of k * |d(lane line)/dx| from x_ref to x along the lane line.
    """
    pts = pts.copy()
    pts["c"] = geo.c_of(pts.x_px.to_numpy(), pts.y_px.to_numpy())
    cars = pts[pts.cls == "car"]
    med_h = cars.groupby("track_id").apply(lambda g: (g.y2 - g.y1).median(), include_groups=False)
    cars = cars[(cars.y2 - cars.y1) >= occlusion_frac * cars.track_id.map(med_h)]
    cars = cars[(cars.x1 > 2) & (cars.x2 < width - 2)]
    eps = 2.0
    slope = (geo.y_of(cars.c.to_numpy(), cars.x_px.to_numpy() + eps) -
             geo.y_of(cars.c.to_numpy(), cars.x_px.to_numpy() - eps)) / (2 * eps)
    along_px = (cars.x2 - cars.x1).to_numpy() * np.sqrt(1 + slope ** 2)
    lk = np.log(car_length_m / along_px)
    c_mid = float(np.median([l["c"] for l in lanes]))
    xs, cs_ = width, max(1.0, float(np.ptp([l["c"] for l in lanes])))
    b = pd.DataFrame({"x": cars.x_px.to_numpy(), "c": cars.c.to_numpy(), "lk": lk,
                      "xb": pd.cut(cars.x_px, 8, labels=False), "cb": pd.cut(cars.c, 6, labels=False)})
    b = b.groupby(["xb", "cb"]).agg(x=("x", "median"), c=("c", "median"), lk=("lk", "median"),
                                    n=("lk", "size")).reset_index()
    b = b[b.n >= 10]
    X = np.c_[np.ones(len(b)), (b.x - width / 2) / xs, (b.c - c_mid) / cs_]
    wts = np.sqrt(b.n.to_numpy())
    coef, *_ = np.linalg.lstsq(X * wts[:, None], b.lk.to_numpy() * wts, rcond=None)
    model = {"coef": [float(v) for v in coef], "x_mid": width / 2, "x_scale": xs, "c_mid": c_mid, "c_scale": cs_}
    rf = vehicle_size_frame(model, geo, lanes, width, lane_width_m, n_c, n_x)
    rf.info = {"method": "vehicle_size", "lane_width_m": lane_width_m, "car_length_m": car_length_m,
               "car_boxes": int(len(cars)), "position_bins": int(len(b)), "model": model,
               "scale_m_per_px_at_centre": round(float(np.exp(coef[0])), 4),
               "scale_ratio_right_vs_left_edge": round(float(np.exp(coef[1])), 3),
               "scale_ratio_lowest_vs_highest_lane": round(float(np.exp(coef[2])), 3)}
    return rf, rf.info


def vehicle_size_frame(model: dict, geo: Geometry, lanes: list[dict], width: int, lane_width_m: float,
                       n_c: int = 80, n_x: int = 241) -> RoadFrame:
    a, b, e = model["coef"]
    kfun = lambda x, c: np.exp(a + b * (x - model["x_mid"]) / model["x_scale"] +  # noqa: E731
                               e * (c - model["c_mid"]) / model["c_scale"])
    c_nodes, d_nodes = lateral_nodes(lanes, lane_width_m)
    cg = np.linspace(c_nodes[0], c_nodes[-1], n_c)
    xg = np.linspace(0, width, n_x)
    S = np.zeros((n_c, n_x))
    for i, c in enumerate(cg):
        dy = np.gradient(geo.y_of(np.full_like(xg, c), xg), xg)
        f = kfun(xg, c) * np.sqrt(1 + dy ** 2)
        cum = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(xg))])
        S[i] = cum - np.interp(geo.x_ref, xg, cum)
    return RoadFrame("vehicle_size", s_grid=RegularGridInterpolator((cg, xg), S), d_of_c=(c_nodes, d_nodes),
                     geo=geo, info={"model": model})


# --------------------------------------------------------------------------- lanes in road coordinates
def lane_centres_d(rf: RoadFrame, geo: Geometry, lanes: list[dict]) -> dict[str, float]:
    """Lateral position (metres) of each lane centre, from its image centreline."""
    out = {}
    for l in lanes:
        x = np.linspace(l["x_range"][0], l["x_range"][1], 15)
        _, d = rf.sd(x, geo.y_of(l["c"], x))
        out[l["id"]] = float(np.median(d))
    return out


def save(rf: RoadFrame, path: str | Path, extra: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({**rf.to_dict(), **extra}, indent=1, default=float))


def load_road_frame(path: str | Path) -> RoadFrame:
    """Rebuild a saved calibration (configs/calib/<camera>.json)."""
    from lane_intel.lanes import load_lanes
    d = json.loads(Path(path).read_text())
    if d["kind"] == "gt_homography":
        return RoadFrame("gt_homography", H=np.array(d["H"]), origin=np.array(d["origin"]),
                         u_along=np.array(d["u_along"]), u_across=np.array(d["u_across"]), info=d["info"])
    geo, lanes = load_lanes(d["lanes_file"])
    rf = vehicle_size_frame(d["info"]["model"], geo, lanes, d["frame_width"], d["info"]["lane_width_m"])
    rf.info = d["info"]
    return rf


def calib_path(cfg_path: str | Path) -> Path:
    return Path("configs/calib") / f"{Path(cfg_path).stem}.json"


def main() -> None:
    import argparse
    from datetime import datetime, timezone
    from lane_intel.lanes import lanes_path, load_lanes
    from lane_intel.world import ground_path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--interim", default="data/interim")
    a = ap.parse_args()
    cfg = json.loads(Path(a.cfg).read_text())
    cal = cfg["calibration"]
    W = cfg["frame_size"][0]
    g = pd.read_parquet(ground_path(a.interim, a.cfg))
    lf = lanes_path(cfg, a.cfg)
    geo, lanes = load_lanes(lf)
    extra = {"camera": Path(a.cfg).stem, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "lanes_file": str(lf).replace("\\", "/"), "frame_width": W,
             "frame": "stabilised reference frame (first video frame), full-resolution pixels"}
    if cal["method"] == "gt_homography":
        t0, t1 = cfg["clip_offset_s"] + g.t.min() - 1, cfg["clip_offset_s"] + g.t.max() + 1
        gt = load_gt(cal["gt_csv"], t0, t1)
        rf, info, corr = fit_gt_homography(g, gt, cfg["native_fps"], cfg["clip_offset_s"], W)
        extra["speed_check"] = speed_check(rf, corr)
        corr.to_parquet(Path(a.interim) / f"{Path(a.cfg).stem}_gt_correspondences.parquet", index=False)
    elif cal["method"] == "vehicle_size":
        rf, info = fit_vehicle_size(g, geo, lanes, W, cal["lane_width_m"], cal["car_length_m"])
    else:
        raise SystemExit(f"unknown calibration method {cal['method']!r}")
    d = lane_centres_d(rf, geo, lanes)
    extra["lane_centre_d_m"] = {k: round(v, 2) for k, v in d.items()}
    widths = {}
    for dirn in sorted({l["direction"] for l in lanes}):
        ls = sorted((l for l in lanes if l["direction"] == dirn), key=lambda l: d[l["id"]])
        for p, q in zip(ls, ls[1:]):
            widths[f"{p['id']}-{q['id']}"] = round(abs(d[q["id"]] - d[p["id"]]), 2)
    extra["lane_spacing_m"] = widths
    out = calib_path(a.cfg)
    save(rf, out, extra)
    print(json.dumps({k: v for k, v in {**info, **extra}.items() if k not in ("time_scan_moving_median_err_m", "model")},
                     indent=1, default=float))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
