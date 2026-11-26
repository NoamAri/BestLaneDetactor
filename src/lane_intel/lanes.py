"""Phase 2 (lanes): find each lane from where vehicles actually drive, and assign vehicles to lanes.

Vehicles follow their lanes, so the paths of tracked vehicles reveal the lanes. No manual drawing
is needed; the result is saved as an editable JSON file per camera.

Geometry: lanes are one family of lines. Either every lane is the same curve shifted up or down
(parallel lanes, straight or following a curved road), or straight lanes meet at a vanishing point
(strong perspective). The family is chosen per camera by cross-validation on vehicle paths.

Method
  1. Fit each candidate family to long, moving tracks (they travel at least ``min_dx_boxes`` box
     widths): straight parallel, curved parallel (degree 2 or 3; one shared shape, one offset per
     track, least squares), and vanishing point (least-squares intersection of per-track lines).
  2. Lateral coordinate of any point: follow the line from the VP through the point and read its
     height c at the reference column x_ref (image centre). Every point of one lane has the same c,
     whatever its distance from the camera, so perspective is removed. A track's c is the median
     over its boxes, so stopped vehicles count too.
     Model choice: 2-fold cross-validation over tracks. Fit on half the vehicles, measure on the
     other half how far each vehicle's boxes stray from its lane line (image pixels); the simplest
     family within 2% of the best wins. A more flexible model only wins if it predicts better.
  3. Per direction (sign of net motion), a smoothed histogram of track c values has one peak per
     lane (one vote per track). Pass 1 uses moving tracks only. Pass 2 adds barely moving tracks,
     which cannot tell their direction but sit inside one carriageway: each joins the direction
     whose pass-1 extent (lanes +/- a lane spacing) contains it, or the nearest. This recovers a
     lane where traffic stands still (no moving votes in a jam).
  4. Number lanes per direction: lane 1 = leftmost in the direction of travel. In image terms a
     vehicle moving left (-x) has its left side toward +y (down), so for left-moving traffic lane 1
     is the lowest lane; for right-moving traffic it is the highest. On a road with right-hand
     traffic, lane 1 is therefore the one next to the median.

Assignment: a box belongs to the lane of its vehicle's direction whose c is nearest, if it lies
within that lane's band (half-way to the neighbouring lane in c, or half a gap at the outer edges);
otherwise lane -1. A track's lane is the most frequent lane of its boxes.

Usage
    python -m lane_intel.lanes --cfg configs/T8_D1.json --points data/interim/T8_D1_0-30s_points.parquet [--write-points]
    # writes the file named by the camera config's "lanes_file" (default configs/lanes/<camera>.json);
    # --write-points fills lane_frame / lane_track in the points file for later per-lane measurements

Known limit: a slip road that leaves the motorway at an angle does not share the motorway's lane
shape and is not found as a lane; its vehicles get lane -1 or the nearest motorway lane.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class LaneParams:
    min_obs: int = 6
    min_dx_boxes: float = 3.0         # a track used for the geometry travels at least this many box widths
    still_px_s: float = 15.0          # at 3840 px width; slower tracks count for both directions
    smooth_frac: float = 0.10         # histogram smoothing, fraction of the median box height
    min_sep_frac: float = 0.6         # peaks closer than this x box height are one lane
    min_tracks: int = 4               # a lane needs at least this many tracks
    far_vp: float = 20.0              # VP further than this x image width -> parallel model

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Geometry:
    """Lane-line family. Two kinds:

    shape   every lane is the same curve shifted vertically: y = c + f(x), f(x) = sum a_k (x - x_ref)^k,
            k = 1..deg (deg 1 = straight parallel lanes; 2-3 = a curved road seen without strong
            perspective). c is the lane's height at x_ref.
    vp      straight lanes meeting at a vanishing point (strong perspective along the road).
    """
    x_ref: float
    kind: str = "shape"
    coef: tuple[float, ...] = (0.0,)          # a_1..a_deg (shape)
    vp: tuple[float, float] | None = None     # (X, Y) (vp)

    def f(self, x):
        u = np.asarray(x, float) - self.x_ref
        return sum(a * u ** (k + 1) for k, a in enumerate(self.coef))

    def c_of(self, x, y):
        """Height at x_ref of the lane line through (x, y)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.kind == "vp":
            X, Y = self.vp
            return Y + (y - Y) * (self.x_ref - X) / (x - X)
        return y - self.f(x)

    def y_of(self, c, x):
        """Height at x of the lane line whose height at x_ref is c."""
        x = np.asarray(x, float)
        if self.kind == "vp":
            X, Y = self.vp
            return Y + (np.asarray(c, float) - Y) * (x - X) / (self.x_ref - X)
        return np.asarray(c, float) + self.f(x)

    def describe(self) -> str:
        if self.kind == "vp":
            return f"straight lanes meeting at ({self.vp[0]:.0f}, {self.vp[1]:.0f})"
        return "parallel straight lanes" if len(self.coef) == 1 else f"parallel curved lanes (degree {len(self.coef)})"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "x_ref": self.x_ref, "coef": list(self.coef), "vp": list(self.vp) if self.vp else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return cls(d["x_ref"], d.get("kind", "shape"), tuple(d.get("coef") or (0.0,)), tuple(d["vp"]) if d.get("vp") else None)


def track_table(points: pd.DataFrame, width: int, p: LaneParams) -> pd.DataFrame:
    """Per track: observations, net speed (px/s), direction (+1 right, -1 left, 0 both), median anchor."""
    g = points.sort_values("t").groupby("track_id")
    t = pd.DataFrame({"t0": g.t.first(), "t1": g.t.last(), "n": g.size(), "x0": g.x_px.first(),
                      "x1": g.x_px.last(), "xm": g.x_px.median(), "ym": g.y_px.median(), "w": (g.x2.median() - g.x1.median())})
    span = (t.t1 - t.t0)
    t["v"] = np.where(span > 0, (t.x1 - t.x0) / span.where(span > 0, 1), 0.0)
    still = p.still_px_s * width / 3840
    t["dir"] = np.where(t.v > still, 1, np.where(t.v < -still, -1, 0))
    return t


def _fit_shape(sub: pd.DataFrame, x_ref: float, deg: int) -> tuple[float, ...]:
    """f with one free offset per track: regress within-track-demeaned y on demeaned (x - x_ref)^k."""
    u = sub.x_px.to_numpy() - x_ref
    U = np.stack([u ** k for k in range(1, deg + 1)], 1)
    tid = sub.track_id.to_numpy()
    Ud = U - pd.DataFrame(U).groupby(tid).transform("mean").to_numpy()
    yd = sub.y_px.to_numpy() - pd.Series(sub.y_px.to_numpy()).groupby(tid).transform("mean").to_numpy()
    coef, *_ = np.linalg.lstsq(Ud, yd, rcond=None)
    return tuple(float(c) for c in coef)


def _fit_vp(sub: pd.DataFrame, x_ref: float, width: int, p: LaneParams) -> Geometry | None:
    lines = []
    for _, g in sub.groupby("track_id"):
        m, b = np.polyfit(g.x_px, g.y_px, 1)
        lines.append((m, b, len(g)))
    if len(lines) < 2:
        return None
    L = np.array(lines)
    m, b, w = L[:, 0], L[:, 1], np.sqrt(L[:, 2])
    n = np.stack([m, -np.ones_like(m)], 1) / np.sqrt(m ** 2 + 1)[:, None]
    d = -b / np.sqrt(m ** 2 + 1)
    A = (n * w[:, None]).T @ n
    if np.linalg.cond(A) > 1e6:
        return None
    X, Y = np.linalg.solve(A, (n * w[:, None]).T @ d)
    if abs(X - x_ref) > p.far_vp * width or (min(sub.x_px) <= X <= max(sub.x_px)):
        return None                                   # parallel in practice, or a VP inside the road
    return Geometry(x_ref, "vp", (0.0,), (float(X), float(Y)))


def _residual(geo: Geometry, sub: pd.DataFrame) -> float:
    """Median over tracks of the spread (image px) of a track's boxes around its lane line."""
    c = pd.Series(geo.c_of(sub.x_px, sub.y_px), index=sub.index)
    ct = c.groupby(sub.track_id).transform("median")
    r = pd.Series(sub.y_px.to_numpy() - geo.y_of(ct.to_numpy(), sub.x_px.to_numpy()), index=sub.index)
    return float(r.groupby(sub.track_id).std().median())


def estimate_geometry(points: pd.DataFrame, tracks: pd.DataFrame, width: int, p: LaneParams) -> tuple[Geometry, dict]:
    """Choose the lane-line family by 2-fold cross-validation over long tracks."""
    x_ref = width / 2
    long = tracks[(tracks.n >= p.min_obs) & ((tracks.x1 - tracks.x0).abs() >= p.min_dx_boxes * tracks.w)]
    sub = points[points.track_id.isin(long.index)]
    if long.shape[0] < 4:
        return Geometry(x_ref), {}
    fold = sub.track_id.map(lambda t: hash(int(t)) % 2)
    fits = {
        "straight": lambda d: Geometry(x_ref, "shape", _fit_shape(d, x_ref, 1)),
        "curve2": lambda d: Geometry(x_ref, "shape", _fit_shape(d, x_ref, 2)),
        "curve3": lambda d: Geometry(x_ref, "shape", _fit_shape(d, x_ref, 3)),
        "vp": lambda d: _fit_vp(d, x_ref, width, p),
    }
    scores = {}
    for name, fit in fits.items():
        errs = []
        for k in (0, 1):
            g = fit(sub[fold == k])
            if g is None:
                break
            errs.append(_residual(g, sub[fold != k]))
        if len(errs) == 2:
            scores[name] = round(float(np.mean(errs)), 3)
    best = min(scores, key=lambda n: (scores[n], list(fits).index(n)))
    simplest_close = next(n for n in fits if n in scores and scores[n] <= scores[best] * 1.02)  # prefer simpler
    return fits[simplest_close](sub), scores


def estimate_lanes(points: pd.DataFrame, width: int, p: LaneParams = LaneParams()) -> tuple[Geometry, list[dict], dict]:
    tr = track_table(points, width, p)
    geo, scores = estimate_geometry(points, tr, width, p)
    pts = points.copy()
    pts["c"] = geo.c_of(pts.x_px, pts.y_px)
    tr["c"] = pts.groupby("track_id").c.median()
    use = tr[tr.n >= p.min_obs]
    h_box = float((points.y2 - points.y1).median())
    sigma = max(1.0, p.smooth_frac * h_box)
    unit = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    def peaks_of(cs):
        lo, hi = cs.min() - 3 * h_box, cs.max() + 3 * h_box
        bins = np.arange(lo, hi + 1.0, 1.0)
        sm = gaussian_filter1d(np.histogram(cs, bins=bins)[0].astype(float), sigma)
        idx, _ = find_peaks(sm, height=unit * p.min_tracks * 0.8, distance=max(1, int(p.min_sep_frac * h_box)))
        return bins[idx] + 0.5

    # pass 1: moving tracks; pass 2: stopped tracks join the carriageway that contains them
    use = use.copy()
    use["vote"] = use.dir
    extents = {}
    for direction in (-1, 1):
        cs = use[use.dir == direction].c.to_numpy()
        if len(cs) >= p.min_tracks:
            pk = peaks_of(cs)
            if len(pk):
                gap = float(np.median(np.diff(np.sort(pk)))) if len(pk) > 1 else h_box
                extents[direction] = (pk.min() - gap, pk.max() + gap)
    still = use.dir == 0
    if extents and still.any():
        def side(c):
            inside = [d for d, (a, b) in extents.items() if a <= c <= b]
            if len(inside) == 1:
                return inside[0]
            return min(extents, key=lambda d: min(abs(c - extents[d][0]), abs(c - extents[d][1])))
        use.loc[still, "vote"] = use.loc[still, "c"].map(side)

    lanes = []
    for direction, name in ((-1, "left"), (1, "right")):
        cs = use[use.vote == direction].c.to_numpy()
        if len(cs) < p.min_tracks:
            continue
        peaks = peaks_of(cs)
        # lane 1 = leftmost in the direction of travel: lowest (largest c) for left-moving traffic
        peaks = sorted(peaks, reverse=(name == "left"))
        members = use[use.vote == direction]
        for k, c in enumerate(peaks, start=1):
            near = members[np.abs(members.c - c) < 0.5 * p.min_sep_frac * h_box]
            xs = points[points.track_id.isin(near.index)].x_px
            lanes.append({"id": f"{'L' if name == 'left' else 'R'}{k}", "number": k, "direction": name,
                          "c": round(float(c), 2),
                          "x_range": [round(float(xs.min()), 1), round(float(xs.max()), 1)] if len(xs) else [0.0, float(width)],
                          "tracks": int(len(near))})
    return geo, lanes, scores


def lane_edges(lanes: list[dict]) -> dict[str, tuple[float, float]]:
    """Band of each lane in c: half-way to the neighbouring lane of the same direction."""
    out = {}
    for direction in ("left", "right"):
        ls = sorted((l for l in lanes if l["direction"] == direction), key=lambda l: l["c"])
        cs = [l["c"] for l in ls]
        for i, l in enumerate(ls):
            gap_lo = cs[i] - cs[i - 1] if i > 0 else (cs[i + 1] - cs[i] if len(cs) > 1 else 40.0)
            gap_hi = cs[i + 1] - cs[i] if i + 1 < len(cs) else (cs[i] - cs[i - 1] if len(cs) > 1 else 40.0)
            out[l["id"]] = (cs[i] - gap_lo / 2, cs[i] + gap_hi / 2)
    return out


def assign_lanes(points: pd.DataFrame, geo: Geometry, lanes: list[dict], width: int,
                 p: LaneParams = LaneParams()) -> pd.DataFrame:
    """Add lane_frame (index into ``lanes``, -1 outside every lane) and lane_track (track majority)."""
    out = points.copy()
    out["lane_frame"] = -1
    out["lane_track"] = -1
    if not lanes or out.empty:
        return out
    tr = track_table(out, width, p)
    dirs = out.track_id.map(tr.dir).to_numpy()
    c = geo.c_of(out.x_px.to_numpy(), out.y_px.to_numpy())
    edges = lane_edges(lanes)
    best = np.full(len(out), -1)
    best_d = np.full(len(out), np.inf)
    for k, l in enumerate(lanes):
        lo, hi = edges[l["id"]]
        want = 1 if l["direction"] == "right" else -1
        ok = (c >= lo) & (c <= hi) & ((dirs == 0) | (dirs == want))
        d = np.abs(c - l["c"])
        better = ok & (d < best_d)
        best[better], best_d[better] = k, d[better]
    out["lane_frame"] = best
    maj = out[out.lane_frame >= 0].groupby("track_id").lane_frame.agg(lambda s: int(s.mode().iloc[0]))
    out["lane_track"] = out.track_id.map(maj).fillna(-1).astype(int)
    return out


def lane_polygons(geo: Geometry, lanes: list[dict], n: int = 24) -> list[dict]:
    """Drawable lanes: centreline and band edges sampled along each lane's x range (full-frame px)."""
    edges = lane_edges(lanes)
    out = []
    for l in lanes:
        x = np.linspace(l["x_range"][0], l["x_range"][1], n)
        lo, hi = edges[l["id"]]
        out.append({"id": l["id"], "number": l["number"], "direction": l["direction"],
                    "centre": np.round(np.c_[x, geo.y_of(l["c"], x)], 1).tolist(),
                    "upper": np.round(np.c_[x, geo.y_of(lo, x)], 1).tolist(),
                    "lower": np.round(np.c_[x, geo.y_of(hi, x)], 1).tolist()})
    return out


def load_lanes(path: str | Path) -> tuple[Geometry, list[dict]]:
    d = json.loads(Path(path).read_text())
    return Geometry.from_dict(d["geometry"]), d["lanes"]


def lanes_path(cfg: dict, cfg_path: str | Path) -> Path:
    return Path(cfg.get("lanes_file") or Path("configs/lanes") / f"{Path(cfg_path).stem}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--points", required=True, nargs="+",
                    help="tracked points of this camera (one or more windows; track ids must be unique across them)")
    ap.add_argument("--write-points", action="store_true",
                    help="also fill lane_frame (and add lane_track) in the points file (single file only)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.cfg).read_text())
    width = cfg["frame_size"][0]
    pts = pd.concat([pd.read_parquet(f) for f in a.points], ignore_index=True)
    if a.write_points and len(a.points) > 1:
        ap.error("--write-points needs a single points file")
    p = LaneParams()
    geo, lanes, scores = estimate_lanes(pts, width, p)
    assigned = assign_lanes(pts, geo, lanes, width, p)
    share = (assigned.lane_frame >= 0).mean()
    per_lane = assigned[assigned.lane_track >= 0].groupby("lane_track").track_id.nunique()
    for k, l in enumerate(lanes):
        l["vehicles"] = int(per_lane.get(k, 0))
    out = lanes_path(cfg, a.cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "about": "Lanes estimated from vehicle paths (lane_intel.lanes). Each lane is the line through the "
                 "geometry family (see geometry) whose height at x_ref is c, full-frame pixels. "
                 "Lane 1 = leftmost in the direction of travel. Editable by hand.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "camera": Path(a.cfg).stem, "source_points": a.points, "params": p.to_dict(),
        "geometry": geo.to_dict(), "geometry_cv_residual_px": scores, "boxes_in_a_lane": round(float(share), 3), "lanes": lanes}, indent=1))
    if a.write_points:
        assigned.to_parquet(a.points[0], index=False)
    print(f"{len(lanes)} lanes, {geo.describe()} (cross-validated residual px: {scores}) -> {out}  "
          f"({share:.0%} of tracked boxes fall inside a lane)")
    for l in lanes:
        print(f"  {l['id']}: lane {l['number']} {l['direction']:5s}  c={l['c']:7.1f}  vehicles {l['vehicles']:4d}")


if __name__ == "__main__":
    main()
