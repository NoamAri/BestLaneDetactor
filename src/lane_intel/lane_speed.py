"""Per-lane space-mean speed, the Edie way (notes/lane_speed_plan.md §3, §5).

For a region R = lane x road stretch [s0, s1] (length L) x time window [T0, T1] (length T):

    D = sum over trajectory pieces of the distance travelled inside R   (metres, in the direction of travel)
    tau = sum over trajectory pieces of the time spent inside R          (seconds)
    v = D / tau (space-mean speed), k = tau / (L T) (density), q = D / (L T) (flow)

Each straight piece between two consecutive boxes is clipped exactly to R (both in s and in t), so
vehicles entering or leaving mid-window, stopped vehicles and split IDs contribute exactly what was
observed. Distance is the signed displacement along the lane's direction of travel, so the position
jitter of a standing vehicle cancels instead of adding fake distance.

Stretch (per direction, the same for all its ranked lanes): the s range every ranked lane covers
(1st-99th percentile of its boxes), minus margin_frac at each end, split into coverage_bins bins; the
stretch is the longest contiguous run of bins where every lane has at least min_bin_coverage_frac of
its median box count (poor detection: trees, image edge; amendment A5 - the pre-registered rule only
trimmed from the ends). At least min_length_m, else flagged.

Run:  python -m lane_intel.lane_speed --cfg configs/T8_D1.json   (after run.py ground, lanes, calibrate)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HEAVY = ("truck", "bus")


# --------------------------------------------------------------------------- stretch
def choose_stretch(road: pd.DataFrame, lane_ids: list[str], st: dict) -> dict:
    sub = road[road.lane.isin(lane_ids)]
    if sub.empty:
        return {"s0": np.nan, "s1": np.nan, "length_m": 0.0, "ok": False, "lanes": lane_ids}
    q = sub.groupby("lane").s.quantile([0.01, 0.99]).unstack()
    lo, hi = float(q[0.01].max()), float(q[0.99].min())
    m = st["margin_frac"] * max(0.0, hi - lo)
    a, b = lo + m, hi - m
    trimmed = 0
    if b > a:
        # Amendment A5: the longest contiguous run of bins where every lane has at least
        # min_bin_coverage_frac of its median box count (poor zones may lie inside, e.g. trees).
        edges = np.linspace(a, b, st["coverage_bins"] + 1)
        cnt = (sub[(sub.s >= a) & (sub.s <= b)].assign(bin=lambda x: np.clip(np.digitize(x.s, edges) - 1, 0, len(edges) - 2))
               .groupby(["lane", "bin"]).size().unstack(fill_value=0).reindex(columns=range(len(edges) - 1), fill_value=0))
        good = ~(cnt.lt(st["min_bin_coverage_frac"] * cnt.median(axis=1), axis=0)).any(axis=0).to_numpy()
        best, cur = (0, -1), None
        for k, gk in enumerate(np.r_[good, False]):
            if gk and cur is None:
                cur = k
            elif not gk and cur is not None:
                if k - cur > best[1] - best[0] + 1:
                    best = (cur, k - 1)
                cur = None
        i, j = best
        trimmed = int(len(good) - (j - i + 1)) if j >= i else len(good)
        a, b = (float(edges[i]), float(edges[j + 1])) if j >= i else (a, a)
    length = max(0.0, b - a)
    return {"s0": round(a, 2), "s1": round(b, 2), "length_m": round(length, 2), "ok": length >= st["min_length_m"],
            "covered_common": [round(lo, 2), round(hi, 2)], "excluded_bins": trimmed, "lanes": lane_ids}


# --------------------------------------------------------------------------- Edie sums
def clip_fraction(t_a, t_b, s_a, s_b, s0, s1, T0, T1) -> np.ndarray:
    """Fraction of each straight piece (t_a,s_a)->(t_b,s_b) that lies inside [s0,s1] x [T0,T1]."""
    ds, dt = s_b - s_a, t_b - t_a
    with np.errstate(divide="ignore", invalid="ignore"):
        u1s, u2s = (s0 - s_a) / ds, (s1 - s_a) / ds
        u1t, u2t = (T0 - t_a) / dt, (T1 - t_a) / dt
    flat = np.abs(ds) < 1e-12
    inside_s = (s_a >= s0) & (s_a <= s1)
    lo_s = np.where(flat, np.where(inside_s, 0.0, 2.0), np.minimum(u1s, u2s))
    hi_s = np.where(flat, np.where(inside_s, 1.0, -1.0), np.maximum(u1s, u2s))
    still = dt <= 0
    lo_t = np.where(still, 2.0, u1t)
    hi_t = np.where(still, -1.0, u2t)
    u0 = np.maximum.reduce([np.zeros_like(lo_s), lo_s, lo_t])
    u1 = np.minimum.reduce([np.ones_like(hi_s), hi_s, hi_t])
    return np.clip(u1 - u0, 0.0, 1.0)


def edie_vehicles(segs: pd.DataFrame, s0: float, s1: float, T0: float, T1: float) -> pd.DataFrame:
    """Per (lane, seg_id): distance D (m, direction of travel) and time tau (s) inside the region."""
    w = clip_fraction(segs.t_a.to_numpy(), segs.t_b.to_numpy(), segs.s_a.to_numpy(), segs.s_b.to_numpy(), s0, s1, T0, T1)
    m = w > 0
    x = segs[m]
    D = x.sign.to_numpy() * (x.s_b - x.s_a).to_numpy() * w[m]
    tau = (x.t_b - x.t_a).to_numpy() * w[m]
    gap = x.gap.to_numpy()
    v = pd.DataFrame({"lane": x.lane.to_numpy(), "seg_id": x.seg_id.to_numpy(), "cls": x.cls.to_numpy(),
                      "D": D, "tau": tau, "tau_gap": tau * gap})
    return v.groupby(["lane", "seg_id"], as_index=False).agg(cls=("cls", "first"), D=("D", "sum"),
                                                             tau=("tau", "sum"), tau_gap=("tau_gap", "sum"))


def lane_table(veh: pd.DataFrame, lanes: list[dict], L: float, T: float, ev: dict,
               changes: pd.DataFrame | None = None, min_vehicle_tau: float = 1.0) -> pd.DataFrame:
    rows = []
    for l in lanes:
        v = veh[veh.lane == l["id"]]
        D, tau = float(v.D.sum()), float(v.tau.sum())
        long = v[v.tau >= min_vehicle_tau]
        rows.append({"lane": l["id"], "number": l["number"], "direction": l["direction"],
                     "auxiliary": bool(l.get("auxiliary", False)),
                     "vehicles": int((v.tau > 0).sum()), "D_m": D, "tau_s": tau,
                     "v_kmh": D / tau * 3.6 if tau > 0 else np.nan,
                     "median_vehicle_kmh": float(np.median(long.D / long.tau) * 3.6) if len(long) else np.nan,
                     "k_veh_km": tau / (L * T) * 1000 if L > 0 else np.nan,
                     "q_veh_h": D / (L * T) * 3600 if L > 0 else np.nan,
                     "heavy_share": float(v[v.cls.isin(HEAVY)].tau.sum() / tau) if tau > 0 else np.nan,
                     "gap_share": float(v.tau_gap.sum() / tau) if tau > 0 else np.nan,
                     "changes_in": int((changes.to_lane == l["id"]).sum()) if changes is not None else 0,
                     "changes_out": int((changes.from_lane == l["id"]).sum()) if changes is not None else 0,
                     "evidence_ok": bool((v.tau > 0).sum() >= ev["min_vehicles"] and tau >= ev["min_vehicle_time_s"])})
    return pd.DataFrame(rows)


def windows_of(t_min: float, t_max: float, window_s: float, step_s: float) -> list[tuple[float, float]]:
    if t_max - t_min < window_s:
        return []
    starts = np.arange(t_min, t_max - window_s + 1e-9, step_s)
    return [(float(a), float(a + window_s)) for a in starts]


def measure(segs: pd.DataFrame, changes: pd.DataFrame, lanes: list[dict], stretches: dict[str, dict],
            t_span: tuple[float, float], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All rolling windows plus the whole span ("all"). Returns (lane rows, vehicle rows)."""
    wins = [("all", *t_span)] + [(f"w{a:07.2f}", a, b) for a, b in windows_of(*t_span, cfg["window_s"], cfg["step_s"])]
    out_l, out_v = [], []
    for direction in sorted(stretches):
        st = stretches[direction]
        dl = [l for l in lanes if l["direction"] == direction]
        sg = segs[segs.lane.isin([l["id"] for l in dl])]
        if not np.isfinite(st["s0"]):
            continue
        for wid, T0, T1 in wins:
            veh = edie_vehicles(sg, st["s0"], st["s1"], T0, T1)
            ch = changes[(changes.t >= T0) & (changes.t < T1)]
            tab = lane_table(veh, dl, st["length_m"], T1 - T0, cfg["evidence"], ch)
            out_l.append(tab.assign(window=wid, t0=T0, t1=T1, stretch_ok=st["ok"]))
            out_v.append(veh.assign(window=wid, direction=direction))
    return pd.concat(out_l, ignore_index=True), pd.concat(out_v, ignore_index=True)


# --------------------------------------------------------------------------- pipeline
def auxiliary_lanes(spec, road: pd.DataFrame, corr_path: Path | None) -> tuple[list[str], dict]:
    """Lanes measured but not ranked. spec: a list of lane ids, or {"from_ground_truth_lanes": [...]}
    (our lanes whose boxes map mostly to those ground-truth lanes)."""
    if isinstance(spec, list):
        return spec, {}
    if corr_path is None or not Path(corr_path).exists():
        raise SystemExit(f"auxiliary lanes from ground truth need {corr_path}: run  python run.py calibrate")
    corr = pd.read_parquet(corr_path)
    m = corr.merge(road[["track_id", "t", "lane"]], on=["track_id", "t"])
    m = m[m.lane != ""]
    major = m.groupby("lane").gt_lane.agg(lambda s: int(s.mode().iloc[0]))
    share = m.groupby("lane").apply(lambda g: float((g.gt_lane == g.gt_lane.mode().iloc[0]).mean()), include_groups=False)
    aux = [l for l, g in major.items() if g in spec["from_ground_truth_lanes"]]
    return aux, {"our_lane_to_gt_lane": major.to_dict(), "agreement": share.round(3).to_dict()}


def compute(cfg: dict, ls: dict, g: pd.DataFrame, geo, lanes: list[dict], rf, corr_path: Path | None) -> dict:
    """The whole measurement for one camera from its ground table (lane_frame filled)."""
    from lane_intel import announce, lane_pieces, world
    from lane_intel.calibrate import lane_centres_d
    lanes = [dict(l) for l in lanes]
    lane_d = lane_centres_d(rf, geo, lanes)
    road = world.road_points(g, rf, lanes, lane_d, ls)
    aux, aux_info = auxiliary_lanes(cfg["lane_speed"]["auxiliary_lanes"], road, corr_path)
    for l in lanes:
        l["auxiliary"] = l["id"] in aux
    lane_dir = {l["id"]: l["direction"] for l in lanes}
    segs, changes = lane_pieces.build(road, lane_d, lane_dir, ls["pieces"]["min_dwell_s"])
    segs = add_sign_and_gaps(segs, lanes, road.t)
    stretches = {}
    for direction in sorted(set(lane_dir.values())):
        ranked = [l["id"] for l in lanes if l["direction"] == direction and not l["auxiliary"]]
        stretches[direction] = choose_stretch(road, ranked, ls["stretch"])
    t_span = (float(road.t.min()), float(road.t.max()))
    lanes_tab, veh = measure(segs, changes, lanes, stretches, t_span, ls)
    ann = announce.announce_all(lanes_tab, veh, ls["announce"], cfg["lane_speed"].get("speed_limit_kmh"),
                                cfg["calibration"].get("lane_scale_sd", 0.0))
    return {"road": road, "segs": segs, "changes": changes, "lanes": lanes, "lane_d": lane_d, "aux": aux,
            "aux_info": aux_info, "stretches": stretches, "t_span": t_span, "lanes_tab": lanes_tab, "veh": veh,
            "ann": ann}


def add_sign_and_gaps(segs: pd.DataFrame, lanes: list[dict], t: pd.Series) -> pd.DataFrame:
    sign = {l["id"]: (1 if l["direction"] == "right" else -1) for l in lanes}
    dt_med = float(np.median(np.diff(np.sort(pd.unique(t)))))
    return segs.assign(sign=segs.lane.map(sign), gap=((segs.t_b - segs.t_a) > 3.5 * dt_med).astype(float))


def load_inputs(cfg_path: str | Path, interim: str | Path = "data/interim", settings: str | Path = "configs/lane_speed.json"):
    from lane_intel import world
    from lane_intel.calibrate import calib_path, load_road_frame
    from lane_intel.lanes import lanes_path, load_lanes
    cfg = json.loads(Path(cfg_path).read_text())
    ls = json.loads(Path(settings).read_text())
    geo, lanes = load_lanes(lanes_path(cfg, cfg_path))
    rf = load_road_frame(calib_path(cfg_path))
    g = pd.read_parquet(world.ground_path(interim, cfg_path))
    if "lane_frame" not in g:
        raise SystemExit("ground table has no lanes: run  python run.py lanes --cfg " + str(cfg_path))
    corr = Path(interim) / f"{Path(cfg_path).stem}_gt_correspondences.parquet"
    return cfg, ls, g, geo, lanes, rf, corr


def run(cfg_path: str | Path, interim: str | Path = "data/interim", out_root: str | Path = "data/out/lane_speed",
        settings: str | Path = "configs/lane_speed.json") -> dict:
    cfg, ls, g, geo, lanes, rf, corr = load_inputs(cfg_path, interim, settings)
    r = compute(cfg, ls, g, geo, lanes, rf, corr)
    cam = Path(cfg_path).stem
    out = Path(out_root) / cam
    out.mkdir(parents=True, exist_ok=True)
    r["lanes_tab"].to_parquet(out / "lanes.parquet", index=False)
    r["veh"].to_parquet(out / "vehicles.parquet", index=False)
    r["changes"].to_parquet(out / "lane_changes.parquet", index=False)
    r["ann"].to_parquet(out / "announcements.parquet", index=False)
    road = r["road"]
    lt, ann = r["lanes_tab"], r["ann"]
    summary = {
        "camera": cam, "span_s": [round(r["t_span"][0], 2), round(r["t_span"][1], 2)],
        "boxes": int(len(g)), "occluded_boxes_dropped": int(len(g) - len(road)),
        "tracks": int(road.track_id.nunique()), "plausibility_cuts": int(road.seg_id.nunique() - road.track_id.nunique()),
        "lane_width_m": round(road.attrs["lane_width_m"], 2),
        "lane_centre_d_m": {k: round(v, 2) for k, v in r["lane_d"].items()},
        "auxiliary_lanes": r["aux"], "auxiliary_rule": r["aux_info"],
        "stretches": r["stretches"], "lane_changes": int(len(r["changes"])),
        "rolling_windows": int((lt.window != "all").groupby(lt.direction).sum().max()) // max(1, len(r["lanes"])),
        "whole_span": lt[lt.window == "all"].drop(columns=["window"]).round(3).to_dict("records"),
        "whole_span_announcement": ann[ann.window == "all"].drop(columns=["window"]).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return summary


def viewer_payload(out_dir: str | Path) -> dict | None:
    """Compact lane-speed results for the browser viewer (None if not computed)."""
    out_dir = Path(out_dir)
    if not (out_dir / "summary.json").exists():
        return None
    summ = json.loads((out_dir / "summary.json").read_text())
    lt = pd.read_parquet(out_dir / "lanes.parquet")
    ann = pd.read_parquet(out_dir / "announcements.parquet")
    r = lambda x, n=2: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)  # noqa: E731
    whole = lt[lt.window == "all"]
    ci = {}
    for a in ann[ann.window == "all"].itertuples():
        ci.update(json.loads(a.lane_ci_json))
    lanes = [{"lane": x.lane, "number": int(x.number), "direction": x.direction, "auxiliary": bool(x.auxiliary),
              "v_kmh": r(x.v_kmh, 1), "ci": ci.get(x.lane), "median_vehicle_kmh": r(x.median_vehicle_kmh, 1),
              "vehicles": int(x.vehicles), "tau_s": r(x.tau_s, 0), "heavy_share": r(x.heavy_share, 3),
              "k_veh_km": r(x.k_veh_km, 1), "q_veh_h": r(x.q_veh_h, 0), "changes_in": int(x.changes_in),
              "changes_out": int(x.changes_out), "evidence_ok": bool(x.evidence_ok)} for x in whole.itertuples()]
    announce_ = {}
    for a in ann[ann.window == "all"].itertuples():
        announce_[a.direction] = {"status": a.status, "fastest": a.fastest, "slowest": a.slowest,
                                  **{f"{t}_{k}": (r(getattr(a, f"{t}_{k}"), 3) if k != "candidate" else getattr(a, f"{t}_candidate"))
                                     for t in ("fastest", "slowest") for k in ("candidate", "p", "ratio", "ratio_ci_lo", "ratio_ci_hi")},
                                  "fastest_calibration_sensitive": bool(a.fastest_calibration_sensitive),
                                  "slowest_calibration_sensitive": bool(a.slowest_calibration_sensitive)}
    roll = lt[lt.window != "all"]
    timeline = {}
    for d, g in roll.groupby("direction"):
        piv = g.pivot_table(index="t0", columns="lane", values="v_kmh")
        a = ann[(ann.window != "all") & (ann.direction == d)].set_index("t0").reindex(piv.index)
        timeline[d] = {"t0": [r(t, 2) for t in piv.index], "window_s": r(g.t1.iat[0] - g.t0.iat[0], 1),
                       "v_kmh": {l: [r(v, 1) for v in piv[l]] for l in piv.columns},
                       "fastest": [x if isinstance(x, str) else None for x in a.fastest],
                       "slowest": [x if isinstance(x, str) else None for x in a.slowest]}
    val = None
    if (out_dir / "validation.json").exists():
        v = json.loads((out_dir / "validation.json").read_text())["main"]
        val = {"median_rel_err": v["speed"]["median_rel_err"], "median_abs_err_kmh": v["speed"]["median_abs_err_kmh"],
               "precision": {t: v["announcement_precision"][t]["gt_gap_confirms"] for t in ("fastest", "slowest")},
               "per_lane": {x["lane"]: r(x["v_kmh_gt"], 1) for x in v["speed_by_lane_whole_span"]}}
    return {"span_s": summ["span_s"], "stretches": {d: {k: st[k] for k in ("s0", "s1", "length_m", "ok")}
                                                    for d, st in summ["stretches"].items()},
            "lanes": lanes, "announcement": announce_, "timeline": timeline, "validation": val,
            "lane_changes": summ["lane_changes"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--out", default="data/out/lane_speed")
    a = ap.parse_args()
    s = run(a.cfg, a.interim, a.out)
    print(f"{s['camera']}: {s['tracks']} tracks, {s['plausibility_cuts']} plausibility cuts, "
          f"{s['occluded_boxes_dropped']} occluded boxes dropped, {s['lane_changes']} lane changes")
    for d, st in s["stretches"].items():
        print(f"  stretch {d}: s {st['s0']} .. {st['s1']} ({st['length_m']} m{'' if st['ok'] else ', TOO SHORT'})")
    for r in s["whole_span"]:
        print(f"  {r['lane']:3s} {r['direction']:5s}{' aux' if r['auxiliary'] else '    '}  {r['v_kmh']:6.1f} km/h  "
              f"(median vehicle {r['median_vehicle_kmh']:6.1f})  vehicles {r['vehicles']:4d}  "
              f"tau {r['tau_s']:7.0f} s  heavy {r['heavy_share']:.0%}  evidence {'ok' if r['evidence_ok'] else 'LOW'}")
    for r in s["whole_span_announcement"]:
        print(f"  {r['direction']}: {r['status']}")


if __name__ == "__main__":
    main()
