"""Lane speeds checked against MiTra ground truth (notes/lane_speed_plan.md §7).

The ground-truth trajectories go through the same measurement as ours: into road coordinates with
the calibration (and its clock offset), lanes mapped to ours (each ground-truth lane -> our lane whose
boxes it holds most), the same stretch, windows, Edie sums and announcement rule.

Reported (targets from the plan):
  * per lane and window: absolute and relative error of v (target: relative <= 10%)
  * ratios between lanes of a direction vs ground truth (expected smaller than the speed error)
  * ranking agreement (fastest, slowest by point estimate) and announcement agreement
  * announcement precision: when we announce, is the ground-truth gap really >= the threshold, with the
    same lane leading? (target >= 90%)
  * each lane's share of distance and time vs ground truth (lane-assignment bias)
  * ablation: the same errors with a step switched off

Run:  python -m lane_intel.validate --cfg configs/T8_D1.json   (after python run.py speeds)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lane_intel import announce, lane_pieces, lane_speed

VEH_CLASS = {"car": "car", "medium vehicle": "truck", "heavy vehicle": "truck", "truck": "truck",
             "bus": "bus", "motorcycle": "motorcycle"}


def gt_road(gt: pd.DataFrame, rf, clip_offset_s: float, time_offset_s: float, lane_map: dict) -> pd.DataFrame:
    w = gt[["x", "y"]].to_numpy() - rf.origin
    return pd.DataFrame({"seg_id": gt.vid.to_numpy(), "track_id": gt.vid.to_numpy(),
                         "t": gt.t.to_numpy() - clip_offset_s - time_offset_s,
                         "s": w @ rf.u_along, "d": w @ rf.u_across,
                         "lane": gt.lane.map(lane_map).fillna("").to_numpy(),
                         "cls": gt.type.astype(str).str.lower().map(VEH_CLASS).fillna("car").to_numpy()})


def measure_gt(cfg: dict, ls: dict, ours: dict, rf, gt: pd.DataFrame) -> dict:
    our_to_gt = ours["aux_info"]["our_lane_to_gt_lane"]
    lane_map = {}
    for o, g in sorted(our_to_gt.items()):
        lane_map.setdefault(g, o)                         # several of our lanes in one GT lane: see merged_view
    road = gt_road(gt, rf, cfg["clip_offset_s"], rf.info["time_offset_s"], lane_map)
    road = road[(road.t >= ours["t_span"][0]) & (road.t <= ours["t_span"][1])]
    lane_dir = {l["id"]: l["direction"] for l in ours["lanes"]}
    segs, changes = lane_pieces.build(road, ours["lane_d"], lane_dir, ls["pieces"]["min_dwell_s"])
    segs = lane_speed.add_sign_and_gaps(segs, ours["lanes"], road.t)
    lanes_tab, veh = lane_speed.measure(segs, changes, ours["lanes"], ours["stretches"], ours["t_span"], ls)
    ann = announce.announce_all(lanes_tab, veh, ls["announce"], cfg["lane_speed"].get("speed_limit_kmh"), 0.0)
    return {"lanes_tab": lanes_tab, "veh": veh, "ann": ann, "lane_map": lane_map}


def merged_view(cfg: dict, ls: dict, ours: dict) -> dict:
    """Our measurement with lanes that share one ground-truth lane merged (like-for-like comparison)."""
    our_to_gt = ours["aux_info"]["our_lane_to_gt_lane"]
    first = {}
    for o, g in sorted(our_to_gt.items()):
        first.setdefault(g, o)
    rename = {o: first[g] for o, g in our_to_gt.items()}
    if all(k == v for k, v in rename.items()):
        return {**ours, "merged": {}}
    lanes = [l for l in ours["lanes"] if rename.get(l["id"], l["id"]) == l["id"]]
    segs = ours["segs"].assign(lane=ours["segs"].lane.map(lambda x: rename.get(x, x)))
    ch = ours["changes"].assign(from_lane=ours["changes"].from_lane.map(lambda x: rename.get(x, x)),
                                to_lane=ours["changes"].to_lane.map(lambda x: rename.get(x, x)))
    ch = ch[ch.from_lane != ch.to_lane]
    lanes_tab, veh = lane_speed.measure(segs, ch, lanes, ours["stretches"], ours["t_span"], ls)
    ann = announce.announce_all(lanes_tab, veh, ls["announce"], cfg["lane_speed"].get("speed_limit_kmh"),
                                cfg["calibration"].get("lane_scale_sd", 0.0))
    return {**ours, "lanes": lanes, "segs": segs, "changes": ch, "lanes_tab": lanes_tab, "veh": veh, "ann": ann,
            "merged": {k: v for k, v in rename.items() if k != v}}


def compare(ours: dict, ref: dict, gap: float) -> dict:
    key = ["window", "direction", "lane"]
    a = ours["lanes_tab"].set_index(key)
    b = ref["lanes_tab"].set_index(key)
    j = a[["v_kmh", "D_m", "tau_s", "evidence_ok", "auxiliary"]].join(
        b[["v_kmh", "D_m", "tau_s", "evidence_ok"]], rsuffix="_gt", how="inner").reset_index()
    j = j[j.evidence_ok & j.evidence_ok_gt & j.v_kmh.notna() & j.v_kmh_gt.notna()]
    j["abs_err_kmh"] = j.v_kmh - j.v_kmh_gt
    j["rel_err"] = j.abs_err_kmh / j.v_kmh_gt
    roll = j[j.window != "all"]
    # shares of distance and time per lane (whole span)
    w = j[j.window == "all"].copy()
    for c in ("D_m", "tau_s"):
        for suf in ("", "_gt"):
            w[f"{c}{suf}_share"] = w[f"{c}{suf}"] / w.groupby("direction")[f"{c}{suf}"].transform("sum")
    # ratios between ranked lanes, same window and direction
    rat = []
    for (win, dirn), g in j[~j.auxiliary].groupby(["window", "direction"]):
        g = g.set_index("lane")
        for x in g.index:
            for y in g.index:
                if x < y:
                    r, rg = g.v_kmh[x] / g.v_kmh[y], g.v_kmh_gt[x] / g.v_kmh_gt[y]
                    rat.append({"window": win, "direction": dirn, "pair": f"{x}/{y}", "ratio": r, "ratio_gt": rg,
                                "rel_err": r / rg - 1})
    rat = pd.DataFrame(rat)
    # rankings and announcements
    A = ours["ann"].set_index(["window", "direction"])
    B = ref["ann"].set_index(["window", "direction"])
    jj = A.join(B, rsuffix="_gt", how="inner").reset_index()
    out_rank, prec = {}, {}
    for title in ("fastest", "slowest"):
        out_rank[title] = {
            "point_leader_agrees": round(float((jj[f"{title}_candidate"] == jj[f"{title}_candidate_gt"]).mean()), 3),
            "rule_agrees": round(float((jj[f"{title}_rule"].fillna("-") == jj[f"{title}_rule_gt"].fillna("-")).mean()), 3),
            "held_title_agrees": round(float((jj[title].fillna("-") == jj[f"{title}_gt"].fillna("-")).mean()), 3),
            "we_announce": int(jj[title].notna().sum()), "gt_announces": int(jj[f"{title}_gt"].notna().sum())}
        ann = jj[jj[title].notna()]
        ok = []
        for r in ann.itertuples():
            g = j[(j.window == r.window) & (j.direction == r.direction) & ~j.auxiliary].set_index("lane").v_kmh_gt
            lane = getattr(r, title)
            if lane not in g.index or len(g) < 2:
                ok.append(False)
                continue
            others = g.drop(lane)
            ratio = g[lane] / others.max() if title == "fastest" else others.min() / g[lane]
            ok.append(bool(ratio >= 1 + gap))
        prec[title] = {"announcements": len(ok), "gt_gap_confirms": round(float(np.mean(ok)), 3) if ok else None}
    return {
        "speed": {"lane_windows": int(len(roll)),
                  "median_abs_err_kmh": round(float(roll.abs_err_kmh.abs().median()), 2),
                  "median_rel_err": round(float(roll.rel_err.abs().median()), 4),
                  "p90_rel_err": round(float(roll.rel_err.abs().quantile(0.9)), 4),
                  "share_within_10pct": round(float((roll.rel_err.abs() <= 0.10).mean()), 3),
                  "bias_rel": round(float(roll.rel_err.median()), 4)},
        "speed_by_lane_whole_span": w[["direction", "lane", "auxiliary", "v_kmh", "v_kmh_gt", "abs_err_kmh", "rel_err",
                                       "D_m_share", "D_m_gt_share", "tau_s_share", "tau_s_gt_share"]].round(4).to_dict("records"),
        "ratios": {"pairs": int(len(rat[rat.window != "all"])),
                   "median_rel_err": round(float(rat[rat.window != "all"].rel_err.abs().median()), 4) if len(rat) else None},
        "ranking": out_rank, "announcement_precision": prec,
    }


def run(cfg_path: str | Path, interim: str | Path = "data/interim", out_root: str | Path = "data/out/lane_speed",
        settings: str | Path = "configs/lane_speed.json", ablations: bool = True) -> dict:
    from lane_intel.calibrate import load_gt
    cfg, ls, g, geo, lanes, rf, corr = lane_speed.load_inputs(cfg_path, interim, settings)
    ours = lane_speed.compute(cfg, ls, g, geo, lanes, rf, corr)
    t0, t1 = cfg["clip_offset_s"] + ours["t_span"][0] - 1, cfg["clip_offset_s"] + ours["t_span"][1] + 1
    gt = load_gt(cfg["calibration"]["gt_csv"], t0, t1)
    ours = merged_view(cfg, ls, ours)
    ref = measure_gt(cfg, ls, ours, rf, gt)
    gap = ls["announce"]["min_gap"]
    res = {"camera": Path(cfg_path).stem, "span_s": ours["t_span"], "gt_lane_to_our_lane": ref["lane_map"],
           "merged_for_comparison": ours["merged"],
           "lane_assignment_agreement": ours["aux_info"]["agreement"], "stretches": ours["stretches"],
           "main": compare(ours, ref, gap)}
    if ablations:
        variants = {
            "no occlusion filter": {**ls, "occlusion_height_frac": 0.0},
            "no lane hysteresis": {**ls, "pieces": {"min_dwell_s": 0.0}},
            "with plausibility cuts (pre-registered)": {**ls, "plausibility": {**ls["plausibility"], "enabled": True}},
            "with plausibility cuts, step + lateral only": {**ls, "plausibility": {**ls["plausibility"], "enabled": True,
                                                                                   "max_accel_ms2": 1e9}},
        }
        res["ablation"] = {}
        for name, lsv in variants.items():
            o = merged_view(cfg, lsv, lane_speed.compute(cfg, lsv, g, geo, lanes, rf, corr))
            c = compare(o, ref, gap)
            res["ablation"][name] = {"speed": c["speed"], "ratios": c["ratios"], "ranking": c["ranking"],
                                     "announcement_precision": c["announcement_precision"]}
    out = Path(out_root) / res["camera"]
    out.mkdir(parents=True, exist_ok=True)
    ref["lanes_tab"].to_parquet(out / "gt_lanes.parquet", index=False)
    ref["ann"].to_parquet(out / "gt_announcements.parquet", index=False)
    (out / "validation.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()
    r = run(a.cfg, a.interim, ablations=not a.no_ablation)
    m = r["main"]
    print(json.dumps({k: m[k] for k in ("speed", "ratios", "ranking", "announcement_precision")}, indent=1))
    print("whole span per lane:")
    for x in m["speed_by_lane_whole_span"]:
        print(f"  {x['lane']:3s}{' aux' if x['auxiliary'] else '    '} ours {x['v_kmh']:6.1f}  gt {x['v_kmh_gt']:6.1f} km/h "
              f"({x['rel_err']:+.1%})  distance share {x['D_m_share']:.2f} vs {x['D_m_gt_share']:.2f}  "
              f"time share {x['tau_s_share']:.2f} vs {x['tau_s_gt_share']:.2f}")
    for name, v in r.get("ablation", {}).items():
        print(f"ablation '{name}': median rel err {v['speed']['median_rel_err']:.2%}, within 10% "
              f"{v['speed']['share_within_10pct']:.0%}, precision {v['announcement_precision']}")


if __name__ == "__main__":
    main()
