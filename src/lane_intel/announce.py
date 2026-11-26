"""Which lane is fastest / slowest, and is the difference real? (notes/lane_speed_plan.md §6)

Per window and direction, over the ranked lanes (not auxiliary, enough evidence):

1. Cluster bootstrap: in each of B resamples, draw vehicles with replacement within each lane (a
   vehicle's distance and time together) and recompute every lane's D / tau.
2. P(fastest), P(slowest): share of resamples in which a lane ranks first / last. The ratio of the
   leader to the runner-up has a percentile interval from the same resamples.
3. Announce "fastest: lane a" only if P(fastest) >= min_p_rank, v_a / v_runner-up >= 1 + min_gap and
   the interval of that ratio lies above 1. Same for slowest with the second-slowest lane.
4. Free flow: with a known speed limit, every ranked lane >= free_flow_frac of it -> no announcement.
5. Over rolling windows (hysteresis): a holder keeps its title while its gap to the best other lane
   stays >= keep_gap and the ratio interval stays above 1. Another lane takes over only after meeting
   the full rule continuously for takeover_s of fresh data; the first announcement, or the same lane
   regaining its title, needs no wait.
6. Calibration: the bootstrap is repeated with each lane's scale multiplied by N(1, lane_scale_sd)
   (the measured lane-to-lane calibration error). If the rule then fails, the announcement is marked
   calibration_sensitive.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def bootstrap(veh: pd.DataFrame, lane_ids: list[str], B: int, rng: np.random.Generator,
              scale_sd: float = 0.0) -> np.ndarray:
    """(B, n_lanes) resampled space-mean speeds (m/s)."""
    out = np.empty((B, len(lane_ids)))
    for k, l in enumerate(lane_ids):
        v = veh[(veh.lane == l) & (veh.tau > 0)]
        D, tau = v.D.to_numpy(), v.tau.to_numpy()
        idx = rng.integers(0, len(v), size=(B, len(v)))
        out[:, k] = D[idx].sum(1) / tau[idx].sum(1)
    if scale_sd > 0:
        out *= rng.normal(1.0, scale_sd, size=out.shape)
    return out


def _rule(v: np.ndarray, bs: np.ndarray, ids: list[str], cfg: dict, title: str) -> dict:
    """Evaluate the announcement rule for one title on point speeds v and resamples bs."""
    order = np.argsort(-v) if title == "fastest" else np.argsort(v)
    a, r = order[0], order[1]
    rank = np.argmax(bs, 1) if title == "fastest" else np.argmin(bs, 1)
    p = float((rank == a).mean())
    if title == "fastest":
        ratio, rb = v[a] / v[r], bs[:, a] / bs[:, r]
    else:
        ratio, rb = v[r] / v[a], bs[:, r] / bs[:, a]
    lo, hi = np.nanpercentile(rb, [50 * (1 - cfg["ci"]), 100 - 50 * (1 - cfg["ci"])])
    ok = p >= cfg["min_p_rank"] and ratio >= 1 + cfg["min_gap"] and lo > 1
    return {"lane": ids[a], "other": ids[r], "p": round(p, 4), "ratio": round(float(ratio), 4),
            "ratio_ci": [round(float(lo), 4), round(float(hi), 4)], "meets_rule": bool(ok)}


def _keeps(holder: str, v: np.ndarray, bs: np.ndarray, ids: list[str], cfg: dict, title: str) -> bool:
    if holder not in ids:
        return False
    h = ids.index(holder)
    others = [k for k in range(len(ids)) if k != h]
    if title == "fastest":
        o = max(others, key=lambda k: v[k])
        ratio, rb = v[h] / v[o], bs[:, h] / bs[:, o]
    else:
        o = min(others, key=lambda k: v[k])
        ratio, rb = v[o] / v[h], bs[:, o] / bs[:, h]
    lo = np.nanpercentile(rb, 50 * (1 - cfg["ci"]))
    return bool(ratio >= 1 + cfg["keep_gap"] and lo > 1)


def judge(tab: pd.DataFrame, veh: pd.DataFrame, cfg: dict, speed_limit_kmh: float | None,
          lane_scale_sd: float, rng: np.random.Generator) -> dict:
    """Rule results for one window and direction (no hysteresis)."""
    ranked = tab[(~tab.auxiliary) & tab.evidence_ok & tab.stretch_ok & (tab.v_kmh.notna())]
    ids = ranked.lane.tolist()
    res = {"ranked_lanes": ids, "excluded_low_evidence": tab[(~tab.auxiliary) & ~tab.evidence_ok].lane.tolist()}
    if len(ids) < 2:
        return {**res, "status": "not enough data", "fastest": None, "slowest": None, "_v": None, "_bs": None, "_ids": ids}
    v = ranked.v_kmh.to_numpy() / 3.6
    bs = bootstrap(veh, ids, cfg["bootstrap"], rng)
    res["_v"], res["_bs"], res["_ids"] = v, bs, ids
    if speed_limit_kmh and (ranked.v_kmh >= cfg["free_flow_frac"] * speed_limit_kmh).all():
        return {**res, "status": "free flow, no meaningful difference", "fastest": None, "slowest": None}
    for title in ("fastest", "slowest"):
        r = _rule(v, bs, ids, cfg, title)
        if lane_scale_sd > 0:
            bsc = bs * rng.normal(1.0, lane_scale_sd, size=bs.shape)
            r["calibration_sensitive"] = bool(r["meets_rule"] and not _rule(v, bsc, ids, cfg, title)["meets_rule"])
        res[title] = r
    return res


def _status(f, s) -> str:
    parts = [f"fastest: {f}" if f else None, f"slowest: {s}" if s else None]
    parts = [p for p in parts if p]
    return "; ".join(parts) if parts else "no meaningful difference"


def announce_all(lanes_tab: pd.DataFrame, veh: pd.DataFrame, cfg: dict, speed_limit_kmh: float | None = None,
                 lane_scale_sd: float = 0.0) -> pd.DataFrame:
    """Rule per window and direction, plus the hysteresis holders over the rolling windows."""
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for direction in sorted(lanes_tab.direction.unique()):
        wins = lanes_tab[lanes_tab.direction == direction][["window", "t0", "t1"]].drop_duplicates()
        wins = pd.concat([wins[wins.window == "all"], wins[wins.window != "all"].sort_values("t0")])
        state = {t: {"holder": None, "last": None, "streak": None} for t in ("fastest", "slowest")}
        for w in wins.itertuples():
            tab = lanes_tab[(lanes_tab.window == w.window) & (lanes_tab.direction == direction)]
            vv = veh[(veh.window == w.window) & (veh.direction == direction)]
            j = judge(tab, vv, cfg, speed_limit_kmh, lane_scale_sd, rng)
            row = {"window": w.window, "t0": w.t0, "t1": w.t1, "direction": direction,
                   "ranked_lanes": j["ranked_lanes"], "excluded_low_evidence": j["excluded_low_evidence"],
                   "lane_ci_json": json.dumps({} if j["_bs"] is None else {
                       l: [round(float(x) * 3.6, 2) for x in np.nanpercentile(j["_bs"][:, k], [2.5, 97.5])]
                       for k, l in enumerate(j["_ids"])})}
            held = {}
            for title in ("fastest", "slowest"):
                r = j.get(title)
                row[f"{title}_rule"] = r["lane"] if r and r["meets_rule"] else None
                row[f"{title}_candidate"] = r["lane"] if r else None
                row[f"{title}_other"] = r["other"] if r else None
                row[f"{title}_p"] = r["p"] if r else np.nan
                row[f"{title}_ratio"] = r["ratio"] if r else np.nan
                row[f"{title}_ratio_ci_lo"] = r["ratio_ci"][0] if r else np.nan
                row[f"{title}_ratio_ci_hi"] = r["ratio_ci"][1] if r else np.nan
                row[f"{title}_calibration_sensitive"] = bool(r.get("calibration_sensitive", False)) if r else False
                if w.window == "all":
                    held[title] = row[f"{title}_rule"]
                    continue
                s = state[title]
                q = row[f"{title}_rule"]
                s["streak"] = (s["streak"] if s["streak"] and s["streak"][0] == q else (q, w.t0)) if q else None
                if s["holder"] and not (j["_v"] is not None and _keeps(s["holder"], j["_v"], j["_bs"], j["_ids"], cfg, title)):
                    s["last"], s["holder"] = s["holder"], None
                if s["holder"] is None and q:
                    fresh = s["streak"] and w.t0 - s["streak"][1] >= cfg["takeover_s"] - 1e-9
                    if s["last"] is None or q == s["last"] or fresh:
                        s["holder"] = q
                held[title] = s["holder"]
            row["fastest"], row["slowest"] = held["fastest"], held["slowest"]
            row["status"] = j["status"] if "status" in j else _status(held["fastest"], held["slowest"])
            rows.append(row)
    return pd.DataFrame(rows)
