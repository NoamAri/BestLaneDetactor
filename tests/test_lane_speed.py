"""Lane speeds: synthetic traffic where the true speeds, lane changes and identity jumps are known."""
import cv2
import numpy as np
import pandas as pd
import pytest

from lane_intel import announce, lane_pieces, lane_speed, world
from lane_intel.calibrate import RoadFrame

FPS = 30.0
EV = {"min_vehicles": 5, "min_vehicle_time_s": 20.0}
ANN = {"min_gap": 0.10, "min_p_rank": 0.90, "ci": 0.95, "bootstrap": 1000, "seed": 0,
       "keep_gap": 0.05, "takeover_s": 60.0, "free_flow_frac": 0.85}
PLAUS = {"max_speed_kmh": 200.0, "max_accel_ms2": 5.0, "accel_window_s": 0.5,
         "max_lateral_jump_lane_frac": 0.5, "lateral_jump_window_s": 0.5}
LANES = [{"id": "R1", "number": 1, "direction": "right"}, {"id": "R2", "number": 2, "direction": "right"},
         {"id": "R3", "number": 3, "direction": "right"}]
LANE_D = {"R1": 0.0, "R2": 3.6, "R3": 7.2}
DIR = {l["id"]: l["direction"] for l in LANES}


def traj(seg_id, lane, v, s0=0.0, t0=0.0, t1=20.0, d=None, cls="car", noise=0.0, rng=None):
    t = np.arange(t0, t1, 1 / FPS)
    s = s0 + v * (t - t0) + (rng.normal(0, noise, len(t)) if noise else 0)
    return pd.DataFrame({"seg_id": seg_id, "track_id": seg_id, "t": t, "s": s,
                         "d": LANE_D[lane] if d is None else d, "lane": lane, "cls": cls})


def segments_of(road):
    segs, ch = lane_pieces.build(road, LANE_D, DIR, 0.5)
    return segs.assign(sign=1, gap=0.0), ch


# --------------------------------------------------------------------------- Edie sums
def test_clip_fraction_is_exact():
    f = lane_speed.clip_fraction(np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]),
                                 np.array([0.0, 30.0, 0.0]), np.array([100.0, 30.0, 100.0]),
                                 20.0, 50.0, 0.0, 10.0)
    assert f == pytest.approx([0.3, 1.0, 0.3])
    g = lane_speed.clip_fraction(np.array([0.0]), np.array([10.0]), np.array([0.0]), np.array([100.0]), 0, 100, 2.0, 4.0)
    assert g == pytest.approx([0.2])


def test_space_mean_speed_and_id_split_invariance():
    rng = np.random.default_rng(0)
    cars = [traj(i, "R1", rng.uniform(15, 25), s0=-50 + 7 * i, rng=rng) for i in range(12)]
    road = pd.concat(cars, ignore_index=True)
    segs, _ = segments_of(road)
    veh = lane_speed.edie_vehicles(segs, 0.0, 200.0, 2.0, 18.0)
    v_true = veh.D.sum() / veh.tau.sum()
    # split every vehicle's ID at a different moment: distance and time sums must not change
    split = road.copy()
    cut = split.t > 5 + split.seg_id * 0.7
    split.loc[cut, "seg_id"] += 1000
    segs2, _ = segments_of(split)
    veh2 = lane_speed.edie_vehicles(segs2, 0.0, 200.0, 2.0, 18.0)
    lost = 12 * (1 / FPS) * 25                                         # at most one step per split, ~25 m/s
    assert abs(veh2.D.sum() - veh.D.sum()) < lost
    assert veh2.D.sum() / veh2.tau.sum() == pytest.approx(v_true, rel=0.01)
    assert len(veh2) > len(veh)                                        # more "vehicles", same speed


def test_standing_vehicle_jitter_adds_time_not_distance():
    rng = np.random.default_rng(1)
    road = traj(1, "R2", 0.0, s0=50.0, noise=0.3, rng=rng)
    segs, _ = segments_of(road)
    veh = lane_speed.edie_vehicles(segs, 0.0, 100.0, 0.0, 19.0)
    assert veh.tau.sum() == pytest.approx(19.0, abs=0.05)
    assert abs(veh.D.sum()) < 1.0                                      # signed steps cancel (|steps| would be ~200 m)


def test_lane_change_is_split_where_the_vehicle_crosses():
    t = np.arange(0, 10, 1 / FPS)
    d = np.clip((t - 4) / 2, 0, 1) * 3.6                               # R1 -> R2 between 4 s and 6 s, midline at 5 s
    lane = np.where(d < 1.8, "R1", "R2").astype(object)
    lane[(t > 4.9) & (t < 5.0)] = "R2"                                 # flicker at the line
    lane[(t > 5.05) & (t < 5.15)] = "R1"
    road = pd.DataFrame({"seg_id": 1, "track_id": 1, "t": t, "s": 20 * t, "d": d, "lane": lane, "cls": "car"})
    segs, ch = segments_of(road)
    assert len(ch) == 1 and (ch.from_lane.iat[0], ch.to_lane.iat[0]) == ("R1", "R2")
    veh = lane_speed.edie_vehicles(segs, -1, 1e3, 0, 10).set_index("lane")
    assert veh.loc["R1", "D"] == pytest.approx(100, abs=2.5)           # 5 s at 20 m/s in each lane
    assert veh.loc["R2", "D"] == pytest.approx(100 - 20 / FPS, abs=2.5)


# --------------------------------------------------------------------------- plausibility
def test_identity_jump_is_cut_once_and_smooth_driving_is_not():
    rng = np.random.default_rng(2)
    t = np.arange(0, 8, 1 / FPS)
    s = 10 * t + rng.normal(0, 0.3, len(t))
    assert len(world.plausibility_cuts(t, s, np.zeros_like(t), PLAUS, 3.6)) == 0
    brake = np.where(t < 3, 15 * t, 45 + 15 * (t - 3) - 1.5 * (t - 3) ** 2).clip(max=None)   # 3 m/s^2
    brake = brake + rng.normal(0, 0.3, len(t))
    assert len(world.plausibility_cuts(t, brake, np.zeros_like(t), PLAUS, 3.6)) == 0
    jump = s + np.where(t >= 4, 12.0, 0.0)                             # box lands on the car 12 m ahead
    cuts = world.plausibility_cuts(t, jump, np.zeros_like(t), PLAUS, 3.6)
    assert len(cuts) == 1 and abs(t[cuts[0]] - 4) < 0.1
    side = np.where(t >= 4, 3.6, 0.0) + rng.normal(0, 0.1, len(t))    # box lands on the neighbour lane's car
    cuts = world.plausibility_cuts(t, s, side, PLAUS, 3.6)
    assert len(cuts) == 1 and abs(t[cuts[0]] - 4) < 0.1


# --------------------------------------------------------------------------- announcement
def veh_table(speeds_kmh, n=30, sd=0.05, seed=0, window="w"):
    rng = np.random.default_rng(seed)
    rows = []
    for k, v in enumerate(speeds_kmh):
        for i in range(n):
            tau = rng.uniform(3, 8)
            rows.append({"lane": f"R{k + 1}", "seg_id": k * 1000 + i, "cls": "car", "window": window, "direction": "right",
                         "D": v / 3.6 * rng.normal(1, sd) * tau, "tau": tau, "tau_gap": 0.0})
    return pd.DataFrame(rows)


def lane_rows(veh, window="w", t0=0.0):
    tab = lane_speed.lane_table(veh, LANES, 100.0, 60.0, EV)
    return tab.assign(window=window, t0=t0, t1=t0 + 60, stretch_ok=True)


def test_clear_fastest_lane_is_announced_and_equal_lanes_are_not():
    veh = veh_table([45, 30, 30])
    j = announce.judge(lane_rows(veh), veh, ANN, None, 0.0, np.random.default_rng(0))
    assert j["fastest"]["meets_rule"] and j["fastest"]["lane"] == "R1" and j["fastest"]["p"] >= 0.9
    assert not j["slowest"]["meets_rule"]                              # R2 and R3 are equal
    veh = veh_table([30, 30, 30], seed=3)
    j = announce.judge(lane_rows(veh), veh, ANN, None, 0.0, np.random.default_rng(0))
    assert not j["fastest"]["meets_rule"] and not j["slowest"]["meets_rule"]


def test_small_gap_or_few_vehicles_is_not_announced():
    veh = veh_table([33, 31, 30])                                      # 6% gap: below 10%
    j = announce.judge(lane_rows(veh), veh, ANN, None, 0.0, np.random.default_rng(0))
    assert not j["fastest"]["meets_rule"]
    few = veh_table([45, 30, 30], n=3)                                 # fails minimum evidence
    j = announce.judge(lane_rows(few), few, ANN, None, 0.0, np.random.default_rng(0))
    assert j["status"] == "not enough data"


def test_free_flow_and_calibration_sensitivity():
    veh = veh_table([110, 95, 95])
    j = announce.judge(lane_rows(veh), veh, ANN, 110.0, 0.0, np.random.default_rng(0))
    assert j["status"].startswith("free flow")
    veh = veh_table([34, 30, 30], sd=0.01, n=60)                       # 13% gap, very precise
    j = announce.judge(lane_rows(veh), veh, ANN, None, 0.08, np.random.default_rng(0))
    assert j["fastest"]["meets_rule"] and j["fastest"]["calibration_sensitive"]


def test_hysteresis_keeps_the_title_through_a_short_wobble_and_needs_a_fresh_minute_to_hand_over():
    seq = [[45, 30, 30]] * 4 + [[42, 41, 30]] + [[45, 30, 30]] * 2 + [[30, 45, 30]] * 16
    tabs, vehs = [], []
    for k, sp in enumerate(seq):
        w = f"w{k:03d}"
        v = veh_table(sp, seed=k, window=w)
        tabs.append(lane_rows(v, window=w, t0=5.0 * k))
        vehs.append(v)
    ann = announce.announce_all(pd.concat(tabs), pd.concat(vehs), ANN)
    holder = ann.set_index("t0").fastest
    assert holder[0.0] == "R1"
    assert pd.isna(holder[20.0])                                       # gap 2% < keep_gap: title lost
    assert holder[25.0] == "R1"                                        # same lane regains immediately
    first_r2 = 5.0 * 7
    assert pd.isna(holder[first_r2])                                   # R2 leads but has not led for 60 s
    assert holder[first_r2 + 60] == "R2"


# --------------------------------------------------------------------------- end to end, perspective
def test_far_lane_is_slower_in_pixels_but_equal_in_metres():
    """Strong perspective: equal true speeds must give equal lane speeds; a faster lane must win."""
    A = np.array([[0.05, 0.0], [0.0, 0.09]])
    H = np.eye(3)
    H[:2, :2] = A
    H[2, :2] = [0.0, 1e-3]                                             # strong perspective along image y
    Hinv = np.linalg.inv(H)
    rf = RoadFrame("gt_homography", H=H, origin=np.zeros(2), u_along=np.array([1.0, 0.0]),
                   u_across=np.array([0.0, 1.0]))
    true = {"R1": 25.0, "R2": 25.0, "R3": 32.0}                        # m/s
    rows, tid = [], 0
    rng = np.random.default_rng(4)
    for k, (lane, v) in enumerate(true.items()):
        for i in range(10):
            tid += 1
            t = np.arange(0, 20, 1 / FPS)
            sx = -120 + 25 * i + v * t
            m = (sx > 0) & (sx < 180)
            w = np.c_[sx[m], np.full(m.sum(), 60.0 + 3.6 * k)]
            q = cv2.perspectiveTransform(w.reshape(-1, 1, 2), Hinv).reshape(-1, 2) + rng.normal(0, 0.5, (m.sum(), 2))
            rows.append(pd.DataFrame({"t": t[m], "frame_idx": np.round(t[m] * FPS).astype(int), "track_id": tid,
                                      "cls": "car", "x_px": q[:, 0], "y_px": q[:, 1], "x1": q[:, 0] - 40,
                                      "x2": q[:, 0] + 40, "y1": q[:, 1] - 15, "y2": q[:, 1] + 15, "lane_frame": k}))
    g = pd.concat(rows, ignore_index=True)
    ls = {"occlusion_height_frac": 0.7, "plausibility": PLAUS}
    road = world.road_points(g, rf, LANES, {"R1": 60.0, "R2": 63.6, "R3": 67.2}, ls)
    assert road.seg_id.nunique() == road.track_id.nunique()            # no false cuts
    segs, ch = lane_pieces.build(road, {"R1": 60.0, "R2": 63.6, "R3": 67.2}, DIR, 0.5)
    segs = segs.assign(sign=1, gap=0.0)
    st = lane_speed.choose_stretch(road, ["R1", "R2", "R3"], {"margin_frac": 0.05, "coverage_bins": 12,
                                                              "min_bin_coverage_frac": 0.5, "min_length_m": 40})
    assert st["ok"]
    veh = lane_speed.edie_vehicles(segs, st["s0"], st["s1"], 0, 20)
    tab = lane_speed.lane_table(veh, LANES, st["length_m"], 20, {"min_vehicles": 5, "min_vehicle_time_s": 5}).set_index("lane")
    for lane, v in true.items():
        assert tab.loc[lane, "v_kmh"] == pytest.approx(v * 3.6, rel=0.01)
    per_track = g.sort_values("t").groupby("track_id").agg(lane=("lane_frame", "first"), x0=("x_px", "first"),
                                                            x1=("x_px", "last"), t0=("t", "first"), t1=("t", "last"))
    px_speed = ((per_track.x1 - per_track.x0) / (per_track.t1 - per_track.t0)).groupby(per_track.lane).median()
    assert px_speed[1] > px_speed[0] * 1.1                             # R2 looks >10% faster in pixels than R1
