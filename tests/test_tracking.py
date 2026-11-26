"""Tracking package: variant config, runner behaviour on known scenarios, quality metrics, selection rule.

Scenarios use synthetic detections whose correct answer is known: one car driving along a
lane at constant speed, with a hidden stretch or a stretch of faint detections.
"""
import numpy as np
import pandas as pd
import pytest

from lane_intel.tracking.config import TrackerConfig, parse_variants
from lane_intel.tracking.experiment import judge
from lane_intel.tracking.quality import QualityParams, evaluate, identity_jumps
from lane_intel.tracking.runner import run_tracker

FPS = 30000 / 1001
BASE = {"tracker": "bytetrack", "min_conf": 0.25, "lost_buffer_s": 1.0,
        "params": {"track_activation_threshold": 0.5, "high_conf_det_threshold": 0.5,
                   "minimum_iou_threshold": 0.1, "minimum_consecutive_frames": 2}}


def variant(**over) -> TrackerConfig:
    spec = {**BASE, **{k: v for k, v in over.items() if k != "params"}}
    spec["params"] = {**BASE["params"], **over.get("params", {})}
    return parse_variants({"variants": {"v": spec}})["v"]


def car(seconds=10.0, speed=60.0, y=200.0, x0=300.0, conf=0.9, hidden=(), faint=(), faint_conf=0.15):
    """Detections of one 80x40 px car moving +speed px/s. hidden/faint: (t_from, t_to) in seconds."""
    rows = []
    for f in range(int(seconds * FPS)):
        t = f / FPS
        if hidden and hidden[0] <= t < hidden[1]:
            continue
        c = faint_conf if faint and faint[0] <= t < faint[1] else conf
        x = x0 + speed * t
        rows.append((t, f, "car", 2, c, x, y - 40, x + 80, y))
    return pd.DataFrame(rows, columns=["t", "frame_idx", "cls", "cls_id", "conf", "x1", "y1", "x2", "y2"])


# ------------------------------------------------------------------ config
def test_extends_merges_params_and_overrides_fields():
    v = parse_variants({"variants": {
        "base": BASE,
        "child": {"extends": "base", "min_conf": 0.1, "params": {"minimum_iou_threshold": 0.03}},
        "grandchild": {"extends": "child", "lost_buffer_s": 4.0},
    }})
    g = v["grandchild"]
    assert (g.min_conf, g.lost_buffer_s, g.tracker) == (0.1, 4.0, "bytetrack")
    assert g.params["minimum_iou_threshold"] == 0.03
    assert g.params["track_activation_threshold"] == 0.5          # inherited from base


def test_changing_tracker_type_replaces_params():
    v = parse_variants({"variants": {
        "base": BASE,
        "oc": {"extends": "base", "tracker": "ocsort", "params": {"minimum_iou_threshold": 0.1}},
        "oc_bare": {"extends": "base", "tracker": "ocsort"},
    }})
    assert v["oc"].params == {"minimum_iou_threshold": 0.1}
    assert v["oc_bare"].params == {}


@pytest.mark.parametrize("spec,err", [
    ({"v": {**BASE, "params": {"no_such_arg": 1}}}, "does not accept"),
    ({"v": {**BASE, "params": {"frame_rate": 10}}}, "lost_buffer_s"),
    ({"v": {**BASE, "tracker": "magic"}}, "tracker must be one of"),
    ({"a": {"extends": "b"}, "b": {"extends": "a"}}, "circular"),
    ({"v": {"extends": "missing"}}, "unknown tracker variant"),
])
def test_invalid_variants_are_rejected(spec, err):
    with pytest.raises((ValueError, KeyError), match=err):
        parse_variants({"variants": spec})


def test_repo_variants_file_resolves():
    from lane_intel.tracking.config import load_variants
    v = load_variants("configs/tracking.json")
    assert "baseline" in v and v["baseline"].min_conf == 0.25 and v["baseline"].lost_buffer_s == 1.0
    assert all(c.tracker in {"bytetrack", "ocsort", "botsort", "sort"} for c in v.values())


# ------------------------------------------------------------------ runner
def ids_of(dets, cfg):
    pts, stats, _ = run_tracker(dets, cfg, FPS, (0, int(dets.frame_idx.max())))
    return pts.track_id.nunique(), pts, stats


def test_memory_shorter_than_occlusion_breaks_the_track():
    dets = car(hidden=(3.0, 5.0))                                   # hidden for 2 s
    assert ids_of(dets, variant(lost_buffer_s=1.0))[0] == 2
    assert ids_of(dets, variant(lost_buffer_s=4.0))[0] == 1


def test_faint_boxes_keep_the_track_only_when_fed_to_the_tracker():
    dets = car(faint=(3.0, 5.0), faint_conf=0.15)                   # visible but faint for 2 s
    assert ids_of(dets, variant(min_conf=0.25))[0] == 2              # faint boxes filtered out
    n, pts, _ = ids_of(dets, variant(min_conf=0.10))
    assert n == 1
    assert (pts.conf < 0.25).sum() > 50                              # the faint boxes carry the ID


def test_faint_boxes_never_start_a_track():
    dets = car(conf=0.15)                                           # a car seen only faintly
    assert ids_of(dets, variant(min_conf=0.10))[0] == 0


def test_runner_output_schema_and_coverage():
    _, pts, stats = ids_of(car(), variant())
    assert list(pts.columns[:6]) == ["t", "frame_idx", "track_id", "cls", "cls_id", "conf"]
    assert np.allclose(pts.y_px, pts.y2) and np.allclose(pts.x_px, (pts.x1 + pts.x2) / 2)
    assert stats["coverage_clear"] > 0.99                           # only the first frame lacks an ID


# ------------------------------------------------------------------ quality
def points(tracks):
    """tracks: {id: [(frame, x_anchor, y_anchor), ...]} -> points with 80x40 boxes."""
    rows = []
    for tid, obs in tracks.items():
        for f, x, y in obs:
            rows.append(dict(t=f / FPS, frame_idx=f, track_id=tid, cls="car", cls_id=2, conf=0.9,
                             x1=x - 40, y1=y - 40, x2=x + 40, y2=y, x_px=x, y_px=y))
    return pd.DataFrame(rows)


def test_breaks_restarts_and_edges_are_classified():
    v = 60.0  # px/s
    p = points({
        1: [(f, 120 + 400 + v * f / FPS, 300) for f in range(0, 100)],           # starts at window start
        2: [(f, 120 + 400 + v * f / FPS, 300) for f in range(160, 300)],         # same car after a 2 s gap
        3: [(f, v * f / FPS, 500) for f in range(40, 300)],                      # enters at the left edge (x=80)
        4: [(f, 2000, 700) for f in range(100, 110)],                            # 0.3 s flicker
    })
    s, tb = evaluate(p, FPS, (0, 299), 3840)
    tr = tb["tracks"].set_index("track_id")
    assert not tr.loc[1, "born_mid"] and tr.loc[1, "died_mid"]
    assert tr.loc[2, "born_mid"] and not tr.loc[2, "died_mid"]                   # runs to window end
    assert not tr.loc[3, "born_mid"]                                            # entered at the edge
    assert s["short_lt_1s"] == 1 and s["ids"] == 4
    pairs = tb["restarts"]
    assert len(pairs) == 1 and (pairs.track_end.iloc[0], pairs.track_start.iloc[0]) == (1, 2)
    assert pairs.gap_s.iloc[0] == pytest.approx((160 - 99) / FPS, abs=0.001)   # last seen 99, back at 160


def test_identity_jump_needs_both_edges_to_move():
    smooth = [(f, 1000 + 2 * f, 300) for f in range(30)]
    jumped = smooth[:15] + [(f, 1000 + 2 * f, 345) for f in range(15, 30)]       # lane-width jump
    p = points({1: smooth, 2: jumped})
    ev = identity_jumps(p, FPS, QualityParams())
    assert list(ev.track_id) == [2] and ev.axis.iloc[0] == "y" and ev.frame_idx.iloc[0] == 15

    shrink = points({3: smooth}).copy()                                          # partly hidden car:
    shrink.loc[shrink.frame_idx >= 15, "y2"] -= 25                               # only the bottom edge moves
    assert identity_jumps(shrink, FPS, QualityParams()).empty


# ------------------------------------------------------------------ selection rule
def test_selection_rule_applies_guards_then_fragmentation():
    base = {"ids": 300, "fragmentation": 400, "short_lt_1s": 90, "identity_jumps": 20, "coverage_clear": 0.95}
    res = {
        "baseline": base,
        "good": {**base, "ids": 200, "fragmentation": 250},
        "better_but_merges_wrong_cars": {**base, "fragmentation": 150, "identity_jumps": 40},
        "better_but_drops_boxes": {**base, "fragmentation": 140, "coverage_clear": 0.90},
    }
    selected, failures = judge(res)
    assert selected == "good"
    assert failures["better_but_merges_wrong_cars"] and failures["better_but_drops_boxes"]
    assert failures["baseline"] == [] and failures["good"] == []


def test_crosscheck_takes_the_heldout_selection():
    from lane_intel.tracking.crosscheck import crosscheck

    def report(sel, frag):
        return {"selected": sel, "variants": {
            n: {"config": {"step": "0" if n == "baseline" else "2"}, "guard_failures": [],
                "summary": {"fragmentation": f, "identity_jumps": 1}} for n, f in frag.items()}}
    table, final = crosscheck(report("a", {"baseline": 100, "a": 50, "b": 60}),
                              report("b", {"baseline": 100, "a": 90, "b": 70}))
    assert final == "b"
    assert "`b` **(final)**" in table and "50 (-50%)" in table and "70 (-30%)" in table


def test_jumps_at_stitch_junctions_are_counted_separately():
    smooth = [(f, 1000 + 2 * f, 300) for f in range(30)]
    jumped = smooth[:15] + [(f, 1000 + 2 * f, 345) for f in range(15, 30)]
    p = points({2: jumped})
    p["raw_track_id"] = np.where(p.frame_idx < 15, 20, 21)                    # two pieces stitched at f15
    s, tb = evaluate(p, FPS, (0, 29), 3840)
    assert s["identity_jumps"] == 1 and s["junction_jumps"] == 1 and s["identity_jumps_within"] == 0
    assert bool(tb["jumps"].junction.iloc[0])


def test_guard_uses_jumps_within_pieces():
    base = {"ids": 300, "fragmentation": 400, "short_lt_1s": 90, "identity_jumps": 10,
            "identity_jumps_within": 10, "junction_jumps": 0, "coverage_clear": 0.95}
    stitched = {**base, "fragmentation": 250, "identity_jumps": 50, "junction_jumps": 40}
    selected, failures = judge({"ref": base, "stitched": stitched}, "ref")
    assert selected == "stitched" and failures["stitched"] == []
