"""Lanes from vehicle paths: synthetic roads where the lanes are known."""
import numpy as np
import pandas as pd

from lane_intel.lanes import Geometry, assign_lanes, estimate_lanes, lane_polygons

FPS = 25.0
W = 3840


def road(lanes, n_per_lane=8, seconds=6.0, curve=0.0, slope=0.0, still_lane=None, seed=0):
    """lanes: [(c, direction)], c = height at x_ref; lane line y = c + slope*u + curve*u^2, u = x - W/2."""
    rng = np.random.default_rng(seed)
    rows, tid = [], 0
    for li, (c, d) in enumerate(lanes):
        for k in range(n_per_lane):
            tid += 1
            v = 0.0 if li == still_lane else d * rng.uniform(150, 300)
            x0 = rng.uniform(300, W - 300) if v == 0 else (200 if d > 0 else W - 200) + d * k * 300
            for f in range(int(seconds * FPS)):
                x = x0 + v * f / FPS + (rng.normal(0, 2) if v == 0 else 0)
                if not 0 < x < W:
                    continue
                u = x - W / 2
                y = c + slope * u + curve * u * u + rng.normal(0, 1.5)
                rows.append(dict(t=f / FPS, frame_idx=f, track_id=tid, x_px=x, y_px=y,
                                 x1=x - 40, x2=x + 40, y1=y - 35, y2=y, lane_true=li))
    return pd.DataFrame(rows)


def test_two_way_straight_road_lanes_and_numbering():
    # left-moving carriageway on top (c 1200..1280), right-moving below (1400..1480), 40 px apart
    pts = road([(1200, -1), (1240, -1), (1280, -1), (1400, 1), (1440, 1), (1480, 1)])
    geo, lanes, _ = estimate_lanes(pts, W)
    by_id = {l["id"]: l for l in lanes}
    assert sorted(by_id) == ["L1", "L2", "L3", "R1", "R2", "R3"]
    # lane 1 = leftmost in the direction of travel: lowest for left-moving, highest for right-moving
    assert abs(by_id["L1"]["c"] - 1280) < 6 and abs(by_id["L3"]["c"] - 1200) < 6
    assert abs(by_id["R1"]["c"] - 1400) < 6 and abs(by_id["R3"]["c"] - 1480) < 6
    out = assign_lanes(pts, geo, lanes, W)
    truth = {0: "L3", 1: "L2", 2: "L1", 3: "R1", 4: "R2", 5: "R3"}
    got = out.lane_frame.map(lambda i: lanes[i]["id"] if i >= 0 else None)
    assert (got == out.lane_true.map(truth)).mean() > 0.98


def test_curved_road_is_followed():
    pts = road([(1300, -1), (1340, -1), (1380, -1)], curve=6e-5, slope=0.05)   # ~150 px bend over the width
    geo, lanes, scores = estimate_lanes(pts, W)
    assert geo.kind == "shape" and len(geo.coef) >= 2                          # a curved family was chosen
    assert len(lanes) == 3
    out = assign_lanes(pts, geo, lanes, W)
    assert (out.lane_frame >= 0).mean() > 0.97


def test_jammed_lane_without_moving_cars_is_recovered():
    # middle lane of the left carriageway stands still: only the second pass can find it
    pts = road([(1200, -1), (1240, -1), (1280, -1), (1400, 1), (1440, 1)], still_lane=1, n_per_lane=10)
    _, lanes, _ = estimate_lanes(pts, W)
    left = sorted(l["c"] for l in lanes if l["direction"] == "left")
    assert len(left) == 3 and abs(left[1] - 1240) < 6
    assert len([l for l in lanes if l["direction"] == "right"]) == 2


def test_geometry_round_trip_and_polygons():
    g = Geometry(1920.0, "shape", (0.05, 1e-5))
    c = g.c_of(1000.0, 1300.0)
    assert abs(float(g.y_of(c, 1000.0)) - 1300.0) < 1e-9
    v = Geometry(1920.0, "vp", (0.0,), (5000.0, 400.0))
    assert abs(float(v.y_of(v.c_of(1000.0, 1300.0), 1000.0)) - 1300.0) < 1e-9
    lanes = [{"id": "L1", "number": 1, "direction": "left", "c": 1300.0, "x_range": [0, 3840]},
             {"id": "L2", "number": 2, "direction": "left", "c": 1260.0, "x_range": [0, 3840]}]
    polys = lane_polygons(g, lanes)
    assert polys[0]["upper"][0][1] < polys[0]["centre"][0][1] < polys[0]["lower"][0][1]
