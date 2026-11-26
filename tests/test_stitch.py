"""Step 3 stitching: each scenario has a known right answer, including cases that must NOT join."""
import pandas as pd
import pytest

from lane_intel.tracking.config import parse_variants
from lane_intel.tracking.runner import run_tracker
from lane_intel.tracking.stitch import StitchParams, stitch

FPS = 30000 / 1001
W = 3840
V = 60.0          # px/s, a slow-moving jam
WINDOW = (0, 299)


def piece(frames, x0=600.0, v=V, top=300.0, w=80.0, h=40.0):
    """One track piece of a vehicle moving at v px/s: [(frame, x_centre, top, w, h), ...]."""
    return [(f, x0 + v * f / FPS, top, w, h) for f in frames]


def points(tracks):
    rows = []
    for tid, obs in tracks.items():
        for f, x, top, w, h in obs:
            rows.append(dict(t=f / FPS, frame_idx=f, track_id=tid, cls="car", cls_id=2, conf=0.8,
                             x1=x - w / 2, y1=top, x2=x + w / 2, y2=top + h, x_px=x, y_px=top + h))
    return pd.DataFrame(rows)


def ids_after(tracks, **params):
    out, links = stitch(points(tracks), FPS, WINDOW, W, StitchParams(**params))
    return out.groupby("raw_track_id").track_id.first().to_dict(), links


def test_hidden_car_is_joined_and_raw_ids_kept():
    ids, links = ids_after({1: piece(range(0, 100)), 2: piece(range(160, 300))})
    assert ids == {1: 1, 2: 1}
    assert len(links) == 1 and links.gap_s.iloc[0] == pytest.approx((160 - 99) / FPS, abs=1e-3)


@pytest.mark.parametrize("second,why", [
    (piece(range(160, 300), top=345.0), "next lane (45 px lower)"),
    (piece(range(160, 300), x0=900.0), "starts 300 px from where the car should be"),
    (piece(range(160, 300), v=-V, x0=600 + 2 * V * 160 / FPS), "drives the other way"),
    (piece(range(160, 300), w=160.0), "twice as wide: another vehicle"),
    (piece(range(260, 300)), "gap 5.4 s is longer than allowed"),
])
def test_must_not_join(second, why):
    ids, links = ids_after({1: piece(range(0, 100)), 2: second}, max_gap_s=5.0)
    assert ids == {1: 1, 2: 2}, why
    assert links.empty, why


def test_track_leaving_at_the_image_edge_is_not_joined():
    # piece 1 ends 60 px from the right edge: it left the picture, so it is not a break
    x_end = W - 60
    first = piece(range(0, 100), x0=x_end - V * 99 / FPS)
    second = piece(range(130, 200), x0=x_end - V * 99 / FPS)
    ids, links = ids_after({1: first, 2: second})
    assert links.empty and ids == {1: 1, 2: 2}


def test_chains_become_one_vehicle():
    ids, links = ids_after({1: piece(range(0, 80)), 2: piece(range(110, 180)), 3: piece(range(220, 300))})
    assert ids == {1: 1, 2: 1, 3: 1} and len(links) == 2


def test_one_to_one_picks_the_better_candidate():
    ids, links = ids_after({
        1: piece(range(0, 100)),
        2: piece(range(130, 300)),                 # exactly where predicted
        3: piece(range(130, 300), x0=640.0),       # 40 px off, still within tolerance
    })
    assert ids == {1: 1, 2: 1, 3: 3} and len(links) == 1


def test_stitch_settings_are_validated():
    base = {"tracker": "bytetrack", "min_conf": 0.25, "lost_buffer_s": 1.0}
    assert parse_variants({"variants": {"v": {**base, "stitch": {"max_gap_s": 3}}}})["v"].stitch == {"max_gap_s": 3}
    with pytest.raises(ValueError, match="unknown stitch settings"):
        parse_variants({"variants": {"v": {**base, "stitch": {"max_gapp_s": 3}}}})


def test_runner_stitches_what_the_tracker_could_not_bridge():
    rows = []
    for f in range(300):
        t = f / FPS
        if 3.0 <= t < 6.0:                        # hidden for 3 s, longer than the 1 s memory
            continue
        x = 600 + V * t
        rows.append((t, f, "car", 2, 0.9, x - 40, 300.0, x + 40, 340.0))
    dets = pd.DataFrame(rows, columns=["t", "frame_idx", "cls", "cls_id", "conf", "x1", "y1", "x2", "y2"])
    base = {"tracker": "bytetrack", "min_conf": 0.25, "lost_buffer_s": 1.0,
            "params": {"track_activation_threshold": 0.5, "high_conf_det_threshold": 0.5}}
    v = parse_variants({"variants": {"plain": base, "stitched": {"extends": "plain", "stitch": {"max_gap_s": 5}}}})
    plain, _, _ = run_tracker(dets, v["plain"], FPS, (0, 299), W)
    joined, stats, links = run_tracker(dets, v["stitched"], FPS, (0, 299), W)
    assert plain.track_id.nunique() == 2
    assert joined.track_id.nunique() == 1 and joined.raw_track_id.nunique() == 2
    assert stats["stitch_links"] == 1 and len(links) == 1
