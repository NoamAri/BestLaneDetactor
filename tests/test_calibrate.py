"""Stabilisation and calibration: synthetic cameras and roads where the true answer is known."""
import subprocess

import cv2
import numpy as np
import pandas as pd
import pytest

from lane_intel import calibrate as C
from lane_intel import stabilize, world
from lane_intel.lanes import estimate_lanes

W = 3840
FPS = 30.0
H_ID = {f"h{r}{c}": float(r == c) for r in range(3) for c in range(3)}


# --------------------------------------------------------------------------- stabilisation
def test_apply_moves_points_by_the_frame_homography():
    shift = {**H_ID, "h02": 5.0, "h12": -3.0}
    stab = pd.DataFrame([{"frame_idx": 0, **H_ID}, {"frame_idx": 1, **shift}])
    x, y = stabilize.apply(stab, [0, 1, 1], [10.0, 10.0, 100.0], [20.0, 20.0, 50.0])
    assert np.allclose(x, [10, 15, 105]) and np.allclose(y, [20, 17, 47])
    with pytest.raises(ValueError):
        stabilize.apply(stab, [7], [0.0], [0.0])


def test_estimate_recovers_a_drifting_camera(tmp_path):
    """The camera drifts: frame n shows the scene shifted by (-0.3 n, -0.15 n) px (sub-pixel steps)."""
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (420, 720)).astype(np.float32), (0, 0), 2.0)
    scene = cv2.normalize(scene, None, 0, 255, cv2.NORM_MINMAX)
    video = tmp_path / "drift.mp4"
    n_frames = 30
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "640x360",
                             "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(video)],
                            stdin=subprocess.PIPE)
    for n in range(n_frames):
        M = np.float32([[1, 0, -(20 + 0.3 * n)], [0, 1, -(20 + 0.15 * n)]])     # crop window moves right/down
        frame = cv2.warpAffine(scene, M, (640, 360), flags=cv2.INTER_CUBIC)
        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    cfg = {"video": str(video), "frame_size": [640, 360], "native_fps": 30.0}
    no_boxes = pd.DataFrame(columns=["x1", "y1", "x2", "y2"])
    stab = stabilize.estimate(cfg, 0.0, None, no_boxes)
    assert len(stab) == n_frames
    f = stab.frame_idx.to_numpy()
    x, y = stabilize.apply(stab, f, np.full(len(f), 320.0), np.full(len(f), 180.0))
    assert np.abs(x - 320 - 0.3 * f).max() < 0.15
    assert np.abs(y - 180 - 0.15 * f).max() < 0.15


# --------------------------------------------------------------------------- ground-truth homography
def _oblique_road(offset_s=0.05, seed=0):
    """World: two carriageways, heading -0.45 rad. Camera: anisotropic + mild perspective (known H).

    Returns our tracked points (image), the ground truth (world metres, its own clock) and H_true.
    Video time t corresponds to ground-truth time t + offset_s.
    """
    rng = np.random.default_rng(seed)
    a = -0.45
    u = np.array([np.cos(a), np.sin(a)])                   # along the road
    n = np.array([-u[1], u[0]])                            # across the road
    A = np.stack([u * 0.0625, n * 0.100], 1)               # pixel offsets -> metres (0.0625 along x, 0.10 down y)
    H_true = np.eye(3)
    H_true[:2, :2], H_true[:2, 2] = A, np.array([500.0, 300.0]) - A @ np.array([W / 2, 1350])
    H_true[2, :2] = [2e-6, -1e-6]                          # a little perspective
    Hinv = np.linalg.inv(H_true)
    centre = np.array([500.0, 300.0])
    # lanes: right-moving (+u) lateral offsets 4..13 m, fast; left-moving -4..-13 m, slow
    gt_rows, pts_rows, vid = [], [], 0
    for dirn, lats, v_ms in ((1, (5.0, 8.6, 12.2), 10.0), (-1, (-5.0, -8.6, -12.2), 1.5)):
        for lat in lats:
            for k in range(6):
                vid += 1
                v = v_ms * rng.uniform(0.85, 1.15)
                s0 = -110 + k * 40 if dirn > 0 else 110 - k * 40
                for f in range(int(12 * FPS)):
                    tg = f / FPS                                       # ground-truth clock
                    s = s0 + dirn * v * tg
                    if abs(s) > 115:
                        continue
                    p = centre + s * u + lat * n
                    gt_rows.append(dict(vid=vid, type="car", t=tg, x=p[0], y=p[1], v_kmh=v * 3.6,
                                        lane=0, angle=a if dirn > 0 else a + np.pi, width=1.8))
                    tv = tg - offset_s                                  # video clock
                    q = cv2.perspectiveTransform(p.reshape(1, 1, 2), Hinv).reshape(2) + rng.normal(0, 0.8, 2)
                    pts_rows.append(dict(t=tv, frame_idx=int(round(tv * FPS)), track_id=vid, cls="car",
                                         x_px=q[0], y_px=q[1], x1=q[0] - 36, x2=q[0] + 36, y1=q[1] - 17, y2=q[1] + 17))
    pts = pd.DataFrame(pts_rows)
    pts = pts[pts.t >= 0].reset_index(drop=True)
    return pts, pd.DataFrame(gt_rows), H_true


def test_gt_homography_recovers_map_time_offset_and_speed_scale():
    pts, gt, H_true = _oblique_road(offset_s=0.05)
    rf, info, corr = C.fit_gt_homography(pts, gt, FPS, clip_offset_s=0.0, width=W)
    assert info["reprojection_m"]["median"] < 0.12              # noise is 0.8 px (~0.05-0.08 m)
    assert abs(info["time_offset_s"] - 0.05) <= 0.02
    assert info["paired_tracks"] >= 30
    chk = C.speed_check(rf, corr)
    assert abs(chk["ratio_median"] - 1) < 0.01
    # the fitted map agrees with the true one across the road area
    grid = np.array([[x, y] for x in (300, 1900, 3500) for y in (1250, 1350, 1450)], float)
    w_fit = rf.world(grid[:, 0], grid[:, 1]) + rf.origin
    w_true = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H_true).reshape(-1, 2)
    assert np.abs(w_fit - w_true).max() < 0.3


# --------------------------------------------------------------------------- vehicle-size scale field
def _lane_road(k_of_x, lanes_c=(1300, 1340, 1380), seconds=8.0, seed=1):
    """Straight horizontal right-moving lanes; a car's box is car_length / k(x) px long."""
    rng = np.random.default_rng(seed)
    rows, tid = [], 0
    for c in lanes_c:
        for j in range(10):
            tid += 1
            x = 50.0 + j * 330
            for f in range(int(seconds * FPS)):
                if not 60 < x < W - 60:
                    break
                L = 4.5 / k_of_x(x)
                y = c + rng.normal(0, 1.0)
                rows.append(dict(t=f / FPS, frame_idx=f, track_id=tid, cls="car", x_px=x, y_px=y,
                                 x1=x - L / 2, x2=x + L / 2, y1=y - 15, y2=y + 15))
                x += 12.0 * 0.04 / k_of_x(x) / FPS * 25          # moves ~ a constant speed in metres
    return pd.DataFrame(rows)


def test_vehicle_size_recovers_a_uniform_scale():
    pts = _lane_road(lambda x: 0.05)
    geo, lanes, _ = estimate_lanes(pts, W)
    rf, info = C.fit_vehicle_size(pts, geo, lanes, W, lane_width_m=3.6, car_length_m=4.5)
    assert abs(info["scale_m_per_px_at_centre"] - 0.05) / 0.05 < 0.02
    s, d = rf.sd(np.array([1000.0, 3000.0]), np.array([1340.0, 1340.0]))
    assert abs((s[1] - s[0]) - 100.0) < 2.0                         # 2000 px * 0.05 m/px
    lanes_d = C.lane_centres_d(rf, geo, lanes)
    assert np.allclose(np.diff(sorted(lanes_d.values())), 3.6, atol=0.05)


def test_vehicle_size_follows_perspective_along_the_road():
    k = lambda x: 0.05 * np.exp(0.6 * (x - W / 2) / W)               # noqa: E731  far side 1.8x coarser
    pts = _lane_road(k)
    geo, lanes, _ = estimate_lanes(pts, W)
    rf, info = C.fit_vehicle_size(pts, geo, lanes, W, lane_width_m=3.6, car_length_m=4.5)
    xs = np.linspace(500, 3300, 2001)
    true = np.trapezoid(k(xs), xs)
    s, _ = rf.sd(np.array([500.0, 3300.0]), np.array([1340.0, 1340.0]))
    assert abs((s[1] - s[0]) - true) / true < 0.02
    assert abs(info["scale_ratio_right_vs_left_edge"] - np.exp(0.6)) < 0.05


def test_lateral_nodes_keep_the_median_gap_in_proportion():
    lanes = [{"c": 100, "direction": "left"}, {"c": 140, "direction": "left"}, {"c": 180, "direction": "left"},
             {"c": 260, "direction": "right"}, {"c": 300, "direction": "right"}]
    c, d = C.lateral_nodes(lanes, 3.6)
    inner = dict(zip(c[1:-1], d[1:-1]))
    assert inner == pytest.approx({100: 0.0, 140: 3.6, 180: 7.2, 260: 14.4, 300: 18.0})


# --------------------------------------------------------------------------- ground table
def test_build_ground_makes_ids_unique_and_keeps_a_shared_frame_once(tmp_path):
    def pts(frames, ids):
        return pd.DataFrame([dict(t=f / FPS, frame_idx=f, track_id=i, cls="car", x_px=100.0 + f, y_px=50.0,
                                  x1=90.0, x2=110.0, y1=40.0, y2=60.0) for f in frames for i in ids])
    pts(range(0, 31), [1, 2]).to_parquet(tmp_path / "cam_0-1s_points.parquet")
    pts(range(30, 61), [1, 3]).to_parquet(tmp_path / "cam_1-1s_points.parquet")
    for tag, frames in (("cam_0-1s", range(0, 31)), ("cam_1-1s", range(30, 61))):
        pd.DataFrame([{"frame_idx": f, **H_ID, "h02": 2.0} for f in frames]).to_parquet(tmp_path / f"{tag}_stab.parquet")
    cfg = {"lane_speed": {"windows": [[0, 1], [1, 1]]}}
    g = world.build_ground(cfg, "configs/cam.json", tmp_path)
    assert g.frame_idx.value_counts().max() == 2                    # frame 30 kept once (2 tracks)
    assert set(g.track_id) == {1, 2, world.TRACK_ID_STRIDE + 1, world.TRACK_ID_STRIDE + 3}
    assert np.allclose(g.x_px - g.x_raw, 2.0)                       # stabilisation applied
