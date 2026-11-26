"""Camera motion check and correction (notes/lane_speed_plan.md §1, step 1).

A hovering drone drifts and wobbles. Every pixel position is therefore moved into one fixed
reference frame (the first frame of the video) before anything is measured in metres:

  1. In the reference frame, pick corner features on the static background: vehicle boxes are
     masked out using the detections of that frame.
  2. Follow those features into every later frame with pyramidal Lucas-Kanade optical flow
     (directly against the reference, so errors do not accumulate frame to frame).
  3. Fit a homography frame -> reference with RANSAC; moving vehicles and swaying trees are outliers.

Output per frame: the 3x3 homography in full-resolution pixels, the inlier count, and the median
inlier error. The camera's displacement over time is reported so it can be judged, and applied to
tracked points with ``apply``.

Run:  python -m lane_intel.stabilize --cfg configs/T8_D1.json --start 0 --dur 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lane_intel.video import iter_gray

WORK_WIDTH = 1920          # frames are analysed at this width (4K halved) for speed
MAX_FEATURES = 3000
RANSAC_PX = 1.0            # at WORK_WIDTH


def _work_width(cfg: dict) -> int:
    return min(WORK_WIDTH, cfg["frame_size"][0])


def _reference(cfg: dict, scale: float, ref_boxes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    _, ref = list(iter_gray(cfg["video"], tuple(cfg["frame_size"]), _work_width(cfg), 0.0, 0.5 / cfg["native_fps"]))[0]
    mask = np.full(ref.shape, 255, np.uint8)
    for b in ref_boxes.itertuples():
        x1, y1, x2, y2 = (int(v * scale) for v in (b.x1 - 6, b.y1 - 6, b.x2 + 6, b.y2 + 6))
        mask[max(0, y1):y2, max(0, x1):x2] = 0
    feats = cv2.goodFeaturesToTrack(ref, MAX_FEATURES, 0.01, 12, mask=mask, blockSize=7)
    return ref, feats


def estimate(cfg: dict, start_s: float, duration_s: float, ref_boxes: pd.DataFrame) -> pd.DataFrame:
    """Per-frame homography (full-res pixels) mapping that frame onto the video's first frame."""
    fps = cfg["native_fps"]
    scale = _work_width(cfg) / cfg["frame_size"][0]
    ref, feats = _reference(cfg, scale, ref_boxes)
    S = np.diag([scale, scale, 1.0])
    first = round(start_s * fps)
    rows = []
    for i, frame in iter_gray(cfg["video"], tuple(cfg["frame_size"]), _work_width(cfg), start_s, duration_s):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref, frame, feats, None, winSize=(31, 31), maxLevel=4)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(frame, ref, nxt, None, winSize=(31, 31), maxLevel=4)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - feats, axis=2)[:, 0] < 0.5)
        src, dst = nxt[ok].reshape(-1, 2), feats[ok].reshape(-1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX) if len(src) >= 30 else (None, None)
        if H is None:
            H, n_in, err = np.eye(3), 0, np.nan
        else:
            inl = inl[:, 0].astype(bool)
            proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
            n_in, err = int(inl.sum()), float(np.median(np.linalg.norm(proj - dst[inl], axis=1)) / scale)
            H = np.linalg.inv(S) @ H @ S                      # back to full-resolution pixels
        rows.append({"frame_idx": first + i, "n_features": int(len(feats)), "n_tracked": int(ok.sum()),
                     "n_inliers": n_in, "inlier_err_px": err, **{f"h{r}{c}": H[r, c] for r in range(3) for c in range(3)}})
    return pd.DataFrame(rows)


def matrices(stab: pd.DataFrame) -> dict[int, np.ndarray]:
    cols = [f"h{r}{c}" for r in range(3) for c in range(3)]
    return {int(f): m.reshape(3, 3) for f, m in zip(stab.frame_idx, stab[cols].to_numpy())}


def apply(stab: pd.DataFrame, frame_idx, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Move points (full-res pixels, per frame) into the reference frame."""
    cols = [f"h{r}{c}" for r in range(3) for c in range(3)]
    Hs = stab.set_index("frame_idx")[cols].reindex(np.asarray(frame_idx)).to_numpy()
    if np.isnan(Hs).any():
        raise ValueError("stabilisation missing for some frames")
    Hs = Hs.reshape(-1, 3, 3)
    p = np.stack([np.asarray(x, float), np.asarray(y, float), np.ones(len(Hs))], 1)
    q = np.einsum("nij,nj->ni", Hs, p)
    return q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]


def displacement(stab: pd.DataFrame, points: np.ndarray) -> pd.DataFrame:
    """How far given image points are moved by the correction, per frame (px)."""
    out = {}
    for f, H in matrices(stab).items():
        q = cv2.perspectiveTransform(points.reshape(-1, 1, 2), H).reshape(-1, 2)
        out[f] = np.linalg.norm(q - points, axis=1)
    return pd.DataFrame.from_dict(out, orient="index")


def stab_path(det_path: str | Path) -> Path:
    return Path(str(det_path).replace("_detections.parquet", "_stab.parquet"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--ref-detections", required=True, help="detections parquet containing frame 0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.cfg).read_text())
    det = pd.read_parquet(a.ref_detections)
    ref_boxes = det[(det.frame_idx == det.frame_idx.min()) & (det.conf >= 0.1)]
    t = time.time()
    stab = estimate(cfg, a.start, a.dur, ref_boxes)
    stab.to_parquet(a.out)
    band = cfg.get("band_y", [0, cfg["frame_size"][1]])
    W = cfg["frame_size"][0]
    probe = np.array([[x, (band[0] + band[1]) / 2] for x in (0.05 * W, 0.5 * W, 0.95 * W)], float)
    disp = displacement(stab, probe)
    summary = {"frames": len(stab), "seconds": round(time.time() - t, 1),
               "inliers_median": int(stab.n_inliers.median()), "inliers_min": int(stab.n_inliers.min()),
               "inlier_err_px_median": round(float(stab.inlier_err_px.median()), 3),
               "road_shift_px_median": round(float(disp.max(axis=1).median()), 2),
               "road_shift_px_max": round(float(disp.max(axis=1).max()), 2)}
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
