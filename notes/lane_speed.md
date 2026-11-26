# Lane speeds: results log

Plan and pre-registration: `notes/lane_speed_plan.md` (amendments A2, A3 at its end).
Settings: `configs/lane_speed.json`, per camera `calibration` and `lane_speed` blocks.

Pipeline per camera (each a `run.py` step):

| Step | Code | Output |
|---|---|---|
| `stabilize` (per window) | `stabilize.py` | `data/interim/<tag>_stab.parquet`: homography frame -> first frame |
| `ground` | `world.py` | `data/interim/<camera>_ground.parquet`: all windows, stabilised box centres, unique track ids |
| `lanes` | `lanes.py` | `configs/lanes/<camera>.json` in the stabilised frame, lane per box |
| `calibrate` | `calibrate.py` | `configs/calib/<camera>.json`: pixels -> s (m along the road), d (m across) |

## 1. Calibration

### 1.1 Camera motion (stabilize.py)

Corner features on the background of the first frame (vehicle boxes masked), followed into every frame
with pyramidal Lucas-Kanade (forward-backward check < 0.5 px), homography by RANSAC; directly against
the first frame, so errors do not accumulate.

| Camera / window | Frames | Inliers (median, min) | Fit error (px) | Road shift median / max (px) |
|---|---|---|---|---|
| MiTra 0-30 s | 900 | 2666 / 2540 | 0.21 | 13.6 / 19.9 |
| MiTra 30-60 s | 899 | 2608 / 2500 | 0.24 | 12.3 / 16.4 |
| Upload | 257 | 573 / 526 | 0.11 | 3.1 / 4.4 |

MiTra: across the road the motion reaches 19 px at the left edge (half a lane: lanes are 34-40 px
apart), along it 14 px. Small for speeds over a minute (~0.1 km/h), large for lane membership, so
lanes are estimated and assigned in the stabilised frame. Test: a synthetic drifting camera (sub-pixel
steps) is recovered to < 0.15 px.

### 1.2 MiTra: homography from ground truth

The MiTra ground truth gives every vehicle's position (UTM metres) on the drone's clock. Method
(`fit_gt_homography`): rough start from road geometry alone (road heading and which carriageway is
fast, from both sides), refined by per-frame one-to-one matching (Hungarian) with a shrinking gate
(12 m -> 1.5 m) and RANSAC; then identity pairs (a track paired with the vehicle it matches in > 80%
of frames), then a 10 ms scan of the clock offset on moving vehicles (standing ones do not constrain
time), then the final homography (RANSAC 1.5 m) on every box of every pair.

Match quality: 98% of tracks match one ground-truth vehicle in > 90% of their frames.

| 0-30 s | Unstabilised | Stabilised |
|---|---|---|
| Position error, median (target 0.5 m) | 0.69 m | **0.45 m** (cars 0.42, trucks 0.84, buses 1.3) |
| Distance along the road, ours / ground truth (108 vehicles) | 0.980 | **0.986** (p10-p90 0.95-1.01) |
| Median error of distance travelled (target 10%) | 2.5% | **2.0%** |
| Clock offset (ground truth ahead of video) | | 0.03 s |

Per lane the distance ratio is 0.95-1.00 (L1-L4, R1-R3), so calibration does not favour a lane. The
camera is oblique: 0.062 m/px along the image x, 0.100 m/px down the image y at the centre, so lanes
34-40 px apart are 3.3-4.1 m wide (the ramp lane R4 merges and is narrower).

Residuals by time before stabilisation drifted by 0.7 m along and 1.1 m across the road within 30 s;
after, 0.25 m and 0.07 m. Remaining: ~0.35 m at the far left and right edges (lens), and a larger error
for tall vehicles.

### 1.3 Ground point (amendment A2)

| Ground point | Cars in their ground-truth lane | Trucks | Buses |
|---|---|---|---|
| Box centre | **98.1%** | **95.9%** | **98.7%** |
| Box bottom, shifted by half the vehicle width (pre-registered) | 86.8% | 84.6% | 46.6% |

(our lanes mapped to ground-truth lanes by the cars' majority). Trucks and cars of one ground-truth lane
sit at the same box-centre height, within a few pixels. The position error is also lowest near the
box centre. Our lanes map one-to-one onto the ground truth's: L1-L4 = GT 4-7, R1-R3 = GT 3, 2, 1,
R4 = GT 10 (the on-ramp, an auxiliary lane).

### 1.4 Cameras without ground truth: scale from vehicles

Across the road: neighbouring lanes of one carriageway are one lane width apart (assumed 3.6 m);
the gap between carriageways keeps its proportion. Along the road: a car's box length measured along
its lane line is 4.5 m (pre-registered). Metres per pixel k(x, c) = exp(a + b x' + e c') is fitted to
median log k in position bins (unoccluded cars fully inside the frame); s is the integral of k along
each lane line. Tests: a uniform scale and a perspective ramp are recovered to < 2%.

Validated on MiTra against the ground-truth homography (same vehicles, distance along the road):

| Car length assumed | Ratio to ground truth | Median error |
|---|---|---|
| **4.5 m (pre-registered, used)** | 0.957 | **4.4%** |
| 4.7 m | 1.000 | 2.7% |
| 5.0 m | 1.063 | 6.6% |

A box is slightly longer than the car, so 4.5 m reads distances ~4% short. That scales every lane of
a camera alike, so rankings and ratios are unaffected; only absolute km/h are ~4% low. Per lane:
0.95-0.97 in the moving lanes, 0.90 in jammed L2 (10 vehicles): a lane-dependent error of up to ~7%
is possible, which is below the 10% gap needed to announce but not negligible.

Uploaded clip: 6 lanes (curved, degree 2), 0.099 m/px along the road at the centre, the scale changes
2.1x from the left edge to the right (the camera looks along the road). The model follows the measured
car lengths within +-4% except the leftmost 150 px (+9%, cars entering the frame); the measuring
stretch trims the edges.
