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
| `speeds` | `world.py`, `lane_pieces.py`, `lane_speed.py`, `announce.py` | `data/out/lane_speed/<camera>/`: lanes, vehicles, lane changes, announcements, summary |
| `validate` (MiTra) | `validate.py` | `data/out/lane_speed/<camera>/validation.json` |

## 1. Calibration

### 1.1 Camera motion (stabilize.py)

Corner features on the background of the first frame (vehicle boxes masked), followed into every frame
with pyramidal Lucas-Kanade (forward-backward check < 0.5 px), homography by RANSAC; directly against
the first frame, so errors do not accumulate.

| Camera / window | Frames | Inliers (median, min) | Fit error (px) | Road shift median / max (px) |
|---|---|---|---|---|
| MiTra 0-30 s | 900 | 2666 / 2540 | 0.21 | 13.6 / 19.9 |
| MiTra 30-60 s | 899 | 2608 / 2500 | 0.24 | 12.3 / 16.4 |
| MiTra 60-90 s | 899 | min 2536 | | 7.2 / 32.2 |
| MiTra 90-120 s | 899 | min 2507 | | 42.5 / 50.6 |
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

## 2. Measurement

* **Road points** (`world.road_points`): s, d per box from the calibration; boxes shorter than 70% of
  their track's median height are dropped (occluded).
* **Lane pieces** (`lane_pieces.py`): 0.5 s hysteresis per track, then each confirmed change is moved to
  where the vehicle crosses the line between the two lanes (middle crossing of the midline within
  0.5 s); the step across it is split at the interpolated crossing.
* **Edie sums** (`lane_speed.py`): each straight piece between two boxes is clipped exactly to
  lane x stretch x window; distance is the signed displacement in the direction of travel, so a
  standing vehicle's jitter cancels (summing step lengths would invent several m/s in a jam).
* **Announcement** (`announce.py`): cluster bootstrap over whole vehicles (2000), P(fastest/slowest),
  ratio interval, the pre-registered rule, hysteresis over rolling windows, and a calibration check
  (lane scales perturbed by the measured lane-to-lane calibration spread).

Synthetic known-answer tests (`tests/test_lane_speed.py`): exact clipping; one vehicle split into IDs
leaves v unchanged; a standing car adds time, not distance; a lane change with flicker at the line is
one change at the true crossing; an identity jump is cut once and braking at 3 m/s^2 is not; clear
fastest lane announced, equal lanes or a 6% gap not; too few vehicles -> "not enough data"; free flow;
calibration sensitivity; hysteresis (keeps through a wobble, a new lane needs a fresh minute);
perspective end to end (lanes 14% apart in pixels, equal in metres, recovered within 1%).

## 3. Development on MiTra 0-120 s, and the post-hoc amendments

The first comparison with ground truth (0-120 s, the same measurement applied to the ground-truth
trajectories; amendments up to A4) gave a median relative error of 6.8% with a bias of -5.9%. Two causes
were found, each by a diagnostic, not by tuning:

1. **A tree zone inside the stretch.** Coverage against ground truth along the road (8 bins): in
   s = -106 .. -32 m the near lanes were seen only 22-72% and their speeds read 10-25% low (vehicles
   partly hidden: boxes shrink and lag). Elsewhere coverage 0.9-1.05 and errors 0-5%. The pre-registered
   coverage rule did flag those bins (20-59% of the lane median) but only trimmed from the ends, and the
   zone lies inside. **A5**: the stretch is the longest contiguous run of well-covered bins (same 50%
   threshold, same 12 bins).
2. **False plausibility cuts.** 809 of 944 cuts came from the acceleration test: box shapes wobble
   slowly (shadows, neighbours), so speeds fitted over 0.5 s swing by 3-5 m/s with no real acceleration.
   After making the test aware of that wobble (A5) 535 cuts remained, still mostly ordinary steps
   (median 0.31 m in one frame); only 6% were lateral jumps. Two variants were compared with a rule fixed
   beforehand (lower median relative error, then precision):

   | Variant (0-120 s) | Cuts | Median rel. error | Within 10% | Fastest precision |
   |---|---|---|---|---|
   | (a) no cuts | 0 | **3.4%** | 69% | **0.94** |
   | (b) step + lateral tests only | 175 | 3.8% | 75% | 0.88 |
   | all tests (A5) | 535 | 4.0% | 68% | 0.88 |

   **A6**: plausibility cuts off. Edie sums are invariant to identity breaks, which is what the cuts
   were meant to guard; the code stays for diagnostics and the ablation table.

Validation also merges our lanes that fall in one ground-truth lane (with 120 s of data a fifth left
lane appeared at the edge; ground truth counts it in lane 7) so lanes are compared like for like.

Both amendments were decided on 0-120 s only. **Windows 120-180 s were not looked at** and are the
held-out confirmation (section 4).

## 4. MiTra 0-180 s: final measurement and held-out confirmation

Six 30 s windows, 1949 tracks, stabilised, one calibration and one set of lanes for the whole span
(9 lanes: 5 left, incl. an edge lane with 4 vehicles = low evidence; 4 right, incl. the ramp R4).
Stretches: right 135 m, left 50 m (the far left lanes are poorly covered outside it). 83 lane changes.

| | Rolling 60 s windows, development (0-120 s) | Fixed window 0-120 s | **Held out 120-180 s** |
|---|---|---|---|
| Median relative speed error | 2.6% (0.59 km/h) | 4.4% | **4.2%** |
| Bias | -2.1% | -1.4% | **-1.0%** |
| Lane-ratio error | 4.4% | 5.0% | 6.2% |
| Rule agrees with GT (fastest / slowest) | 96% / 85% | 100% / 100% | **100% / 100%** |
| Announcement precision (fastest / slowest) | 0.909 / 1.0 | - / 1.0 | nothing announced (GT: nothing either) |

All rolling windows: median relative error 2.6% (0.43 km/h), bias -1.5%;
precision fastest 0.947, slowest 1.0 (targets: 10%, 0.90). **Targets met; the held-out
part confirms amendments A5-A6** (error and bias no worse than in development, the rule agrees with ground truth).

Whole span per lane (ours vs ground truth; distance share ours / GT):

| Lane | Ours (km/h) | GT (km/h) | Error | Distance share |
|---|---|---|---|---|
| L1 | 3.4 | 3.4 | -0.8% | 0.39 / 0.30 |
| L2 | 1.5 | 1.3 | +13.5% | 0.15 / 0.10 |
| L3 | 3.0 | 3.1 | -2.0% | 0.15 / 0.25 |
| L4 | 4.1 | 3.8 | +7.9% | 0.31 / 0.36 |
| R1 | 41.6 | 42.6 | -2.3% | 0.35 / 0.30 |
| R2 | 38.4 | 38.9 | -1.4% | 0.35 / 0.31 |
| R3 | 36.1 | 36.7 | -1.6% | 0.26 / 0.23 |
| R4 (ramp) | 44.4 | 50.6 | -12.2% | 0.04 / 0.16 |

Announcement over 0-180 s: right-moving traffic **no meaningful difference** (lane 1 leads lane 2 by 8%, P = 1.00, below the
10% gap); left-moving **slowest: lane 2** (1.5 km/h, P = 0.99). Ground truth gives the same announcement. The ramp lane
reads 12% low (vehicles merge across it, few observations); it is measured but never ranked. The jammed lanes have large
*relative* errors on tiny speeds (L2: 1.5 vs 1.3 km/h) but are ranked correctly.

Ablation (all rolling windows; each row switches one step):

| Variant | Median rel. error | Within 10% | Fastest precision |
|---|---|---|---|
| final (A2-A6) | 2.6% | 65% | 0.947 |
| no occlusion filter | 2.5% | 65% | 0.947 |
| no lane hysteresis | 2.7% | 64% | 0.95 |
| with plausibility cuts (pre-registered) | 3.3% | 65% | 0.9 |
| with plausibility cuts, step + lateral only | 3.0% | 69% | 0.905 |

Known limits: the left stretch is short (50 m) because the far lanes are hidden by the trees elsewhere; the absolute speed
reads ~1-2% low (calibration); a 60 s window cannot separate lanes in a jam with few moving cars (L4 leads by 32% but
P = 0.81, so nothing is announced, correctly).

## 5. Uploaded clip (no ground truth)

0-64 s, one 60 s window, 147 tracks, common stretch 79 m (6 lanes, all moving left):

| Lane | Space-mean (km/h) | 95% interval | Median vehicle | Vehicle pieces |
|---|---|---|---|---|
| 1 | 52.6 | 51.3-54.2 | 53.5 | 33 |
| 2 | 50.9 | 49.2-52.7 | 50.9 | 27 |
| 3 | 46.3 | 44.2-48.4 | 47.0 | 24 |
| 4 | 36.6 | 34.0-39.3 | 37.3 | 26 |
| 5 | 11.3 | 9.2-13.9 | 10.4 | 20 |
| 6 | **4.5** | 3.9-5.1 | 4.8 | 24 |

**Slowest: lane 6** (P = 1.00; lane 5 is 2.5x faster, interval 1.98-3.22). **Fastest: no meaningful
difference** (lane 1 leads lane 2 by 3.4%, interval 0.99-1.08, P = 0.94 but gap < 10%). The file was
named "3.9 km/h" by its uploader; the measurement (made without using that label) gives 4.5 km/h, with the
vehicle-size calibration reading ~4% low on MiTra.
