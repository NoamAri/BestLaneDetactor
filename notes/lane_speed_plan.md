# Plan: speed per lane, and announcing the fastest and slowest lane

Status: **plan only, nothing implemented.** Builds on detection, tracking (`notes/tracking.md`) and
lanes (`notes/lanes.md`). Replaces phases 5–7 of `PLAN.md` where they differ.

## 0. The question, stated precisely

For each direction of travel and each time window: **which lane moves traffic fastest and which
slowest, and is the difference real or noise?**

"Faster lane" must mean: *a driver who stays in this lane covers the measured stretch of road in less
time.* That fixes which mean to use (section 3) and rules out most shortcuts.

The answer is one of three outcomes per direction and window:

| Outcome | When |
|---|---|
| **Fastest: lane k, slowest: lane j** (with the gap and its uncertainty) | the evidence is strong (section 6) |
| **No meaningful difference** | free flow, or every gap is small or uncertain |
| **Not enough data** | too few vehicles or too little observed time in some lane |

The result is always a measurement, never driving advice.

## 1. Why pixels are not enough: calibration first

Speeds today are in image pixels per second. A lane further from the camera looks narrower and
slower in pixels even when traffic in it is faster. **Comparing lanes in px/s is biased by
perspective, so the first step is a pixel-to-metre mapping.** Per camera, in order of preference:

| Camera | Method | Check |
|---|---|---|
| MiTra (has ground truth) | **Automatic**: match our tracks to the ground-truth trajectories (UTM metres, same clock: CSV time 0 = video start, clip offset 300 s) and fit a homography with RANSAC on thousands of matched points | reprojection error in metres; lane width ~3.5–3.75 m |
| Uploaded clip (no ground truth) | **From the road itself**: the lane model already gives lane centrelines; lane width (measured standard, e.g. 3.5–3.6 m, confirmed from the marking pattern) fixes the lateral scale; the dash-and-gap length of the lane markings fixes the scale along the road. Fit a homography to these lines and marks | lane widths equal across lanes after mapping; dash lengths equal along the road |
| Any camera (fallback) | 4–8 hand-placed points with known spacing (marking ends, poles) | as above |

Requirements:
* The mapping is checked **per lane**. A scale error shared by all lanes cancels in a ratio between
  lanes; an error that grows across the image does not, and is the one that matters here.
* A drone camera drifts slowly, so a fixed mapping slides. Correct drift first (phase 3, `stabilize.py`):
  register each frame to a reference frame using static background. Acceptance: parked or stationary
  objects show < 0.5 km/h apparent motion.
* Each camera config records the scale (m/px) and its error, so every speed can carry its calibration
  uncertainty.

## 2. From boxes to clean vehicle positions

For every tracked box, in order:

1. **Ground point.** The bottom-centre of the box lies on the side of the vehicle nearest the
   camera, not under its centre. Shift it by half the vehicle width away from the camera (in metres,
   after mapping). Without this, wide vehicles (trucks, buses) are placed half a lane toward the
   camera. That error is class-dependent, so it would bias lanes with many trucks.
2. **World position** (x_w, y_w) by the mapping, then **road coordinates**: s = distance along the
   lane centreline (arc length), d = lateral offset from it. Use s for speed and d for lane membership.
3. **Occlusion filter.** Drop boxes much shorter than the track's median box (height < 70%, i.e. the
   lower part is hidden by trees or a truck): their ground point is wrong. Keep the track.
4. **Plausibility filter on each step between consecutive boxes of a track**: speed > 200 km/h,
   |acceleration| > 5 m/s², or a lateral jump of more than half a lane in < 0.5 s are physically
   impossible. They mark an identity jump (the box landed on another vehicle). Cut the track there.

Time always comes from timestamps (t = frame / fps), never frame counts: the two cameras run at
29.97 fps and 4 fps.

## 3. Which "mean speed": space-mean, measured the Edie way

There are two common means, and they answer different questions:

* **Time-mean speed**: the average speed of the vehicles passing a point. Fast vehicles pass more
  often, so they are over-represented. Wrong for "which lane gets me through faster".
* **Space-mean speed**: total distance travelled ÷ total time spent, over a stretch of road. Travel
  time over a stretch of length L is exactly L / v_space. **This is the one to use.**

Compute it with **Edie's definitions** over a space-time region R = (lane ℓ) × (road stretch
[s0, s1] of length L) × (time window [t0, t0 + T]):

    D_ℓ = Σ over all vehicle-pieces in R of the distance they travel inside R      (metres)
    τ_ℓ = Σ over all vehicle-pieces in R of the time they spend inside R           (seconds)

    space-mean speed   v_ℓ = D_ℓ / τ_ℓ
    density            k_ℓ = τ_ℓ / (L · T)        vehicles per metre
    flow               q_ℓ = D_ℓ / (L · T)        vehicles per second   (q = k · v holds exactly)

Why this definition fits our data so well:

* **ID changes do not matter.** If the tracker splits one car into two IDs, its distance and time
  are split between the two pieces, and the sums D and τ are unchanged. Vehicle counts would double
  count it; Edie sums do not. Remaining tracker breaks (about 450 per 30 s on MiTra) therefore stop
  being a problem for speed.
* **Lane changes are handled exactly.** A car that drives 40 m in lane 2 and then 60 m in lane 3
  adds 40 m and its time to lane 2, and the rest to lane 3 (section 4).
* **Vehicles entering or leaving mid-window, stopped vehicles, partial views**: all contribute
  exactly the distance and time actually observed. A stopped car adds time and no distance, which
  is what makes a jammed lane slow.
* It is the harmonic mean of individual speeds weighted by distance, which is the travel-time mean.

Also report, for display and as a robustness check, the **median of per-vehicle speeds** in the lane
(each vehicle's D/τ inside R). If the median and v_ℓ disagree strongly, a few outliers or stopped
vehicles dominate, and the panel shows that.

## 4. Lane membership over time ("the lane index can change")

A vehicle's lane can change during its track, and its box can flicker across a lane line near the
boundary. Use per-box lanes (`lane_frame`), never one lane per track:

1. **Smooth the lane sequence of each track with hysteresis**: a vehicle is counted in a new lane
   only after it has stayed there for at least 0.5 s (≥ 2 boxes at 4 fps). Short flickers at lane
   lines are ignored.
2. **Split the track into lane pieces** at confirmed changes. The change itself is placed at the
   moment the ground point crosses the lane line (lateral offset d = half a lane), interpolated
   between the two boxes either side.
3. Each piece contributes its own distance and time to its own lane (section 3).
4. Boxes outside every lane band (`lane_frame = -1`: shoulder, slip road) contribute to no lane.
5. **Count lane changes** per lane and direction (from → to). They are a measurement in themselves,
   and a check: lanes that look fast should attract changes into them.

Lane numbering is fixed per camera (`configs/lanes/<camera>.json`, lane 1 = leftmost in the direction
of travel), so results from different windows and days stay comparable.

## 5. Where and when to measure

**Road stretch [s0, s1]: the same for every lane of a direction.** Comparing lanes over different
stretches would compare different bits of road. Choose, per camera:

* the longest stretch that every lane of the direction covers, minus the image edges (partial
  boxes) and minus zones with poor detection: trees (MiTra near lanes), the sign gantry, merge
  areas;
* at least 40 m long (`PLAN.md`), so even a fast vehicle is observed for over a second;
* auxiliary lanes (MiTra's ramp lane) are measured and shown but **not ranked**: drivers cannot
  simply choose them.

**Time windows.** Rolling 60 s windows every 5 s for the live picture, plus one summary over the
whole clip. Windows overlap by 92%, so consecutive results are strongly correlated; persistence
questions must use non-overlapping windows (see `PLAN.md` phase 8).

**Minimum evidence per lane and window** (else "not enough data"): at least 5 distinct vehicle
pieces and at least 20 s of observed vehicle time τ_ℓ, and the lane's detection coverage must not
be much worse than its neighbours'. Coverage is reported per lane, because a lane with more misses
(far lanes behind trucks) can be biased toward the vehicles that are easy to see.

## 6. Uncertainty, and when to announce

Vehicles in a lane interact (a jam slows them all) and each contributes many boxes, so boxes are not
independent samples. Resample **whole vehicles**:

1. **Cluster bootstrap** (2,000 resamples): in each resample, draw vehicles with replacement within
   each lane (all pieces of a drawn vehicle together), recompute every lane's v_ℓ. One resample
   gives one value per lane, all computed together.
2. From the resamples: a 95% interval for each v_ℓ; for every pair of lanes the interval of the
   **ratio** v_a / v_b; and for every lane **P(fastest)** and **P(slowest)**: the share of resamples
   in which it ranks first or last. Ranking all lanes in each resample handles the multiple
   comparison between k lanes directly.
3. **Announce "fastest: lane a"** only if
   * P(fastest) ≥ 0.90, and
   * the gap to the runner-up is at least 10% (v_a / v_runner-up ≥ 1.10), and
   * the 95% interval of that ratio is above 1.
   Same for "slowest" with the second-slowest lane. Otherwise: no meaningful difference.
4. **Free flow**: if every lane is at ≥ 85% of the speed limit, report "free flow, no meaningful
   difference" (the limit is looked up per road, never guessed).
5. **Stability over time** (rolling windows): once announced, a lane keeps the title until its gap
   falls below 5% or its interval includes 1. A different lane takes over only after leading for at
   least 60 s of fresh data (one full window), so the announcement does not flicker with stop-and-go
   waves.
6. **Calibration uncertainty**: repeat the bootstrap with the mapping perturbed within its measured
   error. If that can change the ranking, the announcement says so.

All thresholds (10%, 0.90, 5%, 60 s, 85%, minimum evidence) are written into the config **before**
any result is looked at, as for the tracker choice, and confirmed on held-out windows.

## 7. Validation

**MiTra (ground truth available).** Compute the same per-lane, per-window quantities from the
ground-truth trajectories, with the ground truth's own lane labels mapped to ours through the
calibration, over the same stretch and windows. Report:

| Check | Target |
|---|---|
| per-lane v_ℓ: absolute error (km/h) and relative error | relative ≤ 10% |
| ratio between lanes v_a / v_b vs ground truth | ratio error smaller than the absolute error (calibration error is mostly shared) |
| ranking (fastest, slowest) agreement | confusion table; announcements must agree with GT announcements |
| announcement precision: when we announce, is the GT gap really ≥ 10%? | ≥ 90% |
| share of each lane's distance and time vs GT | catches lane-assignment bias |
| effect of each step (ground-point shift, occlusion filter, hysteresis) on the errors | ablation table |

**Uploaded clip (no ground truth).** Self-consistency only: lane widths and marking lengths after
calibration; the ranking must not depend on the stretch chosen (compare the near and far halves);
results stable when the window moves by 5 s.

**Synthetic tests with known answers** (as for tracking and lanes): lanes with set speeds; a vehicle
changing lanes half way; one vehicle split into three IDs (v_ℓ must not change); a stopped lane; a
wide truck (ground-point shift); an identity jump (must be cut); perspective (far lane slower in
pixels, equal in metres).

## 8. Things that can go wrong, and the answer to each

| Risk | Effect | Answer |
|---|---|---|
| Perspective | far lanes look slow in px/s | calibrate to metres (1) |
| Scale error varying across the image | biased ratios between lanes | check the mapping per lane (1) |
| Drone drift | stationary things look like they move | stabilise first (1) |
| Track split into several IDs | double-counted vehicles | Edie sums are invariant (3) |
| Box jumps to another vehicle | spurious distance | plausibility cut (2) |
| Lane change | speed credited to the wrong lane | per-box lanes, pieces, hysteresis (4) |
| Flicker at a lane line | fake lane changes | hysteresis 0.5 s (4) |
| Tall or wide vehicles | ground point half a lane off | shift by half the width (2) |
| Bottom of a car hidden | ground point too high | drop boxes with a cut-off height (2) |
| Different stretch per lane | comparing different road | one common stretch (5) |
| Lane with more misses | biased sample | coverage per lane; minimum evidence (5) |
| Few vehicles | noisy speeds | minimum evidence, bootstrap by vehicle (5, 6) |
| Many lanes compared at once | a "winner" by chance | rank probability P(fastest) ≥ 0.9 (6) |
| Stop-and-go waves | ranking flips | 60 s windows, hysteresis on the announcement (6) |
| Trucks only in right lanes | right lanes slower for that reason | show truck share per lane; optional cars-only ranking |
| Ramp or merge lane | not a real choice | measured, not ranked (5) |
| 4 fps video | few boxes per vehicle | distances over ≥ 1 s; Edie sums do not need per-vehicle fits |
| Thresholds tuned after the fact | over-optimistic announcements | fixed in config first, held-out windows (6) |

## 9. Implementation steps (when approved)

| Step | Module | Output | Done when |
|---|---|---|---|
| 1 | `stabilize.py` | per-frame correction to a reference frame | stationary objects < 0.5 km/h |
| 2 | `calibrate.py` | m/px mapping per camera, with error | MiTra reprojection < 0.5 m; lane widths 3.5–3.75 m |
| 3 | `world.py` | ground point, (x_w, y_w), (s, d) per box; occlusion and plausibility flags | synthetic truck and occlusion tests pass |
| 4 | `lane_pieces.py` | lane pieces per track, lane-change events | synthetic lane-change test passes |
| 5 | `lane_speed.py` | Edie D, τ, v, k, q per lane, window and direction → `windows.parquet` | ID-split invariance test passes |
| 6 | `announce.py` | bootstrap, P(fastest/slowest), status, hysteresis | hand-built cases pass |
| 7 | `validate.py` | MiTra comparison tables (section 7) | targets met, or reasons recorded |
| 8 | viewer | "Lane speeds" panel: km/h per lane with intervals, the announcement and its confidence, coverage, truck share; a per-lane speed timeline | browser check |

Steps 1–2 decide everything after them, so they come first and get their own checks and notes.

## 10. Decisions needed from you

1. **Thresholds** for announcing: 10% gap, P ≥ 0.90, 60 s windows (proposed above).
2. **Vehicle mix**: rank on all vehicles (proposed) or cars only?
3. **Speed limits** of both roads (to detect free flow); they must be looked up, not guessed.
4. **The uploaded clip's road**: if you know the location, its lane width and marking standard
   make that calibration more reliable.

## 11. Pre-registration (before any lane speed was computed)

Settings: `configs/lane_speed.json` (thresholds, evidence, stretch rule, plausibility limits, ground-point
widths) and per camera `calibration` + `lane_speed` blocks. Open decisions resolved with the proposed
defaults: announce at a 10% gap, P ≥ 0.90, 60 s windows; all vehicles ranked; speed limits unknown, so
the free-flow check is off until they are given; uploaded clip's lane width assumed 3.6 m (to confirm).
Stretch: the common covered range of all ranked lanes of a direction, 5% margins, ends trimmed where a
lane's detection coverage falls below 50% of its median, at least 40 m.
MiTra uses 0–180 s of the clip (six 30 s detection windows tracked separately and joined; Edie sums do
not care that IDs restart between windows).

### Amendment A2: ground point = box centre (recorded before any lane speed was computed)

Section 2.1 pre-registered the box bottom shifted by half the vehicle width. The ground-truth check
(notes/lane_speed.md §1.3) contradicts it: with the box centre, 98.1% of car boxes, 95.9% of truck boxes
and 98.7% of bus boxes fall in the lane the ground truth gives; with the bottom-shift, 86.8%, 84.6% and
46.6%. Trucks and cars of the same ground-truth lane sit at the same box-centre height. The lanes are
also estimated from box centres. So the ground point is the box centre (in the stabilised frame); the
shift stays in the code (`world.ground_point(mode="bottom_shift")`) for the ablation table.
`ground_point.class_width_m` in configs/lane_speed.json is therefore unused.

### Amendment A3: stabilisation, per-camera analysis windows

Added before any lane speed: camera motion is removed (section 1 named a drift check; the check found
up to 19 px of motion across the road, half a lane), and lanes are re-estimated on stabilised points
from all analysis windows of a camera (`lane_speed.windows` in the camera config).
