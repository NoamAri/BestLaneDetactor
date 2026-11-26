# Tracking: keeping one ID per vehicle

Status: steps 0–2 of the tracking plan are done (decision in section 4). Steps 3–6 (stitching,
occlusion zones, junk removal, ground-truth evaluation) come next.
Code: `src/lane_intel/detect.py`, `src/lane_intel/tracking/`. Variants: `configs/tracking.json`.
Run reports (configs, metrics, guard verdicts, detection provenance): `notes/runs/`.
Reproduce: `python run.py detect` then `python run.py tune` (add `--start 30` for the held-out window), then
`python -m lane_intel.tracking.crosscheck --tuning data/out/tracking/T8_D1_0-30s --heldout data/out/tracking/T8_D1_30-30s --out data/out/tracking/crosscheck.md`.

## 1. Problem

On the first 30 s of flight T8, drone D1 (clip frames 0–898, flight time 5:00–5:30), the
original pipeline (YOLO26s cut at confidence 0.25, ByteTrack with a 1 s memory) produced
**386 track IDs** while only about 110 vehicles are in view at any moment. A vehicle hidden
for a moment behind a truck or under the trees came back with a new ID.

Diagnosis on that run, before any change:

| Finding | Number |
|---|---|
| Tracks starting mid-image (not at the image edge, not at the window start) | 243 |
| Tracks ending mid-image | 234 |
| Likely break → restart pairs (same lane band, position consistent with speed, gap ≤ 5 s) | 121 |
| Gap length of those pairs: median / 75th percentile | 1.4 s / 3.1 s |
| Breaks where a truck or bus covers the spot the car vanished | 56 of 121 |
| Tracks shorter than 1 s (flicker, cab/trailer double boxes) | 102 |

The tracker forgot a lost vehicle after 1 s, shorter than most gaps.

Sampling 24 gaps and running YOLO at confidence 0.05 on the middle frame of each gap, at the
position the car should be:

| What YOLO found there | Gaps | Implication |
|---|---|---|
| a box with confidence 0.05–0.25 | 9 | car partly visible, but those boxes were discarded before tracking |
| a box with confidence ≥ 0.25 | 6 | car detected, tracker failed to re-associate it |
| nothing | 9 | fully hidden: needs motion prediction or offline stitching (step 3) |

## 2. Method

### 2.1 Separate detection from tracking (step 0)

* `lane_intel.detect` runs YOLO26s once per window and saves **every box with confidence ≥ 0.05**
  (`max_det` 1000) to `data/interim/<tag>_detections.parquet`, with a provenance sidecar
  (`.meta.json`: model, thresholds, versions, runtime).
* `lane_intel.tracking` runs any tracker over that file. A variant takes seconds instead of a
  12-minute YOLO run, and **every variant sees exactly the same boxes**, so differences between
  variants come from tracking alone.
* The tracker is updated on every frame of the window, including frames where no box passes
  the variant's `min_conf`, so a lost track's memory ages at the true rate.

### 2.2 Metrics (no ground truth needed)

Defined in `src/lane_intel/tracking/quality.py`. Positions are the bottom-centre anchor in
full-frame pixels; the image is 3840 px wide.

* **fragmentation** (primary): tracks that start mid-image + tracks that end mid-image.
  "Mid-image" = more than 120 px from the left/right border and more than 2 frames from the
  window start/end. On this road vehicles only enter and leave at the left/right edges, so
  every mid-image start or end is a break (or a junk track). Lower is better.
* **ids**, **short_lt_1s**, **median_span_s**: context.
* **restart_pairs**: one-to-one pairs (track end → later track start) that look like the
  same vehicle; an estimate of what offline stitching (step 3) could still repair.
* **identity_jumps** (guard): a track's box jumps to another vehicle. Both edges of the box
  move in the same direction by more than half the box size beyond what the track's recent
  speed predicts (+15 px per second of gap). Requiring both edges ignores boxes that shrink or
  grow while a car is partly hidden. Joining tracks more eagerly can cause wrong merges; this
  catches them.
* **coverage_clear** (guard): share of clearly detected boxes (confidence ≥ 0.5) that receive a
  track ID. Prevents "improving" by dropping boxes.

Validation of the metric code against the manual diagnosis on the original run: 386 IDs,
244 mid-image starts, 233 mid-image ends, 119 restart pairs (optimal matching) against
243 / 234 / 121 (greedy) by hand. Unit tests: `tests/test_tracking.py`.

### 2.3 Variants (fixed before the first run)

All in `configs/tracking.json`. Each variant inherits from a parent and changes one thing.

| Variant | Step | Change against its parent |
|---|---|---|
| `baseline` | 0 | the original run: min_conf 0.25, memory 1 s, ByteTrack (activation 0.5, high tier 0.5, IoU ≥ 0.1, 2 frames to confirm) |
| `s1_lowconf` | 1 | min_conf 0.10: faint boxes can extend existing tracks (ByteTrack's second stage) but never start one |
| `s2_bt_buf2s` / `s2_bt_buf4s` / `s2_bt_buf6s` | 2 | from s1: memory 2 / 4 / 6 s |
| `s2_bt_buf4s_iou03` | 2 | from buf4s: accept box overlap down to IoU 0.03 |
| `s2_bt_buf4s_act04` | 2 | from buf4s: confident tier starts at 0.40 instead of 0.50 |
| `s2_ocsort_buf4s` | 2 | OC-SORT, memory 4 s, min_conf 0.10 |
| `s2_botsort_buf4s` | 2 | BoT-SORT, memory 4 s, min_conf 0.10, no camera-motion correction, no re-identification |

BoT-SORT's camera-motion correction needs the video frames; the drone is nearly static and
drift is handled in Phase 3, so it is off here. Re-identification (appearance models) is out of
scope for steps 0–2.

### 2.4 Selection rule (fixed before the first run)

Implemented in `lane_intel.tracking.experiment.judge`:

1. **Guards** against the baseline:
   * identity_jumps ≤ baseline × 1.2 + 2
   * coverage_clear ≥ baseline − 1 percentage point
2. Among variants that pass both guards: **lowest fragmentation**, then fewest IDs, then fewest
   tracks under 1 s.

### 2.5 Reproduction check

Before comparing anything, `baseline` on the new detections must reproduce the original run
(386 IDs). If it does not, the new detection file differs from the old one and the comparison
with the earlier diagnosis is invalid.

### 2.6 Round 2 and held-out confirmation (fixed before the held-out results)

Round 1 (section 3.1) was followed by a visual audit of the identity-jump guard, which then
motivated six more variants (`r2_*`, step `2b` in the variants file). Because those were designed
while looking at window 0–30 s, their scores on that window are optimistic.

* **Tuning window**: clip 0–30 s (frames 0–899). Used to design and tune variants.
* **Held-out window**: clip 30–60 s (frames 899–1797), detected separately, never looked at
  while designing variants.
* **Final choice** = the selection rule of 2.4 applied to the held-out window, over all variants.
  `lane_intel.tracking.crosscheck` reports both windows side by side.

## 3. Results

Detections: 0–30 s → 900 frames, 121,671 boxes (91,320 with confidence ≥ 0.25), 679 s on CPU.
30–60 s → 899 frames, 115,590 boxes (87,584 ≥ 0.25), 641 s. Provenance in the `.meta.json`
next to each file. Every tracker variant ran in 4–10 s.

**Reproduction check: passed.** `baseline` on the new detections gives 386 IDs, identical to the
original run.

### 3.1 Round 1, tuning window 0–30 s

Full table: `notes/runs/T8_D1_0-30s_report.md` (machine-written copy of the run's report). Guard limits: identity jumps ≤ 24.8,
clear-box coverage ≥ 98.5%.

| Variant | IDs | Fragmentation | < 1 s | Restart pairs | Identity jumps | Coverage | Guards |
|---|---|---|---|---|---|---|---|
| `baseline` | 386 | 478 | 102 | 119 | 19 | 99.5% | pass |
| `s1_lowconf` | 349 | 395 (−17%) | 74 | 98 | 28 | 99.5% | fail: jumps |
| `s2_bt_buf2s` | 343 | 382 (−20%) | 71 | 90 | 28 | 99.5% | fail: jumps |
| `s2_bt_buf4s` | 341 | 379 (−21%) | 70 | 91 | 29 | 99.5% | fail: jumps |
| `s2_bt_buf6s` | 341 | 379 (−21%) | 70 | 91 | 29 | 99.5% | fail: jumps |
| `s2_bt_buf4s_iou03` | 331 | 359 (−25%) | 59 | 86 | 99 | 99.5% | fail: jumps |
| `s2_bt_buf4s_act04` | 417 | 501 (+5%) | 98 | 115 | 52 | 99.8% | fail: jumps |
| `s2_ocsort_buf4s` | 268 | 259 (−46%) | 44 | 49 | 44 | 95.3% | fail: jumps, coverage |
| `s2_botsort_buf4s` | 376 | 462 (−3%) | 92 | 131 | 0 | 99.3% | pass (selected in round 1) |

Observations:

* Feeding faint boxes (step 1) gave most of the gain. A longer memory added little beyond 2 s, and
  4 s and 6 s are identical: with ByteTrack's overlap-based matching, a vehicle that moved while
  hidden no longer overlaps its prediction, however long it is remembered.
* Faint boxes raised identity jumps from 19 to 28 and failed the guard. Looser overlap (IoU 0.03)
  made it far worse (99).
* OC-SORT had the lowest fragmentation but failed both guards: more jumps (median gap at a jump
  33 frames, so wrong re-associations after long gaps) and 4.6% of clear boxes left without an ID.

### 3.2 Audit of the identity-jump guard

Before trusting the guard, 8 of the jump events added by `s1_lowconf` were rendered from the video
(before / after crops). **All 8 were real identity errors**: a car's ID moving to the van below it
(twice), a trailer's ID moving to another truck's cab, a car's ID moving to a car in the next lane,
a trailer's ID landing on a faint box spanning two trucks, a car under the trees swapping to another
car, and a truck's ID flipping to a van and back. 20 of the 28 events were on boxes below 0.25.

Cause: in this oblique view, vehicles in a jam overlap each other heavily, and ByteTrack uses the
same low overlap threshold (IoU 0.1) for faint boxes as for confident ones. BoT-SORT requires IoU 0.5
for faint boxes, which explains its 0 jumps, but it was too strict for confident boxes after a
gap (−3% only). This motivated round 2 (section 2.6): keep BoT-SORT's strict rule for faint boxes
and loosen only its first-stage match for confident boxes.

### 3.3 Round 2 and cross-check

Full table: `notes/runs/crosscheck.md`; held-out run: `notes/runs/T8_D1_30-30s_report.md`. Fragmentation change against each window's baseline.

| Variant | Step | Tuning 0–30 s | Jumps | Guards | Held-out 30–60 s | Jumps | Guards |
|---|---|---|---|---|---|---|---|
| `baseline` | 0 | 478 | 19 | pass | 547 | 21 | pass |
| `s1_lowconf` | 1 | −17% | 28 | fail | −12% | 26 | pass |
| `s2_bt_buf4s` | 2 | −21% | 29 | fail | −14% | 27 | pass |
| `s2_ocsort_buf4s` | 2 | −46% | 44 | fail | −41% | 38 | fail |
| `s2_botsort_buf4s` | 2 | −3% | 0 | pass | −4% | 2 | pass |
| `r2_botsort_first01` | 2b | −15% | 2 | pass | −12% | 6 | pass |
| **`r2_botsort_first005`** | 2b | **−21%** | **6** | pass | **−18%** | **10** | pass |
| `r2_botsort_first01_unc02` | 2b | −15% | 2 | pass | −12% | 6 | pass |
| `r2_ocsort_iou03` | 2b | −24% | 23 | fail | −26% | 27 | fail |
| `r2_ocsort_iou03_buf2s` | 2b | −13% | 19 | fail | −14% | 22 | fail |
| `r2_bt_conf015_buf4s` | 2b | −13% | 28 | fail | −12% | 15 | pass |

On the held-out window the selection rule picks `r2_botsort_first005` (fragmentation 449 against
547, IDs 372 against 427, identity jumps 10 against 21, coverage 99.3% against 99.4%). It is the
same variant the tuning window picked. Its gain is smaller on the held-out window (−18% against
−21%), the expected optimism of the window it was designed on.

## 4. Decision

**Tracker: `r2_botsort_first005`.** BoT-SORT; boxes from confidence 0.10 reach the tracker; a
confident box (≥ 0.5) can continue a track at fused similarity (IoU × confidence) ≥ 0.05; a faint
box (0.10–0.5) can only continue a track at IoU ≥ 0.5 and never starts one; 4 s memory; 2 frames to
confirm; no camera-motion correction, no re-identification. Set in `configs/T8_D1.json`
(`tracking.variant`); `python run.py track` produces the points files with it.

Against the original tracker, on video it was not tuned on: about 18% fewer track breaks, 13% fewer
IDs and half the identity errors, with the same clear-box coverage.

Rejected, with reasons: ByteTrack with faint boxes (wrong merges through low-overlap matches),
OC-SORT (wrong re-associations after long gaps, 5% of clear boxes untracked), a longer memory alone
(no effect beyond 2 s: matching is overlap-based).

## 5. Limitations and next steps

* **Most breaks remain.** 449 mid-image starts and ends in 30 s is still far from zero; 109 of
  them look like stitchable break → restart pairs. Fully hidden vehicles (trees, trucks) cannot be
  bridged by an online tracker. Next: **step 3, offline stitching** (`stitch.py`) using the
  restart-pair logic already in `quality.py`, then **step 4** occlusion zones and **step 5**
  removal of short junk tracks (still 98 under 1 s on the held-out window).
* **The metrics are proxies.** Fragmentation counts breaks, not their effect on speed; the jump
  detector was audited on 8 events, not exhaustively. **Step 6** (after camera calibration) will
  measure IDF1 and ID switches against MiTra's ground-truth IDs.
* **Two windows from one drone in one flight**, both congested. Free-flowing traffic (flight T3) or
  another drone may favour different settings; rerun `tune` there before reusing this choice.
* OC-SORT's coverage loss was not investigated further (it was rejected on both guards).
* Round 2 was designed after seeing round 1; the held-out window, not the tuning window, carries
  the evidence for the final choice.

## 6. Step 3: offline stitching (fixed before the step-3 results)

### 6.1 Why

With the chosen tracker, about 60% of the mid-image track ends have a plausible continuation:
a later track that starts in the same lane band where the vehicle should be. By gap length
(0–30 s / 30–60 s): under 0.5 s 23 / 31 (the visible "ID changes while driving"), 0.5–2 s
38 / 47, 2–5 s 46 / 51. An online tracker cannot bridge a fully hidden vehicle; on recorded video
we can look forward and join the pieces afterwards.

### 6.2 Method

`src/lane_intel/tracking/stitch.py`, enabled per variant with a `stitch` block. A link A → B needs:
A ends and B starts mid-image; 0 < gap ≤ `max_gap_s`; B starts within `x_base_frac` × A's width
+ `x_per_s` × gap of where A's end speed predicts; the top edge of the box (not the bottom: trees
and nearer trucks hide the lower part of a car) within `y_px` + `y_per_s` × gap; width ratio ≤
`width_ratio_max`; not moving in opposite directions. Pairs are chosen together, one-to-one,
by an optimal assignment; chains (A → B → C) become one vehicle with A's ID; the tracker's own
ID is kept in `raw_track_id`. No boxes are invented for the hidden stretch.

Variants (configs/tracking.json): `s3_stitch_2s`, `s3_stitch_5s`, `s3_stitch_5s_loose`, all on top
of `r2_botsort_first005`.

### 6.3 Acceptance rule (fixed before the results)

1. **Guards relative to the incumbent** `r2_botsort_first005` (no longer the original baseline: a
   new step must not make identity errors worse than the current best). Configured as
   `tracking.reference` in the camera config. Identity jumps ≤ incumbent × 1.2 + 2; clear-box
   coverage ≥ incumbent − 1 point. The jump check also inspects every stitch junction: a join
   whose restart is more than half a box off the motion prediction (+15 px per second of gap)
   counts as an identity jump.
2. Among variants that pass, lowest fragmentation on the **held-out window (30–60 s)**.
3. **Visual audit** of the selected variant: 20 stitches drawn at random (fixed seed) from the
   held-out window, rendered before/after from the video and judged same vehicle / different
   vehicle. Accepted if at least 18 of 20 are correct (90%). Otherwise the next eligible variant
   in rank order is audited the same way; if none passes, no stitching is used.

### 6.4 First result and protocol amendment A1 (written before the re-run and the fresh audit)

**Result under the rule as written.** On both windows every stitching variant failed the
identity-jump guard (held-out: 45 / 54 / 93 jumps for 2 s / 5 s / loose, limit 14), so the rule
as written selects no stitching. Reports: `notes/runs/step3_*`.

**Audit of the guard on stitch junctions.** 20 joins of `s3_stitch_5s` on the held-out window,
drawn with seed 0, were judged at full resolution (`notes/audits/T8_D1_30-30s_s3_stitch_5s_seed0.*`):
18 the same vehicle, 2 different (a silver car joined to a red one after 4.9 s, a dark car joined
to a white van under the trees after 4.5 s). The guard flagged 13 of the 20 junctions: 12 of them
were correct joins (mostly cars reappearing from behind the gantry pole, whose box grows from the
visible half to the whole car, which the jump check reads as a jump), and it missed one of the
two wrong joins. **At stitch junctions the jump check does not measure what it is meant to
measure.** Inside tracker pieces it remains valid (8 of 8 audited events were real swaps, 3.2).

**Amendment A1.**
1. The jump guard counts only jumps inside tracker pieces (`identity_jumps_within`: both
   observations share `raw_track_id`). Jumps at stitch junctions are still reported
   (`junction_jumps`) but not used for the decision. For variants without stitching nothing
   changes, so all step 0–2 results stand.
2. Join quality is measured directly: the top-ranked eligible variant on the held-out window is
   audited on a **fresh** sample of 20 joins (seed 1; seed 0 has been seen). Accepted if at least
   18 are the same vehicle. Otherwise the next eligible variant is audited the same way (seed 1);
   if none passes, no stitching.
3. The audit tool is `lane_intel.tracking.audit`; samples, cards and verdicts are committed under
   `notes/audits/`.

Because A1 was written after seeing the step-3 guard results, the decision rests on the fresh
audit, not on the guard numbers.
