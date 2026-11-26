# Tracking: keeping one ID per vehicle

Status: steps 0–2 of the tracking plan. Steps 3–6 (stitching, occlusion zones, junk removal,
ground-truth evaluation) come next.
Code: `src/lane_intel/detect.py`, `src/lane_intel/tracking/`. Variants: `configs/tracking.json`.

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
* **Held-out window**: clip 30–60 s (frames 899–1798), detected separately, never looked at
  while designing variants.
* **Final choice** = the selection rule of 2.4 applied to the held-out window, over all variants.
  `lane_intel.tracking.crosscheck` reports both windows side by side.

## 3. Results

_Filled in from `data/out/tracking/T8_D1_0-30s/report.md` after the run._

## 4. Decision

_Filled in after the run._

## 5. Limitations and next steps

_Filled in after the run._
