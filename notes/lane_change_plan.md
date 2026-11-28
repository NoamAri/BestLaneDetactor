# Plan: lane changes (track them, show them, count them correctly)

Status: LC1, LC2, LC5 built; LC3 (T1) tested and rejected; LC4 (ramp merges) not started. Results in section 7.

## 1. What happens today when a vehicle changes lane

* **Lane per moment, not per car.** Every box of a vehicle gets the lane it is in at that moment
  (`lane_frame`, in the stabilised frame).
* **Flicker is ignored (hysteresis).** A new lane counts only after the vehicle stays in it for at least 0.5 s
  (`pieces.min_dwell_s`), so wobbling on a lane line is not a change.
* **The change is placed where the vehicle crosses the line** between the two lanes (the midpoint of the two lane
  centres, interpolated between the boxes either side), not when the label flips (`lane_pieces.move_to_crossings`).
* **Speeds: each lane gets exactly what was driven in it.** The track is split into lane pieces; each piece's distance and
  time go to its own lane in the Edie sums (`lane_speed.edie_vehicles`). A car that drives 40 m in lane 2 and 60 m in lane 1
  adds 40 m (and that time) to lane 2 and 60 m to lane 1. This is why lane changes cannot distort the fastest lane, and it
  is covered by a synthetic test (`tests/test_lane_speed.py::test_lane_change_is_split_where_the_vehicle_crosses`).
* **Events are recorded** (`data/out/lane_speed/<camera>/lane_changes.parquet`: vehicle, time, from, to) and counted per
  lane (`changes_in`, `changes_out`).
* **Not shown in the app.** The viewers give each vehicle a single lane (the one it spent most time in), so a lane change
  is invisible in the car card, the Cars list and on the video.

## 2. Check against MiTra ground truth (0–180 s)

The ground truth's own lane labels went through exactly the same hysteresis and crossing logic.

| | Ours | Ground truth |
|---|---|---|
| All lane changes | 76 | 90 |
| R3 → R2 | 19 | 25 |
| R2 → R1 | 5 | 8 |
| L3 → L4 | 10 | 11 |
| R4 ↔ R3 (on-ramp merges) | 11 | 37 |
| L3 → L2, L1 ↔ L2 (jammed lanes) | 14 | 1 |

* Main moving lanes: about three quarters of real changes found, few false ones.
* **Missed:** merges from the on-ramp (R4). Vehicles there are seen briefly and their tracks often break during the merge.
* **False:** in the jam (L1–L3). A standing truck's box changes shape and slides across the line although the truck does not
  move sideways. 40% of L2's vehicle time is trucks.
* Effect on lane speeds: small. False changes move seconds between two lanes that both crawl at 1–3 km/h; the per-lane
  speed errors stay at about 2–3% (notes/lane_speed.md section 4).

## 3. Better tracking of lane changes

| # | Change | Why | Check |
|---|---|---|---|
| T1 | **A change needs real sideways movement**: the vehicle's lateral position (metres across the road, smoothed over 1 s) must move by at least 60% of a lane width within 4 s around the crossing | removes the jam's false changes (box shape changes without movement) | false changes in L1–L3 fall to ≤ 2; real changes kept |
| T2 | **Lateral position from a robust point**: median over the box's bottom edge, and for trucks and buses the side of the box nearest the camera | long boxes tilt and grow when partly hidden | the same |
| T3 | **Merges at track breaks**: when a track ends at the ramp's merge zone and a new one starts in the next lane within 1.5 s, close by and at a similar speed, count one merge (the identities stay separate) | most ramp merges are lost at a track break | R4 → R3 found ≥ 70% |
| T4 | **Rules fixed before testing** (as for the lane speeds): T1–T3 thresholds go into `configs/lane_speed.json` first; developed on 0–120 s, confirmed on held-out 120–180 s | no tuning on the answer | precision ≥ 0.8, recall ≥ 0.7, per direction, events matched to ground truth by vehicle and time (± 1 s) |
| T5 | Speed answer re-validated after T1–T3 | lane pieces change slightly | per-lane speed error and announcement agreement no worse than today |

## 4. Showing lane changes in the app

**Data in the viewer bundle (per vehicle):** its lane pieces `[{from_frame, to_frame, lane}]` and change events
`[{frame, from, to}]` for the clip, computed by the same code as the measurement.

**On the video**
* The followed car's speed bubble takes the colour of the lane it is in **now** (today: the lane it spent most time in).
* At a change, a short label on the bubble for 2 s: **"↗ lane 1"** (arrow toward the side it moved), with a light haptic tick.
* The lane signs at the edge count up a small **"+1"** on the lane the car moved into (fades after 2 s).

**Car card**
* A **lane strip** under the speed graph: one coloured segment per lane piece over the car's time on screen, aligned with the
  graph, with a marker at each change: "lane 2 → 1 at 12.4 s". The playhead dot moves over both.
* The lane line reads "Lane 2 → 1" when the car changed, and **"Did it pay off?"**: the car's average speed in the 5 s before
  and after the change ("+6 km/h after moving to lane 1"), only when both windows have enough data.

**Cars tab**
* Rows of vehicles that changed lanes show the sequence, "2 → 1", and a small **"changed lanes"** badge.
* A filter chip **"Lane changers"** (with count).
* Overview tile: **lane changes** (count), tap → filters the list.

**Answer tab / lane card** (when lane speeds are shown)
* Per lane: **"+11 in · −3 out"** (vehicles that moved into / out of the lane).
* One sentence per direction: **"Most common move: lane 3 → 2 (19 times)"**, and when there is enough data, whether
  movers gained speed on average ("Drivers who moved to lane 1 went 4 km/h faster afterwards", with the number of cases).
  This is a check as well: lanes that are really faster should attract changes into them.

## 5. Making sure the fastest lane is counted correctly

Already true by construction (section 1). The plan keeps it true and proves it:
1. Keep the split at the crossing; T1–T3 only change *where/whether* a change is placed, never how distance and time are
   summed.
2. New synthetic tests: a car that changes lanes twice (2 → 1 → 2) contributes three pieces; a truck box that slides across
   the line without lateral movement does **not** create a change (T1); a ramp merge at a track break is counted once and
   adds no distance twice (T3).
3. After T1–T3, rerun the MiTra validation (all windows and held-out): per-lane speed error, announcement agreement and
   precision must be no worse than today (notes/lane_speed.md section 4); report the lane-change precision / recall next to them.
4. In the app, the fastest-lane answer and the per-lane numbers already include every lane piece; the new lane-change
   display reads the same pieces, so what is shown and what is counted cannot disagree.

## 6. Order of work

| Phase | Deliverable | Done when |
|---|---|---|
| LC1 | Per-vehicle lane pieces and changes in the viewer bundle; bubble colour = current lane; car card lane strip and "lane 2 → 1" | phone test: a known MiTra changer shows the right sequence and time |
| LC2 | Cars tab: "changed lanes" badge, "Lane changers" filter and tile; per-lane "+in / −out" | counts equal `lane_changes.parquet` |
| LC3 | T1 + T2 (rules first), synthetic tests, validation on 0–120 s and held-out 120–180 s | targets in T4 met, T5 no worse |
| LC4 | T3 ramp merges | R4 → R3 recall ≥ 70%, T5 no worse |
| LC5 | "Did it pay off?" on the car card and the Answer tab sentence | shown only with enough data (≥ 2 s before and after, ≥ 5 cases per sentence) |

## 7. Results (28 Nov 2026)

### LC3 · T1 "real sideways movement" (amendment A7): rejected

Rule set in `configs/lane_speed.json` before measuring (`lane_change`): a change counts only if the median lateral
position over 2 s before vs 2 s after the crossing differs by ≥ 60% of the lane spacing. Development window 0–120 s,
matched to ground truth by from/to lane, ±1 s, ±15 m (`validate.lane_change_check`):

| Development 0–120 s | → right: ours / truth / matched | precision | recall | ← left: ours / truth / matched |
|---|---|---|---|---|
| Labels only (today) | 13 / 20 / 10 | 0.77 | 0.50 | 6 / 4 / 1 |
| A7 as registered | 3 / 20 / 1 | 0.33 | 0.05 | 5 / 4 / 0 |
| A7 redesign: 1.5 s skipped either side, ≥ 0.5 | 8 / 20 / 5 | 0.63 | 0.25 | 6 / 4 / 1 |
| A7 redesign: 1.5 s skipped either side, ≥ 0.35 | 9 / 20 / 6 | 0.67 | 0.30 | 6 / 4 / 1 |

The rule removes real changes in flowing traffic and does not remove the jam's false ones (they come with real sideways
drift of the box, not only shape changes). It is disabled (`lane_change.enabled: false`); the code stays, with a test.
The held-out window 120–180 s was not used for this decision. Lane speeds do not depend on it (2.50% vs 2.58% median
error at 60 s windows).

### Labels-only lane changes, whole 0–180 s (validation.json, `lane_changes.all`)

| | Ours | Truth | Matched | Precision | Recall |
|---|---|---|---|---|---|
| → right (flowing) | 21 | 32 | 15 | 0.71 | 0.47 |
| ← left (jam) | 11 | 11 | 1 | 0.09 | 0.09 |

Targets (precision ≥ 0.8, recall ≥ 0.7) are **not met**. Lane speeds with the owner's 25 s windows: median error 2.3%,
announcement precision 18/18 fastest and 18/18 slowest; held-out 120–180 s: 4.2%, rule agreement 100%.

### Built: LC1, LC2, LC5 (phone app)

* Viewer bundle: per track `lp` (lane pieces) and `lc` (changes with km/h before and after), from the same lane pieces
  the measurement sums (`lane_speed.track_moves` → `lane_pieces.parquet`, `lane_changes.parquet`;
  `export_viewer.add_lane_moves`). What is shown and what is counted cannot disagree.
* Video: the followed car's box and bubble take its current lane's colour; "↑ lane 4" for 2 s after a change, with a haptic tick.
* Car card: lane strip aligned with the speed graph, each change with its time, **"Did it pay off?"** (average speed over
  5 s before vs after; only with ≥ 2 s seen on each side).
* Cars tab: "Lane changes" tile, "Lane changers" filter, the lanes in driving order in each row ("3 → 2"), a "changed lane" badge.
* Answer tab (when lane speeds are shown): per direction the number of changes, the most common move and, with ≥ 5
  cases, the median speed gain of drivers who moved into a lane, plus the ground-truth check. Where the check shows
  precision < 0.5 (the MiTra jam) the app says the changes are unreliable instead. Lane card: "moved into / out of this lane".
* Not done: the "+1" on the lane signs; LC4 ramp merges at track breaks.
