# Lanes

Code: `src/lane_intel/lanes.py` · step: `python run.py lanes [--cfg ...]` · output: `configs/lanes/<camera>.json`
(editable) and `lane_frame` / `lane_track` in the window's points file. Tests: `tests/test_lanes.py`.

## Method

Lanes are found from where tracked vehicles drive; no manual drawing.

1. **Geometry family**, chosen per camera by 2-fold cross-validation over long moving tracks (fit on half
   the vehicles, measure on the other half how far each vehicle's boxes stray from its lane line, in
   pixels; the simplest family within 2% of the best wins): straight parallel lanes, curved parallel
   lanes (one shared shape y = c + f(x), degree 2 or 3), or straight lanes meeting at a vanishing point.
2. **Lateral coordinate** c of every box: the height at the image centre of the lane line through it.
3. **Lanes per direction** = peaks of the histogram of track c values (one vote per track). Pass 1:
   moving tracks; pass 2 adds stopped tracks inside the carriageway they sit in, which recovers a lane
   where traffic stands still.
4. **Numbering**: lane 1 = leftmost in the direction of travel (left-moving traffic: the lowest lane in
   the image; right-moving: the highest). With right-hand traffic this is the lane next to the median.
5. **Assignment**: nearest lane of the vehicle's direction within the lane's band; a track's lane is its
   most frequent box lane.

Three choices were corrected after looking at the result: slicing the image into columns (failed on
diagonal lanes), counting stopped cars for both directions (phantom lanes), and comparing models on the
lateral coordinate instead of pixels (unfair to the vanishing-point model). The final method was checked
by drawing the lanes on frames of both videos.

## Results

| Camera | Geometry (cross-validated residual, px) | Lanes | Boxes inside a lane |
|---|---|---|---|
| MiTra T8 D1, 0–30 s | straight parallel ({'straight': 3.572, 'curve2': 3.634, 'curve3': 3.556}) | 4 left + 4 right | 99% |
| Uploaded slowest_car_1943pm | curved, degree 2 ({'straight': 4.014, 'curve2': 2.466, 'curve3': 2.424, 'vp': 6.059}) | 6 left | 99% |

**MiTra check against ground truth** (drone D1 CSV at flight time 5:00): the jammed direction has 4 lanes
(GT lanes 4–7) and the moving direction 3 lanes plus a ramp lane (GT 1–3 and 10): 4 + 4, as found.

## Limits and next steps

* A slip road leaving at an angle (top left of the uploaded clip) does not share the motorway's shape and
  is not found as a lane.
* Lane widths in pixels shrink with distance from the camera, so **speeds per lane are not comparable in
  px/s**. Comparing lanes (fastest / slowest) needs the calibration to metres (phase 2, `calibrate.py`).
* Lane mapping to MiTra's own lane numbers needs the same calibration (pixels to UTM).
