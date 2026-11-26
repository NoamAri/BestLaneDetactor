# BestLaneDetactor (Lane Intelligence)

Per-lane traffic measurements from fixed aerial video of a congested motorway.
Data: [MiTra](https://opara.zih.tu-dresden.de/items/68482417-ddc5-40ac-885a-83124bd459e8), A50 Milan, CC BY 4.0.

## Quick start

Everything runs inside a project venv (`.venv/`) through `make`:

```bash
make setup          # create .venv, install the package + deps, fetch yolo26s.pt
make clip           # cut a 3-min 4K clip of drone D1 straight from the remote 70 GB zip (~2.2 GB)
make extract        # detect + track -> data/interim/T8_D1_0-30s_points.parquet
make viewer         # bundle video + tags for the browser -> data/out/viewer/
make serve          # open http://localhost:8000
make test           # pytest
```

Window and clip are variables, e.g. `make extract viewer START=30 DUR=60` or `make clip CLIP_SS=420`.

## Pipeline so far

| Stage | Module | Output |
|---|---|---|
| Frame reading | `lane_intel.video` | ffmpeg decode, road-band crop (honours the MP4 edit list) |
| Detect + track | `lane_intel.extract` | `points.parquet`: `t, frame_idx, track_id, cls, conf, x1..y2, x_px, y_px` (full-frame px) |
| Viewer bundle | `lane_intel.export_viewer` | `video.mp4` (road-strip proxy), `detections.json`, `meta.json`, `index.html` |
| Viewer | `app/viewer/index.html` | frame stepping, boxes/IDs/trails, zoom, track inspector, track register |

Model choice: `yolo26s.pt` on the road strip at native width (`imgsz=3840`) finds ~99 of the
vehicles in view; the same model on the whole frame downscaled to 1280 px finds ~27.

## Data notes

- Drone videos: 3840x2160, H.264, 29.97 fps, 991 s. Ground-truth CSV time 0 = video start.
- Ground truth (`T8_D#.csv`): UTM 32N metres, speed in km/h, lane numbers 1-7 plus 10.
- `data/` and `*.pt` are not in git.
