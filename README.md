# BestLaneDetactor (Lane Intelligence)

Per-lane traffic measurements from fixed aerial video of a congested motorway.
Data: [MiTra](https://opara.zih.tu-dresden.de/items/68482417-ddc5-40ac-885a-83124bd459e8), A50 Milan, CC BY 4.0.

## Quick start (Windows, macOS, Linux)

Needs Python 3.11+, git and ffmpeg (Windows: `winget install Gyan.FFmpeg`, then open a new terminal).

```bash
git clone -b claude/happy-faraday-r67vcd https://github.com/NoamAri/BestLaneDetactor.git
cd BestLaneDetactor
python run.py setup      # .venv + all packages + yolo26s.pt (CUDA build of torch if an NVIDIA GPU is found)
python run.py clip       # 3-min 4K clip of drone D1, read straight out of the remote 70 GB zip (~2.2 GB)
python run.py detect     # YOLO once on the first 30 s (minutes on a GPU, ~12 min on CPU)
python run.py tune       # every tracker variant on the saved detections + comparison report (seconds)
python run.py track      # final tracks with the configured tracker
python run.py lanes      # find the lanes from the tracks; lane per vehicle
python run.py viewer     # build the viewer for every clip in configs/viewer.json
python run.py serve      # opens http://localhost:8000
```

`python run.py all` runs these in order. Other windows: `--start 30 --dur 60` on each step.
On Linux/macOS the same steps exist as `make` targets (`make detect tune track viewer serve`,
variables `START`, `DUR`, `CLIP_SS`).

## Add another video

The viewer offers every clip listed in `configs/viewer.json` (a Video selector appears with two or more).

1. Put the file under `data/raw/` and write a camera config next to `configs/T8_D1.json`
   (example: `configs/slowest_car_1943pm.json`): `video`, `native_fps`, `frame_size`, `band_y`
   (rows with road), `detector.imgsz` (larger than the width for small vehicles), `label`, `viewer.crop_y`/`width`.
2. `python run.py detect --cfg configs/<camera>.json --start 0 --dur <seconds>`
3. `python run.py tune --cfg configs/<camera>.json --start 0 --dur <seconds>`, then `track` and `lanes` with the same options
4. Add the clip to `configs/viewer.json`, then `python run.py viewer` (or `--only <id>`) and `python run.py pages`.

## Pipeline

| Stage | Module | Output |
|---|---|---|
| Frame reading | `lane_intel.video` | ffmpeg decode, road-band crop (honours the MP4 edit list) |
| 1 · Detection | `lane_intel.detect` | `<tag>_detections.parquet`: every box with conf ≥ 0.05 + provenance `.meta.json` |
| 2 · Tracking | `lane_intel.tracking` | `<tag>_points.parquet`: `t, frame_idx, track_id, cls, conf, x1..y2, x_px, y_px` |
| Tracker experiment | `lane_intel.tracking.experiment` | `data/out/tracking/<tag>/report.md`, per-variant points and diagnostics |
| Lanes | `lane_intel.lanes` | `configs/lanes/<camera>.json` (lane lines, lane 1 = leftmost in travel direction); `lane_frame`, `lane_track` in the points file |
| Viewer bundle (one clip) | `lane_intel.export_viewer` | `video.mp4`, `tags_<variant>.json`, `meta.json` |
| Viewer site (all clips) | `lane_intel.viewer_site` | `index.html`, `clips.json`, `clips/<id>/…` from `configs/viewer.json` |
| Viewer | `app/viewer/index.html` | video selector, frame stepping, tracks with speed and direction, quality panel, register |
| Local server | `lane_intel.serve` | static files with HTTP Range (needed for video seeking) |

Detection runs once; tracking reads the saved detections, so tracker settings are compared on
identical input in seconds. Method, metrics, selection rule and results:
[notes/tracking.md](notes/tracking.md). File-by-file explanation: `app/architecture/index.html`
(also `docs/architecture.html`).

## Data notes

- Drone videos: 3840x2160, H.264, 29.97 fps, 991 s. Ground-truth CSV time 0 = video start.
- Ground truth (`T8_D#.csv`): UTM 32N metres, speed in km/h, lane numbers 1-7 plus 10.
- `data/` and `*.pt` are not in git; `docs/` holds the small GitHub Pages copy of the viewer.
