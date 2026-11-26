import json

import pandas as pd
import pytest

from lane_intel.export_viewer import build_bundle, track_summary

FPS = 30000 / 1001


def _points(tracks):
    """tracks: {track_id: (cls, [(frame_idx, x_px), ...])} -> points frame with 40x20 boxes."""
    rows = []
    for tid, (cls, obs) in tracks.items():
        for f, x in obs:
            rows.append(dict(t=f / FPS, frame_idx=f, track_id=tid, cls=cls, cls_id=2, conf=0.8,
                             x1=x - 20, y1=200.0, x2=x + 20, y2=220.0, x_px=x, y_px=220.0))
    return pd.DataFrame(rows)


def test_track_summary_motion_and_class():
    df = _points({
        1: ("car", [(f, 100 + 100 * f / FPS) for f in range(31)]),       # +100 px/s
        2: ("truck", [(f, 500.0) for f in range(10)]),                   # stationary
        3: ("car", [(5, 50.0)]),                                         # single observation
    })
    df.loc[(df.track_id == 1) & (df.frame_idx < 3), "cls"] = "truck"     # minority label
    s = {r["track_id"]: r for r in track_summary(df, FPS)}
    assert s[1]["px_per_s"] == pytest.approx(100, abs=0.2)
    assert s[1]["cls"] == "car"
    assert s[1]["n"] == 31 and s[1]["first"] == 0 and s[1]["last"] == 30
    assert s[2]["px_per_s"] == 0
    assert s[3]["px_per_s"] == 0 and s[3]["span_s"] == 0


def test_bundle_end_to_end(synth_video, tmp_path):
    cfg = {"flight": "TX", "drone": "D0", "video": synth_video, "native_fps": FPS,
           "clip_offset_s": 300.0, "frame_size": [640, 360], "model": "m.pt", "imgsz": 640, "conf": 0.25}
    df = _points({1: ("car", [(f, 100 + 3 * f) for f in range(64)]),
                  2: ("bus", [(f, 400.0) for f in range(20, 50)])})
    meta = build_bundle(cfg, df, tmp_path, start=0, duration=40 / FPS, crop_y=(100, 260), width=320)

    for name in ("video.mp4", "detections.json", "meta.json", "index.html"):
        assert (tmp_path / name).stat().st_size > 0
    dets = json.loads((tmp_path / "detections.json").read_text())
    assert all(0 <= d[0] < 40 for d in dets)                      # window applied
    assert len(dets) == 40 + 20
    assert meta["frame_count"] == 40 and meta["video_size"] == [320, 80]
    assert meta["classes"] == ["bus", "car"]
    assert json.loads((tmp_path / "meta.json").read_text())["crop"] == {"x": 0, "y": 100, "w": 640, "h": 160}
    assert (tmp_path / "index.html").read_text().startswith("<!doctype html>")
