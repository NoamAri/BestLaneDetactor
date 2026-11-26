"""Multi-clip viewer site: index, per-clip folders, published file list, per-camera thresholds."""
import json

import pandas as pd
import pytest

from lane_intel.tracking.quality import QualityParams
from lane_intel.viewer_site import build_site, site_files

FPS = 30000 / 1001


def _points(n_frames, x0):
    return pd.DataFrame([dict(t=f / FPS, frame_idx=f, track_id=1, cls="car", cls_id=2, conf=0.8,
                              x1=x0 + 2 * f - 20, y1=200.0, x2=x0 + 2 * f + 20, y2=220.0,
                              x_px=x0 + 2 * f, y_px=220.0) for f in range(n_frames)])


def test_two_clips_build_an_index_and_their_own_folders(synth_video, tmp_path):
    clips = []
    for cid, label in (("a", "Clip A"), ("b", None)):
        cfg = {"label": f"Camera {cid}", "time_label": "Video time", "video": synth_video, "native_fps": FPS,
               "clip_offset_s": 0.0, "frame_size": [640, 360], "detector": {"model": "m.pt", "imgsz": 640},
               "viewer": {"crop_y": [100, 260], "width": 320}}
        (tmp_path / f"{cid}.json").write_text(json.dumps(cfg))
        _points(64, 100).to_parquet(tmp_path / f"{cid}.parquet")
        clip = {"id": cid, "cfg": str(tmp_path / f"{cid}.json"), "start": 0, "dur": 40 / FPS,
                "points": str(tmp_path / f"{cid}.parquet")}
        if label:
            clip["label"] = label
        clips.append(clip)
    out = tmp_path / "site"
    (out).mkdir()
    (out / "meta.json").write_text("{}")                    # leftover of the old single-clip layout
    index = build_site({"clips": clips}, out)

    assert [c["id"] for c in index] == ["a", "b"]
    assert index[0]["label"] == "Clip A" and index[1]["label"] == "Camera b"   # spec label, else config label
    assert json.loads((out / "clips.json").read_text()) == index
    meta = json.loads((out / "clips/a/meta.json").read_text())
    assert meta["label"] == "Camera a" and meta["time_label"] == "Video time"
    assert meta["video_size"] == [320, 80] and meta["px_scale"] == pytest.approx(640 / 3840)
    assert not (out / "meta.json").exists()
    files = site_files(out)
    assert files[:2] == ["index.html", "clips.json"]
    assert "clips/b/video.mp4" in files and "clips/b/tags_points.json" in files
    assert all((out / f).exists() for f in files)


def test_quality_thresholds_scale_with_image_width():
    full, small = QualityParams(), QualityParams.scaled(960)
    assert small.edge_px == pytest.approx(full.edge_px / 4)
    assert small.restart_dy_px == pytest.approx(full.restart_dy_px / 4)
    assert small.jump_frac == full.jump_frac and small.restart_max_gap_s == full.restart_max_gap_s
    assert QualityParams.scaled(3840) == full
