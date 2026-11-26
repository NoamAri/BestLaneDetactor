import json

import pandas as pd
import pytest

from lane_intel.export_viewer import ViewerVariant, build_bundle, track_summary, wrap_page

FPS = 30000 / 1001


def _points(tracks):
    """tracks: {track_id: (cls, [(frame_idx, x_px), ...])} -> points frame with 40x20 boxes."""
    rows = []
    for tid, (cls, obs) in tracks.items():
        for f, x in obs:
            rows.append(dict(t=f / FPS, frame_idx=f, track_id=tid, cls=cls, cls_id=2, conf=0.8,
                             x1=x - 20, y1=200.0, x2=x + 20, y2=220.0, x_px=x, y_px=220.0))
    return pd.DataFrame(rows)


def test_track_summary_motion_class_and_break_flags():
    df = _points({
        1: ("car", [(f, 1000 + 100 * f / FPS) for f in range(31)]),     # +100 px/s, from window start
        2: ("truck", [(f, 500.0) for f in range(10, 20)]),             # stationary, starts and ends mid-image
        3: ("car", [(5, 50.0)]),                                       # single observation at the left edge
    })
    df.loc[(df.track_id == 1) & (df.frame_idx < 3), "cls"] = "truck"  # minority label
    s = {r["track_id"]: r for r in track_summary(df, FPS, frame_range=(0, 60), width=3840)}
    assert s[1]["px_per_s"] == pytest.approx(100, abs=0.2)
    assert s[1]["cls"] == "car"
    assert s[1]["n"] == 31 and s[1]["first"] == 0 and s[1]["last"] == 30
    assert not s[1]["born_mid"] and s[1]["died_mid"]
    assert s[2]["px_per_s"] == 0 and s[2]["born_mid"] and s[2]["died_mid"]
    assert s[3]["px_per_s"] == 0 and s[3]["span_s"] == 0 and not s[3]["born_mid"]


def test_bundle_end_to_end_with_two_variants(synth_video, tmp_path):
    cfg = {"flight": "TX", "drone": "D0", "video": synth_video, "native_fps": FPS,
           "clip_offset_s": 300.0, "frame_size": [640, 360],
           "detector": {"model": "data/models/m.pt", "imgsz": 640}}
    a = _points({1: ("car", [(f, 100 + 3 * f) for f in range(64)]),
                 2: ("bus", [(f, 400.0) for f in range(20, 50)])})
    b = _points({7: ("car", [(f, 100 + 3 * f) for f in range(64)])})
    meta = build_bundle(cfg, [ViewerVariant("base", a, label="Base", quality={"ids": 2}),
                              ViewerVariant("tuned", b, label="Tuned")],
                        tmp_path, start=0, duration=40 / FPS, crop_y=(100, 260), width=320)

    for name in ("video.mp4", "tags_base.json", "tags_tuned.json", "meta.json", "index.html"):
        assert (tmp_path / name).stat().st_size > 0
    tags = json.loads((tmp_path / "tags_base.json").read_text())
    assert all(0 <= d[0] < 40 for d in tags)                       # window applied
    assert len(tags) == 40 + 20
    assert meta["frame_count"] == 40 and meta["video_size"] == [320, 80]
    assert meta["classes"] == ["bus", "car"] and meta["model"] == "m.pt"
    assert [v["name"] for v in meta["variants"]] == ["base", "tuned"]   # first one opens by default
    assert meta["variants"][0]["quality"] == {"ids": 2}
    assert {t["track_id"] for t in meta["variants"][1]["tracks"]} == {7}
    saved = json.loads((tmp_path / "meta.json").read_text())
    assert saved["crop"] == {"x": 0, "y": 100, "w": 640, "h": 160}
    assert (tmp_path / "index.html").read_text().startswith("<!doctype html>")


def test_wrap_page_moves_title_into_head():
    page = wrap_page("<title>X</title>\n<style>a{}</style>\n<div>hi</div>")
    head, body = page.split("</head>")
    assert "<title>X</title>" in head and "<style>" in body and page.startswith("<!doctype html>")
