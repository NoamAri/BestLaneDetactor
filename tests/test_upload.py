"""Upload gate: only good videos are analysed, with the right reason when they are not."""
import json
import subprocess

import pytest

from lane_intel import upload as U

RULES = json.loads((U.RULES).read_text())
GOOD_PIC = {"frames": 45, "bg_inliers_min": 650, "bg_inliers_median": 700, "camera_shift_frac_max": 0.004,
            "max_step_diff": 10.0, "median_step_diff": 6.0, "brightness": 110.0, "sharpness": 400.0}
GOOD_VEH = {"vehicles_per_frame": 30, "vehicle_length_px": 60.0, "band_y": [100, 600]}


def info(**kw):
    return {"ok": True, "duration_s": 45.0, "width": 1920, "height": 1080, "fps": 30.0, "size_mb": 6.0, **kw}


def rule_of(fn, *a):
    with pytest.raises(U.NotGood) as e:
        fn(*a)
    return e.value.rule


# ---------------------------------------------------------------- stage A
def test_good_file_passes():
    U.check_file("clip.mp4", info(), RULES)


@pytest.mark.parametrize("kw,rule", [
    ({"duration_s": 20.0}, "A4"), ({"duration_s": 30.0}, "A4"), ({"duration_s": 60.0}, "A4"), ({"duration_s": 75.0}, "A4"),
    ({"size_mb": 14.0}, "A3"), ({"width": 640, "height": 360}, "A5"), ({"fps": 2.0}, "A5"), ({"ok": False}, "A5"),
])
def test_file_rules(kw, rule):
    assert rule_of(U.check_file, "clip.mp4", info(**kw), RULES) == rule


def test_not_a_video_extension():
    assert rule_of(U.check_file, "notes.pdf", info(), RULES) == "A2"


def test_duration_boundaries_are_strict():
    U.check_file("clip.mov", info(duration_s=30.5), RULES)
    U.check_file("clip.mov", info(duration_s=59.5), RULES)


# ---------------------------------------------------------------- stage B
def test_good_picture_passes():
    U.check_picture(GOOD_PIC, GOOD_VEH, RULES)


@pytest.mark.parametrize("kw,rule", [
    ({"max_step_diff": 55.0, "median_step_diff": 5.0}, "B2"),               # one sudden jump = a cut
    ({"bg_inliers_min": 9, "camera_shift_frac_max": 4.9, "max_step_diff": 36.0, "median_step_diff": 32.0}, "B1"),  # a pan
    ({"camera_shift_frac_max": 0.2}, "B1"),
    ({"brightness": 10.0}, "B3"), ({"brightness": 240.0}, "B3"), ({"sharpness": 3.0}, "B3"),
])
def test_picture_rules(kw, rule):
    assert rule_of(U.check_picture, {**GOOD_PIC, **kw}, GOOD_VEH, RULES) == rule


def test_too_few_or_too_small_vehicles():
    assert rule_of(U.check_picture, GOOD_PIC, {**GOOD_VEH, "vehicles_per_frame": 3}, RULES) == "B4"
    assert rule_of(U.check_picture, GOOD_PIC, {**GOOD_VEH, "vehicle_length_px": 9.0}, RULES) == "B4"


def test_probe_and_stats_on_a_real_still_video(tmp_path):
    """A still, textured scene: a steady camera with no cuts, measured from real frames."""
    v = tmp_path / "still.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=10", "-t", "6",
                    "-vf", "geq=lum='lum(X,Y)':cb=128:cr=128,boxblur=1", "-pix_fmt", "yuv420p", str(v)], check=True)
    i = U.probe(v)
    assert i["ok"] and i["width"] == 1280 and abs(i["duration_s"] - 6) < 0.2
    rules = {**RULES, "picture": {**RULES["picture"], "sample_fps": 2.0}}
    ps = U.picture_stats(v, rules)
    assert ps["frames"] >= 10 and ps["camera_shift_frac_max"] < 0.05


# ---------------------------------------------------------------- config, answer, site list
def test_make_config_is_self_contained():
    cfg = U.make_config("issue-7", "My road", U.ROOT / "data/raw/uploads/issue-7.mp4", info(width=1280, height=720, fps=10.0),
                        [120, 560], RULES)
    assert cfg["native_fps"] == 10.0 and cfg["band_y"] == [120, 560] and cfg["lane_speed"]["windows"] == [[0, 45.0]]
    assert cfg["calibration"]["method"] == "vehicle_size" and cfg["tracking"]["variant"] == RULES["analysis"]["tracker_variant"]
    assert cfg["label"] == "Uploaded · My road" and cfg["lanes_file"].startswith("configs/uploads/")


def test_answer_lines_say_close_call():
    summ = {"whole_span": [{"lane": "L1", "number": 1, "v_kmh": 52.6}, {"lane": "L6", "number": 6, "v_kmh": 4.5}],
            "whole_span_announcement": [{"direction": "left", "fastest": None, "fastest_candidate": "L1", "slowest": "L6"}]}
    assert U.answer_lines(summ) == ["← Left-moving traffic: fastest lane 1 (53 km/h) (close call), slowest lane 6 (4 km/h)"]


def test_uploaded_clips_follow_the_built_in_ones_and_only_the_newest_are_kept(tmp_path):
    (tmp_path / "clips.json").write_text(json.dumps([{"id": "mitra", "label": "MiTra", "dir": "clips/mitra/"}]))
    reg = [{"id": f"issue-{i}", "label": f"Uploaded · {i}", "dir": f"clips/issue-{i}/", "created_utc": ""} for i in (3, 2)]
    (tmp_path / "uploads.json").write_text(json.dumps(reg))
    U.merge_clips_json(tmp_path)
    ids = [c["id"] for c in json.loads((tmp_path / "clips.json").read_text())]
    assert ids == ["mitra", "issue-3", "issue-2"]
    U.merge_clips_json(tmp_path)                                    # idempotent
    assert [c["id"] for c in json.loads((tmp_path / "clips.json").read_text())] == ids


def test_slug():
    assert U.slug("Ayalon North, from the bridge!") == "ayalon-north-from-the-bridge"
    assert U.slug("???") == "video"


def test_names_shown_in_the_app_are_plain_text():
    assert U.clean_label('<img src=x onerror="alert(1)">Ayalon') == "img src=x onerror=alert(1)Ayalon"
    assert "<" not in U.clean_label("<script>bad()</script>") and U.clean_label("   ") == "video"
    assert len(U.clean_label("x" * 200)) == 60
