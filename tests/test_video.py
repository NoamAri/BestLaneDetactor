from lane_intel.video import iter_frames


def test_reads_every_frame_with_band_crop(synth_video):
    frames = list(iter_frames(synth_video, crop_y=(100, 228), width=640))
    assert len(frames) == 64
    assert [i for i, _ in frames] == list(range(64))
    assert frames[0][1].shape == (128, 640, 3)


def test_start_and_duration_select_a_window(synth_video):
    fps = 30000 / 1001
    frames = list(iter_frames(synth_video, (0, 360), 640, start_s=10 / fps, duration_s=20 / fps))
    assert len(frames) == 20
