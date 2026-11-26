import subprocess

import pytest

FPS = "30000/1001"


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory):
    """A 64-frame 640x360 test video at 29.97 fps (same rate as MiTra)."""
    path = tmp_path_factory.mktemp("vid") / "synth.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate={FPS}",
         "-frames:v", "64", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return str(path)
