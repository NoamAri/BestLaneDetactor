"""Frame reading through ffmpeg (honours MP4 edit lists, crops before piping)."""
from __future__ import annotations

import subprocess
from collections.abc import Iterator

import numpy as np


def iter_frames(path: str, crop_y: tuple[int, int], width: int,
                start_s: float = 0.0, duration_s: float | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, BGR crop) for every decoded frame.

    frame_idx counts from the start of the file (not from start_s), so it can
    be turned into a timestamp with frame_idx / fps.
    """
    y0, y1 = crop_y
    h = y1 - y0
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_s:
        cmd += ["-ss", f"{start_s:.6f}"]
    cmd += ["-i", path]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.6f}"]
    cmd += ["-vf", f"crop={width}:{h}:0:{y0}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * h * 3 * 4)
    nbytes = width * h * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(h, width, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()
