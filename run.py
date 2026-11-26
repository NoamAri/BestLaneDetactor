"""One entry point for every step, on Windows, macOS and Linux (no `make` needed).

    python run.py setup      create .venv, install everything, download yolo26s.pt
    python run.py clip       cut a 3-min 4K clip of drone D1 from the remote MiTra zip (~2.2 GB)
    python run.py detect     run YOLO once on a window -> data/interim/<tag>_detections.parquet (slow)
    python run.py tune       try every tracker variant on those detections, pick one by the
                             fixed rule -> data/out/tracking/<tag>/report.md (seconds)
    python run.py track      track with the configured variant -> data/interim/<tag>_points.parquet
    python run.py extract    detect + track in one go
    python run.py viewer     bundle video + tags for the browser -> data/out/viewer/
    python run.py serve      start the viewer at http://localhost:8000 and open the browser
    python run.py pages      copy the viewer and the architecture page into docs/ for GitHub Pages
    python run.py test       run the tests
    python run.py all        setup, clip, detect, tune, track, viewer, serve

Options: --start 0 --dur 30 (window, seconds into the clip), --variant NAME (track step),
         --clip-ss 300 --clip-len 180 (where to cut the clip from the full flight video).
Only the Python standard library is used here, so it runs before the venv exists.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
CFG = ROOT / "configs" / "T8_D1.json"
MODEL = ROOT / "data" / "models" / "yolo26s.pt"
VIEWER = ROOT / "data" / "out" / "viewer"

# MiTra flight T8 drone videos, stored uncompressed inside one remote zip; D1 = bytes [59, 12377194959)
T8_VIDEO_ZIP = ("https://opara.zih.tu-dresden.de/server/api/core/bitstreams/"
                "052086e7-e2ec-4d72-a085-330f540b1537/content")
D1_START, D1_END = 59, 12377194959
SANDBOX_CA = Path("/root/.ccr/ca-bundle.crt")  # only inside the Claude cloud sandbox


def sh(cmd: list, **kw) -> None:
    print("\n> " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def need_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    hint = ("winget install Gyan.FFmpeg   (then open a NEW terminal)" if os.name == "nt"
            else "brew install ffmpeg  /  sudo apt install ffmpeg")
    sys.exit(f"ffmpeg is not on PATH. Install it:\n    {hint}")


def need_venv() -> None:
    if not PY.exists():
        sys.exit("No .venv yet. Run:  python run.py setup")


def clip_path(ss: float) -> Path:
    return ROOT / "data" / "raw" / f"T8_D1_dev_{int(ss)}s.mp4"


# ---------------------------------------------------------------- steps
def setup(a) -> None:
    if sys.version_info < (3, 11):
        sys.exit(f"Python 3.11+ is needed (this is {sys.version.split()[0]}).")
    need_ffmpeg()
    if not PY.exists():
        print(f"Creating virtual environment in {VENV}")
        venv.EnvBuilder(with_pip=True).create(VENV)
    sh([PY, "-m", "pip", "install", "--upgrade", "pip"])
    if shutil.which("nvidia-smi"):
        # PyPI's Windows torch wheel is CPU-only; take the CUDA build when an NVIDIA GPU is present.
        sh([PY, "-m", "pip", "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu128"])
    sh([PY, "-m", "pip", "install", "-e", ".[dev]"])
    if not MODEL.exists():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        print("\nDownloading yolo26s.pt")
        subprocess.run([str(PY), "-c", "from ultralytics import YOLO; YOLO('yolo26s.pt')"],
                       check=True, cwd=MODEL.parent)
    sh([PY, "-c", "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"])
    print("\nSetup done.")


def clip(a) -> None:
    need_ffmpeg()
    out = clip_path(a.clip_ss)
    if out.exists():
        print(f"Clip already there: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    extra = []
    if SANDBOX_CA.exists():
        env["http_proxy"] = os.environ.get("https_proxy", "")
        extra = ["-ca_file", str(SANDBOX_CA)]
    tmp = out.with_suffix(".part.mp4")
    print(f"Cutting {a.clip_len:.0f} s from {a.clip_ss:.0f} s of T8_D1.mp4 (about 2.2 GB, reads only that part)...")
    sh(["ffmpeg", "-v", "error", "-stats", "-y",
        "-protocol_whitelist", "subfile,https,http,httpproxy,tls,tcp,file,crypto", *extra,
        "-ss", str(a.clip_ss), "-i", f"subfile,,start,{D1_START},end,{D1_END},,:{T8_VIDEO_ZIP}",
        "-t", str(a.clip_len), "-c", "copy", tmp], env=env)
    tmp.replace(out)
    print(f"Clip saved: {out}")


def _tag(a) -> str:
    return f"T8_D1_{a.start:g}-{a.dur:g}s"


def _dets(a) -> Path:
    return ROOT / "data" / "interim" / f"{_tag(a)}_detections.parquet"


def _points(a) -> Path:
    return ROOT / "data" / "interim" / f"{_tag(a)}_points.parquet"


def _experiment(a) -> Path:
    return ROOT / "data" / "out" / "tracking" / _tag(a)


def _cfg_for_clip(a) -> Path:
    """configs/T8_D1.json points at the default clip; write a variant if another clip was cut."""
    import json
    cfg = json.loads(CFG.read_text())
    want = str(clip_path(a.clip_ss).relative_to(ROOT)).replace("\\", "/")
    if cfg["video"] == want:
        return CFG
    cfg["video"], cfg["clip_offset_s"] = want, float(a.clip_ss)
    alt = ROOT / "data" / "interim" / f"cfg_T8_D1_{int(a.clip_ss)}s.json"
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_text(json.dumps(cfg, indent=2))
    return alt


def detect(a) -> None:
    need_venv(); need_ffmpeg()
    if not clip_path(a.clip_ss).exists():
        sys.exit("No clip yet. Run:  python run.py clip")
    out = _dets(a)
    if out.exists() and not a.force:
        print(f"Already detected: {out}  (use --force to redo)")
        return
    sh([PY, "-W", "ignore", "-m", "lane_intel.detect", "--cfg", _cfg_for_clip(a),
        "--start", a.start, "--duration", a.dur, "--out", out])


def _need_dets(a) -> None:
    if not _dets(a).exists():
        sys.exit("No detections for this window yet. Run:  python run.py detect")


def tune(a) -> None:
    need_venv(); _need_dets(a)
    sh([PY, "-W", "ignore", "-m", "lane_intel.tracking.experiment", "--cfg", _cfg_for_clip(a),
        "--dets", _dets(a), "--out-dir", _experiment(a)])


def track(a) -> None:
    need_venv(); _need_dets(a)
    cmd = [PY, "-W", "ignore", "-m", "lane_intel.tracking", "--cfg", _cfg_for_clip(a),
           "--dets", _dets(a), "--out", _points(a)]
    if a.variant:
        cmd += ["--variant", a.variant]
    sh(cmd)


def extract(a) -> None:
    detect(a)
    track(a)


def viewer(a) -> None:
    """Bundle the viewer: every tracker in tracking.viewer_variants if 'tune' ran, else the tracked points."""
    import json
    need_venv(); need_ffmpeg()
    cfg_path = _cfg_for_clip(a)
    common = ["--cfg", cfg_path, "--start", a.start, "--duration", a.dur, "--out-dir", VIEWER]
    if (_experiment(a) / "report.json").exists():
        names = json.loads(Path(cfg_path).read_text())["tracking"].get("viewer_variants", ["@selected"])
        sh([PY, "-W", "ignore", "-m", "lane_intel.export_viewer", "--experiment", _experiment(a),
            "--variants", ",".join(names), *common])
    elif _points(a).exists():
        sh([PY, "-W", "ignore", "-m", "lane_intel.export_viewer", "--points", _points(a), *common])
    else:
        sys.exit("Nothing to show yet. Run:  python run.py tune   (or: python run.py track)")


def serve(a) -> None:
    need_venv()
    if not (VIEWER / "index.html").exists():
        sys.exit("No viewer bundle yet. Run:  python run.py viewer")
    url = f"http://localhost:{a.port}"
    threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    print(f"\nViewer at {url}   (Ctrl+C to stop)")
    try:
        subprocess.run([str(PY), "-m", "lane_intel.serve", str(VIEWER), "--port", str(a.port)], cwd=ROOT)
    except KeyboardInterrupt:
        pass


def _wrap_page(fragment: str) -> str:
    """Same as lane_intel.export_viewer.wrap_page (duplicated: run.py must not need the venv)."""
    head, sep, rest = fragment.partition("<style>")
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head.strip() + "\n</head><body>\n" + sep + rest + "\n</body></html>\n")


def pages(a) -> None:
    """Mirror the viewer bundle and the architecture page into docs/ for GitHub Pages."""
    if not (VIEWER / "index.html").exists():
        sys.exit("No viewer bundle yet. Run:  python run.py viewer")
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    for old in [*docs.glob("tags_*.json"), docs / "detections.json"]:
        old.unlink(missing_ok=True)                      # files from an older bundle layout
    for f in VIEWER.iterdir():
        if f.is_file():
            shutil.copy2(f, docs / f.name)
    (docs / ".nojekyll").write_text("")
    arch = ROOT / "app" / "architecture" / "index.html"
    if arch.exists():
        (docs / "architecture.html").write_text(_wrap_page(arch.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"GitHub Pages site written to {docs}. Commit and push it to update the site.")


def test(a) -> None:
    need_venv(); need_ffmpeg()
    sh([PY, "-m", "pytest", "-q"])


def all_steps(a) -> None:
    for step in (setup, clip, detect, tune, track, viewer, serve):
        step(a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = {"setup": setup, "clip": clip, "detect": detect, "tune": tune, "track": track, "extract": extract,
             "viewer": viewer, "serve": serve, "pages": pages, "test": test, "all": all_steps}
    ap.add_argument("step", choices=list(steps))
    ap.add_argument("--start", type=float, default=0.0, help="detection window start, s into the clip")
    ap.add_argument("--dur", type=float, default=30.0, help="detection window length, s")
    ap.add_argument("--clip-ss", type=float, default=300.0, help="clip start in the full flight video, s")
    ap.add_argument("--clip-len", type=float, default=180.0, help="clip length, s")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--variant", default=None, help="track step: tracker variant (default: from the config)")
    ap.add_argument("--force", action="store_true", help="redo detection even if the output exists")
    a = ap.parse_args()
    steps[a.step](a)


if __name__ == "__main__":
    main()
