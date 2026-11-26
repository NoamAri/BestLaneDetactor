"""Build the browser viewer for several video clips; the viewer lets you choose between them.

The clips are listed in a spec file (configs/viewer.json):

    {"clips": [
      {"id": "mitra_t8_d1", "cfg": "configs/T8_D1.json", "start": 0, "dur": 30,
       "experiment": "data/out/tracking/T8_D1_0-30s"},
      {"id": "slowest_car_1943pm", "cfg": "configs/slowest_car_1943pm.json", "start": 0, "dur": 64.25,
       "experiment": "data/out/tracking/slowest_car_1943pm_0-64.25s"}
    ]}

Per clip: ``label`` (default: the camera config's label), ``variants`` (default: the config's
tracking.viewer_variants), and either ``experiment`` (a tracking experiment folder) or ``points``
(a single points parquet).

Site layout written to --out:
    index.html                     the viewer
    clips.json                     [{id, label, dir}], in spec order; the first opens by default
    clips/<id>/video.mp4, meta.json, tags_<variant>.json

Usage
    python -m lane_intel.viewer_site --spec configs/viewer.json --out data/out/viewer [--only id1,id2]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from lane_intel.export_viewer import VIEWER_SRC, ViewerVariant, build_bundle, variants_from_experiment, wrap_page


def clip_variants(clip: dict, cfg: dict) -> list[ViewerVariant]:
    if clip.get("experiment"):
        names = clip.get("variants") or cfg["tracking"].get("viewer_variants", ["@selected"])
        return variants_from_experiment(clip["experiment"], names)
    if clip.get("points"):
        return [ViewerVariant(name="points", points=pd.read_parquet(clip["points"]), label=Path(clip["points"]).stem)]
    raise ValueError(f"clip {clip['id']}: give 'experiment' or 'points'")


def build_site(spec: dict, out: str | Path, only: set[str] | None = None) -> list[dict]:
    out = Path(out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    index = []
    for clip in spec["clips"]:
        cfg = json.loads(Path(clip["cfg"]).read_text())
        cdir = out / "clips" / clip["id"]
        if only is None or clip["id"] in only:
            print(f"== {clip['id']}")
            build_bundle(cfg, clip_variants(clip, cfg), cdir, clip.get("start", 0.0), clip["dur"], write_page=False)
        elif not (cdir / "meta.json").exists():
            raise FileNotFoundError(f"clip {clip['id']} was never built; run without --only first")
        index.append({"id": clip["id"], "label": clip.get("label") or cfg.get("label") or clip["id"],
                      "dir": f"clips/{clip['id']}/"})
    keep = {c["id"] for c in index}
    for d in (out / "clips").iterdir():                  # clips no longer in the spec
        if d.is_dir() and d.name not in keep:
            shutil.rmtree(d)
    for stale in ["video.mp4", "meta.json", *[f.name for f in out.glob("tags_*.json")]]:
        (out / stale).unlink(missing_ok=True)             # single-clip layout from before
    (out / "clips.json").write_text(json.dumps(index, indent=1))
    (out / "index.html").write_text(wrap_page(VIEWER_SRC.read_text()))
    print(f"viewer site -> {out}  ({len(index)} clips)")
    return index


def site_files(out: str | Path) -> list[str]:
    """Every file the site needs, relative to its root (used to publish exactly these)."""
    out = Path(out)
    files = ["index.html", "clips.json"]
    for c in json.loads((out / "clips.json").read_text()):
        meta = json.loads((out / c["dir"] / "meta.json").read_text())
        files += [c["dir"] + f for f in ["video.mp4", "meta.json", *[v["file"] for v in meta["variants"]]]]
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="rebuild only these clip ids (others must already exist)")
    a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",") if x.strip()} or None
    build_site(json.loads(Path(a.spec).read_text()), a.out, only)


if __name__ == "__main__":
    main()
