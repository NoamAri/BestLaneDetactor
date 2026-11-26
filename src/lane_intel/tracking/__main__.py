"""Track one window with one variant (the configured one by default).

Usage
    python -m lane_intel.tracking --cfg configs/T8_D1.json \
        --dets data/interim/T8_D1_0-30s_detections.parquet \
        --out data/interim/T8_D1_0-30s_points.parquet [--variant s1_lowconf]

Writes the points parquet and <out>.meta.json (variant config, run stats, quality summary).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lane_intel.detect import meta_path
from lane_intel.tracking.config import load_variants
from lane_intel.tracking.quality import QualityParams, evaluate
from lane_intel.tracking.runner import run_tracker


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default=None, help="default: tracking.variant in the camera config")
    a = ap.parse_args()

    cfg = json.load(open(a.cfg))
    name = a.variant or cfg["tracking"]["variant"]
    vc = load_variants(cfg["tracking"]["variants_file"])[name]
    dets = pd.read_parquet(a.dets)
    prov_file = meta_path(a.dets)
    prov = json.loads(prov_file.read_text()) if prov_file.exists() else {}
    first = prov.get("frame_first", int(dets.frame_idx.min()))
    last = first + prov["frames"] - 1 if "frames" in prov else int(dets.frame_idx.max())

    pts, stats, links = run_tracker(dets, vc, cfg["native_fps"], (first, last), cfg["frame_size"][0])
    summary, _ = evaluate(pts, cfg["native_fps"], (first, last), cfg["frame_size"][0], stats,
                          QualityParams.scaled(cfg["frame_size"][0]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pts.to_parquet(out, index=False)
    if len(links):
        links.to_csv(out.with_name(out.stem + ".links.csv"), index=False)
    meta_path(out).write_text(json.dumps({"variant": vc.to_dict(), "detections": a.dets,
                                          "stats": stats, "quality": summary}, indent=2))
    print(f"{name}: {summary['ids']} IDs, fragmentation {summary['fragmentation']}, "
          f"{len(pts)} tagged boxes -> {out}")


if __name__ == "__main__":
    main()
