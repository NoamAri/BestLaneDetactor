"""Run several tracker variants on the same saved detections and compare them.

Usage
    python -m lane_intel.tracking.experiment --cfg configs/T8_D1.json \
        --dets data/interim/T8_D1_0-30s_detections.parquet \
        --out-dir data/out/tracking/T8_D1_0-30s [--variants baseline,s1_lowconf,...]

Writes into --out-dir
    points_<variant>.parquet     tracked boxes for each variant (viewer / later stages read these)
    <variant>/tracks.csv         one row per track (lifetime, break flags)
    <variant>/restarts.csv       break -> restart pairs found by the quality check
    <variant>/jumps.csv          identity-jump events
    report.json                  configs, metrics, guard verdicts, selected variant, provenance
    report.md                    the comparison table in Markdown

Selection rule (fixed before the first run, see notes/tracking.md):
    1. Guards, relative to the reference variant (--reference; the camera config's
       tracking.reference, default "baseline"):
       identity_jumps <= baseline * (1 + JUMP_TOLERANCE) + JUMP_SLACK
       coverage_clear >= baseline - COVERAGE_TOLERANCE
    2. Among variants passing both guards: lowest fragmentation,
       then fewest ids, then fewest short tracks.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lane_intel.detect import meta_path
from lane_intel.tracking.config import TrackerConfig, load_variants
from lane_intel.tracking.quality import QualityParams, evaluate
from lane_intel.tracking.runner import run_tracker

BASELINE = "baseline"
JUMP_TOLERANCE = 0.20      # identity jumps may rise at most 20% over the baseline ...
JUMP_SLACK = 2             # ... plus 2 events, so tiny counts do not decide on noise
COVERAGE_TOLERANCE = 0.01  # clear-box coverage may fall at most 1 percentage point

TABLE_COLUMNS = [
    ("ids", "IDs"), ("fragmentation", "Mid-image starts+ends"), ("born_mid", "Starts"),
    ("died_mid", "Ends"), ("short_lt_1s", "< 1 s"), ("restart_pairs", "Restart pairs"),
    ("median_span_s", "Median life (s)"), ("identity_jumps", "Identity jumps"),
    ("coverage_clear", "Clear-box coverage"), ("stitch_links", "Stitches"),
]


def judge(results: dict[str, dict], baseline: str = BASELINE) -> tuple[str | None, dict[str, list[str]]]:
    """Apply the selection rule. Returns (selected variant, reasons a variant failed a guard)."""
    if baseline not in results:
        raise KeyError(f"the selection rule needs the '{baseline}' variant in the run")
    b = results[baseline]
    max_jumps = b["identity_jumps"] * (1 + JUMP_TOLERANCE) + JUMP_SLACK
    min_cov = b["coverage_clear"] - COVERAGE_TOLERANCE
    failures: dict[str, list[str]] = {}
    for name, s in results.items():
        why = []
        if s["identity_jumps"] > max_jumps:
            why.append(f"identity jumps {s['identity_jumps']} > limit {max_jumps:.0f}")
        if s["coverage_clear"] < min_cov:
            why.append(f"coverage {s['coverage_clear']:.3f} < limit {min_cov:.3f}")
        failures[name] = why
    eligible = [n for n in results if not failures[n]]
    if not eligible:
        return None, failures
    best = min(eligible, key=lambda n: (results[n]["fragmentation"], results[n]["ids"], results[n]["short_lt_1s"]))
    return best, failures


def markdown_table(results: dict[str, dict], configs: dict[str, TrackerConfig],
                   failures: dict[str, list[str]], selected: str | None) -> str:
    head = "| Variant | Step | " + " | ".join(lbl for _, lbl in TABLE_COLUMNS) + " | Guards |"
    sep = "|" + "---|" * (len(TABLE_COLUMNS) + 3)
    lines = [head, sep]
    for name, s in results.items():
        cells = []
        for key, _ in TABLE_COLUMNS:
            v = s.get(key)
            cells.append(f"{v:.1%}" if key == "coverage_clear" else str(v))
        mark = " **(selected)**" if name == selected else ""
        guard = "pass" if not failures[name] else "; ".join(failures[name])
        lines.append(f"| `{name}`{mark} | {configs[name].step} | " + " | ".join(cells) + f" | {guard} |")
    return "\n".join(lines)


def run(cfg: dict, dets_path: str | Path, out_dir: str | Path, names: list[str] | None = None,
        params: QualityParams = QualityParams(), reference: str | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = load_variants(cfg["tracking"]["variants_file"])
    reference = reference or cfg["tracking"].get("reference", BASELINE)
    names = names or list(variants)
    for must in (reference, BASELINE):
        if must not in names:
            names = [must] + names

    dets = pd.read_parquet(dets_path)
    prov_file = meta_path(dets_path)
    prov = json.loads(prov_file.read_text()) if prov_file.exists() else {}
    fps = cfg["native_fps"]
    first = prov.get("frame_first", int(dets.frame_idx.min()))
    frame_range = (first, first + prov["frames"] - 1) if "frames" in prov else (int(dets.frame_idx.min()),
                                                                                 int(dets.frame_idx.max()))
    width = cfg["frame_size"][0]

    results: dict[str, dict] = {}
    for name in names:
        vc = variants[name]
        pts, stats, links = run_tracker(dets, vc, fps, frame_range, width)
        summary, tables = evaluate(pts, fps, frame_range, width, stats, params)
        summary["stitch_links"] = stats["stitch_links"]
        tables["links"] = links
        results[name] = summary
        pts.to_parquet(out_dir / f"points_{name}.parquet", index=False)
        vdir = out_dir / name
        vdir.mkdir(exist_ok=True)
        for tname, table in tables.items():
            table.to_csv(vdir / f"{tname}.csv", index=False)
        print(f"{name:26s} ids={summary['ids']:4d}  fragmentation={summary['fragmentation']:4d}  "
              f"short={summary['short_lt_1s']:4d}  restarts={summary['restart_pairs']:4d}  "
              f"jumps={summary['identity_jumps']:3d}  coverage={summary['coverage_clear']:.3f}  "
              f"links={summary['stitch_links']:3d}  "
              f"({stats['seconds']:.1f}s)", flush=True)

    selected, failures = judge(results, reference)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detections": str(dets_path), "detections_provenance": prov,
        "frame_range": list(frame_range), "fps": fps, "image_width": width,
        "quality_params": params.to_dict(),
        "selection_rule": {"reference": reference, "jump_tolerance": JUMP_TOLERANCE, "jump_slack": JUMP_SLACK,
                           "coverage_tolerance": COVERAGE_TOLERANCE,
                           "order": ["fragmentation", "ids", "short_lt_1s"]},
        "variants": {n: {"config": variants[n].to_dict(), "summary": results[n], "guard_failures": failures[n]}
                     for n in names},
        "selected": selected,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    md = markdown_table(results, variants, failures, selected)
    (out_dir / "report.md").write_text(f"# Tracking experiment\n\nDetections: `{dets_path}`  \n"
                                       f"Frames {frame_range[0]}-{frame_range[1]}  \n"
                                       f"Guards relative to: `{reference}`\n\n{md}\n\n"
                                       f"Selected: `{selected}`\n")
    print(f"\nselected: {selected}\nreport -> {out_dir / 'report.md'}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variants", default="", help="comma-separated names; default: all in the variants file")
    ap.add_argument("--reference", default=None, help="variant the guards compare against (default: config)")
    a = ap.parse_args()
    names = [n.strip() for n in a.variants.split(",") if n.strip()] or None
    run(json.load(open(a.cfg)), a.dets, a.out_dir, names, reference=a.reference)


if __name__ == "__main__":
    main()
