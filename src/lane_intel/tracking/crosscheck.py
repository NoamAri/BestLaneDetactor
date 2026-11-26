"""Compare tracker variants across windows: the tuning window vs a held-out window.

Variants are designed while looking at one window (the tuning window), so their scores there
are optimistic. The held-out window was never used to design or tune anything; the selection
rule applied to it decides the final tracker (notes/tracking.md, section 2.6).

Usage
    python -m lane_intel.tracking.crosscheck \
        --tuning data/out/tracking/T8_D1_0-30s --heldout data/out/tracking/T8_D1_30-30s \
        --out data/out/tracking/crosscheck.md

Writes a Markdown table (fragmentation change vs baseline, identity jumps, guard result, on
both windows) and prints the final choice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(exp_dir: str | Path) -> dict:
    return json.loads((Path(exp_dir) / "report.json").read_text())


def _rel(v: float, base: float) -> str:
    return f"{(v - base) / base:+.0%}" if base else "n/a"


def crosscheck(tuning: dict, heldout: dict) -> tuple[str, str | None]:
    """Markdown table of both windows; the final choice is the held-out window's selection."""
    names = [n for n in tuning["variants"] if n in heldout["variants"]]
    bt = tuning["variants"]["baseline"]["summary"]
    bh = heldout["variants"]["baseline"]["summary"]
    lines = [
        "| Variant | Step | Fragmentation, tuning (vs baseline) | Jumps, tuning | Guards, tuning "
        "| Fragmentation, held-out (vs baseline) | Jumps, held-out | Guards, held-out |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        t, h = tuning["variants"][n], heldout["variants"][n]
        ts, hs = t["summary"], h["summary"]
        mark = " **(final)**" if n == heldout["selected"] else ""
        lines.append(
            f"| `{n}`{mark} | {t['config']['step']} "
            f"| {ts['fragmentation']} ({_rel(ts['fragmentation'], bt['fragmentation'])}) | {ts['identity_jumps']} "
            f"| {'pass' if not t['guard_failures'] else 'fail'} "
            f"| {hs['fragmentation']} ({_rel(hs['fragmentation'], bh['fragmentation'])}) | {hs['identity_jumps']} "
            f"| {'pass' if not h['guard_failures'] else 'fail'} |")
    return "\n".join(lines), heldout["selected"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuning", required=True, help="experiment folder of the window used to design variants")
    ap.add_argument("--heldout", required=True, help="experiment folder of the untouched window")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tuning, heldout = load(a.tuning), load(a.heldout)
    table, final = crosscheck(tuning, heldout)
    doc = (f"# Tracker cross-check\n\nTuning window: `{a.tuning}` (selected there: `{tuning['selected']}`)  \n"
           f"Held-out window: `{a.heldout}` (selected there: `{final}`)\n\n{table}\n\n"
           f"Final choice (selection rule on the held-out window): `{final}`\n")
    Path(a.out).write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
