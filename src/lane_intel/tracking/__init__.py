"""Tracking: turn per-frame detections into vehicles that keep one ID over time.

Stage 2 of the pipeline. It reads the detections saved by ``lane_intel.detect`` and never
runs the detector, so a tracker setting can be tried in seconds.

Modules
    config      tracker variants: load ``configs/tracking.json``, resolve ``extends``,
                build the tracker object
    runner      run one variant over saved detections -> points (one row per tracked box)
    quality     measure ID persistence without ground truth (fragmentation, junk tracks,
                identity jumps, coverage)
    experiment  run many variants on the same detections, compare, apply the selection rule
    crosscheck  compare the tuning window with a held-out window; the held-out decides
    __main__    ``python -m lane_intel.tracking`` runs the configured variant for one window

Method and results: notes/tracking.md
"""
from lane_intel.tracking.config import TrackerConfig, build_tracker, load_variants
from lane_intel.tracking.quality import QualityParams, evaluate
from lane_intel.tracking.runner import run_tracker

__all__ = ["TrackerConfig", "build_tracker", "load_variants", "QualityParams", "evaluate", "run_tracker"]
