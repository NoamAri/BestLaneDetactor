"""Tracker variants: one named, fully specified tracker setting per experiment arm.

A variant file (configs/tracking.json) holds ``{"variants": {name: spec}}``. A spec may say
``"extends": "<parent>"``; it then inherits every field of the parent and overrides only what
it states. ``params`` (the tracker constructor arguments) merge key by key when the tracker
type is unchanged and are replaced when the tracker type changes, because different
trackers accept different arguments.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import trackers

TRACKER_CLASSES = {
    "bytetrack": trackers.ByteTrackTracker,
    "ocsort": trackers.OCSORTTracker,
    "botsort": trackers.BoTSORTTracker,
    "sort": trackers.SORTTracker,
}

# The trackers package counts lost_track_buffer in frames at 30 fps and rescales it by
# frame_rate / 30 internally, so seconds convert to "30 fps frames" by multiplying by 30.
BUFFER_REFERENCE_FPS = 30.0


@dataclass(frozen=True)
class TrackerConfig:
    name: str
    tracker: str
    min_conf: float
    lost_buffer_s: float
    params: dict = field(default_factory=dict)
    label: str = ""
    note: str = ""
    step: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve(name: str, specs: dict, seen: tuple = ()) -> dict:
    if name not in specs:
        raise KeyError(f"unknown tracker variant '{name}' (known: {', '.join(sorted(specs))})")
    if name in seen:
        raise ValueError(f"circular 'extends' chain: {' -> '.join(seen + (name,))}")
    spec = dict(specs[name])
    parent_name = spec.pop("extends", None)
    if parent_name is None:
        return spec
    parent = _resolve(parent_name, specs, seen + (name,))
    merged = {**parent, **{k: v for k, v in spec.items() if k != "params"}}
    same_tracker = merged.get("tracker") == parent.get("tracker")
    if "params" in spec:
        merged["params"] = {**parent.get("params", {}), **spec["params"]} if same_tracker else dict(spec["params"])
    elif not same_tracker:
        merged["params"] = {}
    return merged


def _validate(cfg: TrackerConfig) -> None:
    if cfg.tracker not in TRACKER_CLASSES:
        raise ValueError(f"{cfg.name}: tracker must be one of {sorted(TRACKER_CLASSES)}, got '{cfg.tracker}'")
    if not 0.0 <= cfg.min_conf <= 1.0:
        raise ValueError(f"{cfg.name}: min_conf must be within [0, 1]")
    if cfg.lost_buffer_s < 0:
        raise ValueError(f"{cfg.name}: lost_buffer_s must be >= 0")
    accepted = set(inspect.signature(TRACKER_CLASSES[cfg.tracker].__init__).parameters) - {"self"}
    reserved = {"lost_track_buffer", "frame_rate"}
    unknown = set(cfg.params) - accepted
    clash = set(cfg.params) & reserved
    if unknown:
        raise ValueError(f"{cfg.name}: {cfg.tracker} does not accept {sorted(unknown)}")
    if clash:
        raise ValueError(f"{cfg.name}: set {sorted(clash)} through lost_buffer_s / the video fps, not params")


def parse_variants(raw: dict) -> dict[str, TrackerConfig]:
    """Resolve every variant in a parsed variants document."""
    specs = raw["variants"]
    out = {}
    for name in specs:
        r = _resolve(name, specs)
        cfg = TrackerConfig(name=name, tracker=r["tracker"], min_conf=float(r["min_conf"]),
                            lost_buffer_s=float(r["lost_buffer_s"]), params=dict(r.get("params", {})),
                            label=r.get("label", name), note=r.get("note", ""), step=str(r.get("step", "")))
        _validate(cfg)
        out[name] = cfg
    return out


def load_variants(path: str | Path) -> dict[str, TrackerConfig]:
    """Load and resolve configs/tracking.json (or any file with the same layout)."""
    return parse_variants(json.loads(Path(path).read_text()))


def build_tracker(cfg: TrackerConfig, fps: float):
    """Instantiate the tracker class for a variant at the video's frame rate."""
    cls = TRACKER_CLASSES[cfg.tracker]
    return cls(lost_track_buffer=int(round(cfg.lost_buffer_s * BUFFER_REFERENCE_FPS)),
               frame_rate=float(fps), **cfg.params)
