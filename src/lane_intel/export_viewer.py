"""Bundle a browser-friendly video + per-frame tags for app/viewer/index.html.

One bundle can carry several tracker variants of the same window so the viewer can switch
between them (for example the baseline and the tuned tracker).

Writes into --out-dir:
    video.mp4            road-strip proxy (short GOP so frame stepping is snappy)
    tags_<variant>.json  one array, one record per line:
                         [frame, track_id, cls_idx, conf_pct, x1, y1, x2, y2]  (full-frame px)
    meta.json            fps, frame range, crop/scale mapping, class names, and per variant:
                         label, note, quality summary, one summary row per track
    index.html           the viewer, wrapped in a document skeleton for local serving

Usage
    # variants from a tracking experiment (points_<name>.parquet + report.json in --experiment)
    python -m lane_intel.export_viewer --cfg configs/T8_D1.json --duration 30 \
        --experiment data/out/tracking/T8_D1_0-30s --variants baseline,s1_lowconf,s2_bt_buf4s \
        --out-dir data/out/viewer
    # or a single points file
    python -m lane_intel.export_viewer --cfg configs/T8_D1.json --duration 30 \
        --points data/interim/T8_D1_0-30s_points.parquet --out-dir data/out/viewer

Serve locally with:  python -m lane_intel.serve <out-dir> --port 8000  (needs HTTP Range for seeking)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lane_intel.lanes import assign_lanes, lane_polygons, lanes_path, load_lanes
from lane_intel.tracking.quality import QualityParams, track_table

VIEWER_SRC = Path(__file__).resolve().parents[2] / "app" / "viewer" / "index.html"


@dataclass
class ViewerVariant:
    name: str
    points: pd.DataFrame
    label: str = ""
    note: str = ""
    quality: dict = field(default_factory=dict)


def encode_proxy(video: str, out: Path, start_s: float, duration_s: float,
                 crop: tuple[int, int, int, int], out_w: int, crf: int) -> tuple[int, int]:
    x0, y0, w, h = crop
    out_h = int(round(h * out_w / w / 2)) * 2
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start_s:
        cmd += ["-ss", f"{start_s:.6f}"]
    cmd += ["-i", video, "-t", f"{duration_s:.6f}",
            "-vf", f"crop={w}:{h}:{x0}:{y0},scale={out_w}:{out_h}",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-g", "10",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out_w, out_h


def track_summary(df: pd.DataFrame, fps: float, frame_range: tuple[int, int] | None = None,
                  width: int = 3840, params: QualityParams | None = None) -> list[dict]:
    """One record per track for the viewer's register and start/end markers.

    Start/end flags come from tracking.quality.track_table, so the viewer shows exactly the
    breaks the quality metrics count. px_per_s is the average horizontal image motion over
    the whole track (not km/h: that needs calibration).
    """
    if df.empty:
        return []
    if frame_range is None:
        frame_range = (int(df.frame_idx.min()), int(df.frame_idx.max()))
    tr = track_table(df, fps, frame_range, width, params or QualityParams.scaled(width))
    conf = df.groupby("track_id").conf.mean()
    span = tr.span_s.to_numpy()
    px_per_s = np.where(span > 0, (tr.x_last - tr.x_first) / np.where(span > 0, span, 1), 0.0)
    out = pd.DataFrame({
        "track_id": tr.track_id, "cls": tr.cls, "first": tr["first"], "last": tr["last"], "n": tr.n,
        "conf": conf.loc[tr.track_id].round(3).to_numpy(), "span_s": tr.span_s.round(2),
        "px_per_s": np.round(px_per_s, 1),
        "born_mid": tr.born_mid.astype(bool), "died_mid": tr.died_mid.astype(bool),
        "x_first": tr.x_first.round().astype(int), "y_first": tr.y_first.round().astype(int),
        "x_last": tr.x_last.round().astype(int), "y_last": tr.y_last.round().astype(int),
    })
    return json.loads(out.to_json(orient="records"))


def write_tags(df: pd.DataFrame, path: Path, cidx: dict[str, int]) -> None:
    lines = [json.dumps([int(r.frame_idx), int(r.track_id), cidx[r.cls], int(round(r.conf * 100)),
                         int(round(r.x1)), int(round(r.y1)), int(round(r.x2)), int(round(r.y2))],
                        separators=(",", ":"))
             for r in df.sort_values(["frame_idx", "track_id"]).itertuples()]
    path.write_text("[\n" + ",\n".join(lines) + "\n]\n")


def wrap_page(fragment: str) -> str:
    """Page fragment (title/meta/links, then <style>, markup, script) -> complete document."""
    head, sep, rest = fragment.partition("<style>")
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + head.strip() + "\n</head><body>\n" + sep + rest + "\n</body></html>\n")


def build_bundle(cfg: dict, variants: list[ViewerVariant], out_dir: str | Path, start: float,
                 duration: float, crop_y: tuple[int, int] | None = None, width: int | None = None,
                 crf: int = 32, write_page: bool = True, lanes_file: str | Path | None = None,
                 stab: pd.DataFrame | None = None, lane_speed: dict | None = None, road_frame=None) -> dict:
    """Write video.mp4, tags_<variant>.json, meta.json (and index.html); return meta.

    The first variant is the one the viewer opens with. crop_y / width default to the camera
    config's ``viewer`` block. Lanes live in the stabilised reference frame: with ``stab`` (the
    window's camera motion) vehicles are assigned on stabilised points and the lane outlines are
    drawn moved into the video frame (by the window's median camera position). write_page=False when the clip is part of a multi-clip site
    (viewer_site.py writes one index.html at the site root).
    """
    vcfg = cfg.get("viewer", {})
    crop_y = tuple(crop_y or vcfg.get("crop_y") or (0, cfg["frame_size"][1]))
    width = width or vcfg.get("width") or cfg["frame_size"][0]
    if not variants:
        raise ValueError("need at least one variant")
    fps = cfg["native_fps"]
    W = cfg["frame_size"][0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("tags_*.json"):
        old.unlink()

    crop = (0, crop_y[0], W, crop_y[1] - crop_y[0])
    vw, vh = encode_proxy(cfg["video"], out / "video.mp4", start, duration, crop, width, crf)

    f0 = round(start * fps)
    f1 = f0 + round(duration * fps)
    windowed = {v.name: v.points[(v.points.frame_idx >= f0) & (v.points.frame_idx < f1)] for v in variants}
    classes = sorted(set().union(*(set(p.cls.unique()) for p in windowed.values())))
    cidx = {c: i for i, c in enumerate(classes)}

    geo = lanes = None
    if lanes_file and Path(lanes_file).exists():
        geo, lanes = load_lanes(lanes_file)
    meta_variants = []
    lane_of_first = None
    for v in variants:
        df = windowed[v.name]
        fname = f"tags_{v.name}.json"
        write_tags(df, out / fname, cidx)
        tracks = track_summary(df, fps, (f0, f1 - 1), W)
        ref = df
        if stab is not None:
            from lane_intel.stabilize import apply
            ref = df.copy()
            ref["x_px"], ref["y_px"] = apply(stab, df.frame_idx.to_numpy(), df.x_px, df.y_px)
        if road_frame is not None and len(ref):
            add_road_speed(tracks, ref, road_frame, fps)
        if lanes:
            lane_of = assign_lanes(ref, geo, lanes, W).groupby("track_id").lane_track.first()
            if lane_of_first is None:
                lane_of_first = lane_of
            for t in tracks:
                t["lane"] = int(lane_of.get(t["track_id"], -1))
        meta_variants.append({
            "name": v.name, "label": v.label or v.name, "note": v.note, "file": fname,
            "quality": v.quality, "tags": int(len(df)), "tracks": tracks,
        })
        print(f"  {v.name}: {len(df)} tags, {df.track_id.nunique()} tracks -> {fname}")

    det = cfg.get("detector", {})
    meta = {
        "label": cfg.get("label") or f"Flight {cfg.get('flight')} · drone {cfg.get('drone')}",
        "time_label": cfg.get("time_label", "Source time"),
        "px_scale": W / 3840,          # pixel thresholds in the viewer were set on 3840 px frames
        "fps": fps, "clip_offset_s": cfg["clip_offset_s"],
        "frame_first": f0, "frame_count": f1 - f0,
        "frame_size": cfg["frame_size"],
        "crop": {"x": crop[0], "y": crop[1], "w": crop[2], "h": crop[3]},
        "video_size": [vw, vh],
        "classes": classes,
        "model": Path(det.get("model", "?")).name, "imgsz": det.get("imgsz"),
        "edge_px": QualityParams.scaled(W).edge_px,
        "variants": meta_variants,
        "lanes": _on_vehicles(_to_video(lane_polygons(geo, lanes), stab, f0, f1), windowed[variants[0].name],
                              lane_of_first) if lanes else [],
        "lane_speed": lane_speed,
    }
    (out / "meta.json").write_text(json.dumps(_json_clean(meta), allow_nan=False))

    if write_page and VIEWER_SRC.exists():
        (out / "index.html").write_text(wrap_page(VIEWER_SRC.read_text()))
    print(f"viewer bundle -> {out}  ({len(variants)} variant(s), video {vw}x{vh})")
    return meta


def _json_clean(o):
    """NaN / inf -> None, numpy scalars -> Python: browsers reject NaN in JSON."""
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


LIVE_HZ = 5            # live speed samples per second (per vehicle)
LIVE_WINDOW_S = 2.0    # each sample is a straight-line fit of distance over this window (box wobble averages out)
LIVE_MIN_SPAN_S = 0.8  # at the ends of a track the window is shorter; below this span: no live value


def live_speed(t: np.ndarray, s: np.ndarray, times: np.ndarray, sign: float) -> np.ndarray:
    """Speed (km/h, in the direction of travel, >= 0) at each of `times`: least-squares slope of s(t)
    over a LIVE_WINDOW_S window centred on it, using whatever part of the track lies inside. NaN where
    the window holds fewer than 3 boxes or spans less than LIVE_MIN_SPAN_S."""
    out = np.full(len(times), np.nan)
    h = LIVE_WINDOW_S / 2
    lo = np.searchsorted(t, times - h, "left")
    hi = np.searchsorted(t, times + h, "right")
    for k, (a, b) in enumerate(zip(lo, hi)):
        if b - a < 3 or t[b - 1] - t[a] < LIVE_MIN_SPAN_S:
            continue
        tt, ss = t[a:b], s[a:b]
        tc = tt - tt.mean()
        den = float(tc @ tc)
        if den > 0:
            out[k] = max(0.0, sign * float(tc @ (ss - ss.mean())) / den) * 3.6
    return out


def add_road_speed(tracks: list[dict], ref: pd.DataFrame, rf, fps: float = 0.0) -> None:
    """Per track: kmh (average: distance along the road over the track's span), m_per_px (for the
    distance graph) and, with fps, a live speed series vs = {f0, df, kmh: [...]} sampled at LIVE_HZ
    (frame indices f0, f0 + df, ...; null where too little of the track is in view)."""
    s_, _ = rf.sd(ref.x_px.to_numpy(), ref.y_px.to_numpy())
    r = ref.assign(s=s_).sort_values(["track_id", "t"])
    g = r.groupby("track_id")
    first, last = g.first(), g.last()
    span = (last.t - first.t)
    ds = (last.s - first.s).abs()
    dx = (last.x_px - first.x_px).abs()
    df = max(1, int(round(fps / LIVE_HZ))) if fps else 0
    series = {}
    if df:
        for tid, q in g:
            t, s, f = q.t.to_numpy(), q.s.to_numpy(), q.frame_idx.to_numpy()
            if len(t) < 3:
                continue
            fs = np.arange(f[0], f[-1] + 1, df)
            v = live_speed(t, s, fs / fps, 1.0 if s[-1] >= s[0] else -1.0)
            series[tid] = {"f0": int(f[0]), "df": df, "kmh": [None if not np.isfinite(x) else round(float(x), 1) for x in v]}
    for t in tracks:
        k = t["track_id"]
        if k in span.index and span[k] > 0:
            t["kmh"] = round(float(ds[k] / span[k] * 3.6), 1)
            t["m_per_px"] = round(float(ds[k] / dx[k]), 5) if dx[k] > 1 else None
        if k in series:
            t["vs"] = series[k]


def _on_vehicles(polys: list[dict], pts: pd.DataFrame, lane_of) -> list[dict]:
    """Shift each lane's outline (display only) so it runs through its vehicles' boxes.

    Lanes are measured at the ground point (box bottom-centre). In an oblique view a vehicle's
    body rises above that point, so a band drawn at the ground points sits under the cars. The
    shift is the median offset of box centres from the lane's centre line; it is stored as
    display_dy so it is clear the measured geometry itself is unchanged.
    """
    if lane_of is None or pts.empty:
        return polys
    p = pts.assign(lane=pts.track_id.map(lane_of)).dropna(subset=["lane"])
    out = []
    for i, ln in enumerate(polys):
        q = p[p.lane == i]
        if len(q) < 5:
            out.append(ln)
            continue
        c = np.asarray(ln["centre"], float)
        cx = ((q.x1 + q.x2) / 2).to_numpy()
        dy = float(np.median((q.y1 + q.y2).to_numpy() / 2 - np.interp(cx, c[:, 0], c[:, 1])))
        out.append({**ln, "display_dy": round(dy, 1),
                    **{k: [[x, round(y + dy, 1)] for x, y in ln[k]] for k in ("centre", "upper", "lower")}})
    return out


def _to_video(polys: list[dict], stab: pd.DataFrame | None, f0: int, f1: int) -> list[dict]:
    """Lane outlines from the stabilised reference frame into the video frame (median camera position)."""
    if stab is None:
        return polys
    import cv2
    from lane_intel.stabilize import matrices
    Hs = [H for f, H in matrices(stab).items() if f0 <= f < f1]
    if not Hs:
        return polys
    Hinv = np.linalg.inv(np.median(np.stack(Hs), axis=0))
    out = []
    for p in polys:
        q = dict(p)
        for k in ("centre", "upper", "lower"):
            pts = np.asarray(p[k], float).reshape(-1, 1, 2)
            q[k] = np.round(cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2), 1).tolist()
        out.append(q)
    return out


def variants_from_experiment(exp_dir: str | Path, names: list[str]) -> list[ViewerVariant]:
    """Load points_<name>.parquet and labels/quality from an experiment's report.json.

    The name '@selected' stands for the variant the experiment's selection rule picked.
    """
    exp_dir = Path(exp_dir)
    report = json.loads((exp_dir / "report.json").read_text())
    out, seen = [], set()
    for n in names:
        n = report["selected"] if n == "@selected" else n
        if n is None or n in seen:
            continue
        seen.add(n)
        info = report["variants"][n]
        out.append(ViewerVariant(name=n, points=pd.read_parquet(exp_dir / f"points_{n}.parquet"),
                                 label=info["config"]["label"], note=info["config"]["note"],
                                 quality={**info["summary"], "selected": n == report["selected"],
                                          "guard_failures": info["guard_failures"]}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--points", help="a single points parquet")
    src.add_argument("--experiment", help="a tracking experiment folder (report.json + points_*.parquet)")
    ap.add_argument("--variants", default="", help="with --experiment: comma-separated names, '@selected' allowed")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--crop-y", type=int, nargs=2, default=None, help="default: config viewer.crop_y")
    ap.add_argument("--width", type=int, default=None, help="default: config viewer.width")
    ap.add_argument("--crf", type=int, default=32)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    if a.points:
        variants = [ViewerVariant(name="points", points=pd.read_parquet(a.points), label=Path(a.points).stem)]
    else:
        names = [n.strip() for n in a.variants.split(",") if n.strip()] or ["@selected"]
        variants = variants_from_experiment(a.experiment, names)
    build_bundle(cfg, variants, a.out_dir, a.start, a.duration, a.crop_y, a.width, a.crf)


if __name__ == "__main__":
    main()
