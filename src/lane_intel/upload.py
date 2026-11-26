"""Analyse an uploaded traffic video, but only if it is a good one (notes/upload_plan.md).

    python -m lane_intel.upload --video clip.mp4 --id my-road --label "My road" [--publish docs]

Stages (rules in configs/upload_rules.json; the first failure stops everything):
  A  the file      size, length (> 30 s, < 60 s), resolution, frame rate, decodes
  B  the picture   fixed camera, one continuous shot, light and sharpness, enough vehicles
  C  the result    vehicles followed, lanes found, a measurable stretch, plausible speeds
Only a video that passes A and B is analysed; only one that also passes C is published.

The verdict is written to data/out/uploads/<id>/result.json (status good / not_good / error, the
failed rule in plain words, a tip, the answer) for the GitHub workflow to post as a comment.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "configs" / "upload_rules.json"
PY = sys.executable


class NotGood(Exception):
    """A rule failed: the video is not analysed (or not published)."""
    def __init__(self, stage: str, rule: str, message: str, tip: str = ""):
        super().__init__(message)
        self.stage, self.rule, self.message, self.tip = stage, rule, message, tip


def clean_label(s: str) -> str:
    """The name shown in the app: plain text only (no HTML / script characters), one line, at most 60 characters."""
    s = re.sub(r"[<>&\"'`\\{}$]", "", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:60] or "video"


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s or "video")[:40]


# ---------------------------------------------------------------- stage A: the file
def probe(path: str | Path) -> dict:
    """Duration, size, displayed resolution (rotation applied), frame rate, codec."""
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()[:300]}
    d = json.loads(r.stdout)
    vs = [s for s in d.get("streams", []) if s.get("codec_type") == "video"]
    if not vs:
        return {"ok": False, "error": "no video stream"}
    v = vs[0]
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = int(v.get("tags", {}).get("rotate", rot) or rot)
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    dur = float(d.get("format", {}).get("duration") or v.get("duration") or 0)
    return {"ok": True, "duration_s": dur, "width": w, "height": h, "fps": fps, "codec": v.get("codec_name"),
            "size_mb": Path(path).stat().st_size / 1e6}


def check_file(path: str | Path, info: dict, rules: dict) -> None:
    f = rules["file"]
    ext = Path(path).suffix.lower().lstrip(".")
    if ext not in f["extensions"]:
        raise NotGood("A", "A2", f"The file is a .{ext or '?'} file, not a video.", "Attach one video file (mp4, mov or webm).")
    if info.get("size_mb", 0) > f["max_mb"]:
        raise NotGood("A", "A3", f"The file is {info['size_mb']:.0f} MB; the limit is {f['max_mb']:.0f} MB.",
                      "Record at 720p, or use your phone's 'reduce size' / 'smaller' option when sharing.")
    if not info.get("ok"):
        raise NotGood("A", "A5", "The video could not be read.", "Send an mp4 straight from the phone's camera.")
    d = info["duration_s"]
    if not (f["min_duration_s"] < d < f["max_duration_s"]):
        raise NotGood("A", "A4", f"The video is {d:.0f} s long.",
                      f"Please send a video longer than {f['min_duration_s']:.0f} s and shorter than {f['max_duration_s']:.0f} s.")
    if info["width"] < f["min_width_px"]:
        raise NotGood("A", "A5", f"The video is {info['width']} px wide.",
                      f"Please send at least {f['min_width_px']} px wide (720p in landscape).")
    if info["fps"] < f["min_fps"]:
        raise NotGood("A", "A5", f"The video has {info['fps']:.1f} frames per second.", f"Please send at least {f['min_fps']:.0f} fps.")


def normalise(src: str | Path, dst: str | Path, rules: dict) -> None:
    """Same format for every upload: at most N fps (never more than the source: duplicated frames would make
    vehicles look stop-and-go), at most W px wide (even size), no audio, H.264."""
    n = rules["normalise"]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    src_fps = probe(src).get("fps") or n["fps"]
    fps = n["fps"] if src_fps >= n["fps"] - 0.01 else src_fps
    vf = f"fps={fps:g},scale='trunc(min({n['max_width_px']},iw)/2)*2':-2"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)


# ---------------------------------------------------------------- stage B: the picture
def _frames(path: str | Path, fps: float, width: int, gray: bool) -> list[np.ndarray]:
    info = probe(path)
    h = int(round(info["height"] * width / info["width"] / 2)) * 2
    pix = "gray" if gray else "bgr24"
    ch = 1 if gray else 3
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{h}", "-f", "rawvideo",
                        "-pix_fmt", pix, "-"], capture_output=True, check=True)
    n = width * h * ch
    arr = np.frombuffer(p.stdout, np.uint8)
    k = len(arr) // n
    return [arr[i * n:(i + 1) * n].reshape((h, width) if gray else (h, width, 3)) for i in range(k)]


def picture_stats(video: str | Path, rules: dict) -> dict:
    """Camera steadiness, cuts, light and sharpness on 1 frame per second (960 px wide, grey)."""
    pr = rules["picture"]
    W = 960
    fr = _frames(video, pr["sample_fps"], W, gray=True)
    if len(fr) < 3:
        return {"frames": len(fr)}
    ref = fr[0]
    feats = cv2.goodFeaturesToTrack(ref, 2000, 0.01, 8, blockSize=7)
    h = ref.shape[0]
    probe_pts = np.array([[0, 0], [W, 0], [0, h], [W, h], [W / 2, h / 2]], np.float32).reshape(-1, 1, 2)
    inliers, shift = [], []
    for f in fr[1:]:
        if feats is None or len(feats) < 10:
            inliers.append(0); shift.append(np.inf); continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref, f, feats, None, winSize=(31, 31), maxLevel=4)
        ok = st[:, 0] == 1
        if ok.sum() < 10:
            inliers.append(0); shift.append(np.inf); continue
        H, m = cv2.findHomography(nxt[ok].reshape(-1, 2), feats[ok].reshape(-1, 2), cv2.RANSAC, 2.0)
        if H is None:
            inliers.append(0); shift.append(np.inf); continue
        inliers.append(int(m.sum()))
        q = cv2.perspectiveTransform(probe_pts, H)
        shift.append(float(np.max(np.linalg.norm((q - probe_pts).reshape(-1, 2), axis=1))))
    small = [cv2.resize(f, (160, int(160 * h / W))).astype(np.float32) for f in fr]
    diffs = [float(np.mean(np.abs(a - b))) for a, b in zip(small[:-1], small[1:])]
    return {"frames": len(fr),
            "bg_inliers_min": int(min(inliers)), "bg_inliers_median": int(np.median(inliers)),
            "camera_shift_frac_max": float(min(1e9, max(shift)) / W),
            "max_step_diff": float(max(diffs)), "median_step_diff": float(np.median(diffs)),
            "brightness": float(np.median([f.mean() for f in fr])),
            "sharpness": float(np.median([cv2.Laplacian(f, cv2.CV_64F).var() for f in fr]))}


def vehicle_sample(video: str | Path, rules: dict, model_path: str | Path) -> dict:
    """Quick detection on a few frames spread over the clip: vehicles per frame, their size, the road band."""
    from ultralytics import YOLO
    pr, an = rules["picture"], rules["analysis"]
    info = probe(video)
    model = YOLO(str(model_path))
    ts = np.linspace(1.0, info["duration_s"] - 1.0, pr["detect_samples"])
    counts, lengths, y1s, y2s, hs = [], [], [], [], []
    for t in ts:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
        img = np.frombuffer(p.stdout, np.uint8).reshape(info["height"], info["width"], 3)
        r = model.predict(img, imgsz=an["detector_imgsz"], classes=[2, 3, 5, 7], conf=0.25, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy()
        counts.append(len(b))
        if len(b):
            lengths += list(b[:, 2] - b[:, 0]); y1s += list(b[:, 1]); y2s += list(b[:, 3]); hs += list(b[:, 3] - b[:, 1])
    out = {"vehicles_per_frame": float(np.median(counts)), "vehicle_length_px": float(np.median(lengths)) if lengths else 0.0}
    if y1s:
        mh = float(np.median(hs))
        lo = max(0, int(np.percentile(y1s, 2) - 1.5 * mh)) // 2 * 2
        hi = min(info["height"], int(np.percentile(y2s, 98) + 1.5 * mh) + 1) // 2 * 2
        out["band_y"] = [lo, max(lo + 64, hi)]
    return out


def check_picture(ps: dict, vs: dict | None, rules: dict) -> None:
    pr = rules["picture"]
    if ps.get("frames", 0) < 3:
        raise NotGood("B", "B1", "The video could not be sampled.", "Send an mp4 straight from the phone's camera.")
    # a cut is one sudden jump, far above the clip's usual change (a pan changes the picture every second)
    if ps["max_step_diff"] > pr["cut_mean_abs_diff"] and ps["max_step_diff"] > 4 * max(1.0, ps["median_step_diff"]):
        raise NotGood("B", "B2", "The video has a cut or a sudden jump.", "Please send one continuous shot.")
    if ps["bg_inliers_min"] < pr["min_bg_inliers"] or ps["camera_shift_frac_max"] > pr["max_camera_shift_frac"]:
        raise NotGood("B", "B1", "The camera moves too much (it pans, zooms or shakes).",
                      "Hold the phone still (lean it on something) or use a drone that hovers in one place.")
    if not (pr["min_brightness"] <= ps["brightness"] <= pr["max_brightness"]):
        raise NotGood("B", "B3", "The video is too dark." if ps["brightness"] < pr["min_brightness"] else "The video is too bright.",
                      "Film in daylight, without the sun in the picture.")
    if ps["sharpness"] < pr["min_sharpness"]:
        raise NotGood("B", "B3", "The video is too blurry.", "Wipe the lens and let the camera focus on the road.")
    if vs is None:
        return
    if vs["vehicles_per_frame"] < pr["min_vehicles_per_frame"]:
        raise NotGood("B", "B4", f"Only about {vs['vehicles_per_frame']:.0f} vehicles are visible per frame.",
                      f"Film a busier road, or zoom in so that at least {pr['min_vehicles_per_frame']} vehicles are in view.")
    if vs["vehicle_length_px"] < pr["min_vehicle_length_px"]:
        raise NotGood("B", "B4", "The vehicles are too small in the picture.", "Zoom in or film from closer to the road.")


# ---------------------------------------------------------------- configuration and analysis
def make_config(vid: str, label: str, video: str | Path, info: dict, band: list[int], rules: dict) -> dict:
    an = rules["analysis"]
    dur = round(info["duration_s"], 2)
    return {
        "label": f"Uploaded · {label}", "time_label": "Video time", "video": str(Path(video).relative_to(ROOT)).replace("\\", "/"),
        "video_note": "Uploaded for analysis; converted to 10 fps, at most 1920 px wide (lane_intel.upload).",
        "clip_offset_s": 0.0, "native_fps": round(float(info["fps"]), 4), "frame_size": [info["width"], info["height"]],
        "band_y": band,
        "detector": {"model": "data/models/yolo26s.pt", "classes": [2, 3, 5, 7], "imgsz": an["detector_imgsz"],
                     "conf_min": 0.05, "max_det": 1000, "process_every": 1},
        "tracking": {"variants_file": "configs/tracking.json", "variant": an["tracker_variant"], "reference": "baseline",
                     "viewer_variants": ["@selected"]},
        "viewer": {"crop_y": band, "width": info["width"]},
        "lanes_file": f"configs/uploads/{vid}.lanes.json",
        "calibration": {"lane_scale_sd": 0.025, "method": "vehicle_size", "lane_width_m": an["lane_width_m"],
                        "car_length_m": an["car_length_m"]},
        "lane_speed": {"windows": [[0, dur]], "auxiliary_lanes": [], "speed_limit_kmh": None},
        "uploaded": {"id": vid, "label": label, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }


def _sh(args: list) -> None:
    subprocess.run([PY, "-W", "ignore", *map(str, args)], check=True, cwd=ROOT)


def analyse(cfg_path: Path, cfg: dict) -> dict:
    """The same steps as for the two built-in videos, driven by the generated config."""
    stem, dur = cfg_path.stem, cfg["lane_speed"]["windows"][0][1]
    tag = f"{stem}_0-{dur:g}s"
    interim = ROOT / "data" / "interim"
    det, pts, stab = (interim / f"{tag}_{k}.parquet" for k in ("detections", "points", "stab"))
    _sh(["-m", "lane_intel.detect", "--cfg", cfg_path, "--start", 0, "--duration", dur, "--out", det])
    _sh(["-m", "lane_intel.tracking", "--cfg", cfg_path, "--dets", det, "--out", pts, "--variant", cfg["tracking"]["variant"]])
    _sh(["-m", "lane_intel.stabilize", "--cfg", cfg_path, "--start", 0, "--dur", dur, "--ref-detections", det, "--out", stab])
    _sh(["-m", "lane_intel.world", "--cfg", cfg_path])
    _sh(["-m", "lane_intel.lanes", "--cfg", cfg_path, "--points", interim / f"{stem}_ground.parquet", "--write-points"])
    _sh(["-m", "lane_intel.calibrate", "--cfg", cfg_path])
    _sh(["-m", "lane_intel.lane_speed", "--cfg", cfg_path])
    return {"points": pts, "stab": stab, "ground": interim / f"{stem}_ground.parquet",
            "lanes": ROOT / cfg["lanes_file"], "speeds": ROOT / "data" / "out" / "lane_speed" / stem}


# ---------------------------------------------------------------- stage C: the result
def check_result(paths: dict, rules: dict, fps: float) -> dict:
    rr = rules["result"]
    p = pd.read_parquet(paths["points"])
    span = p.groupby("track_id").t.agg(lambda s: s.max() - s.min())
    n_tracks, med = int(span.size), float(span.median()) if span.size else 0.0
    if n_tracks < rr["min_tracks"] or med < rr["min_median_track_s"]:
        raise NotGood("C", "C1", f"Too few vehicles could be followed ({n_tracks}; each for about {med:.1f} s).",
                      "Film a busier road, closer, with the camera still.")
    lanes = json.loads(Path(paths["lanes"]).read_text())
    per_dir = pd.Series([l["direction"] for l in lanes["lanes"]]).value_counts()
    if per_dir.empty or per_dir.max() < rr["min_lanes_one_direction"] or lanes.get("boxes_in_a_lane", 0) < rr["min_boxes_in_lanes"]:
        raise NotGood("C", "C2", "The lanes could not be recognised.",
                      "Film a straight stretch of a multi-lane road, looking down on it or along it.")
    summ = json.loads((Path(paths["speeds"]) / "summary.json").read_text())
    ws = summ["whole_span"]
    ranked = [r for r in ws if not r["auxiliary"] and r["evidence_ok"] and r.get("stretch_ok", True)]
    best_dir = max(summ["stretches"].values(), key=lambda s: s["length_m"])
    if best_dir["length_m"] < rr["min_stretch_m"] or len(ranked) < rr["min_ranked_lanes_with_evidence"]:
        raise NotGood("C", "C3", "Not enough of the road is visible in every lane to compare them.",
                      "Film a longer stretch of road, with all lanes in view.")
    v = [r["v_kmh"] for r in ws if r["v_kmh"] is not None and r["v_kmh"] == r["v_kmh"]]
    if not v or max(v) > rr["max_lane_kmh"] or min(v) < rr.get("min_lane_kmh", 0.0):
        raise NotGood("C", "C4", "The distances could not be measured reliably.", "Film the road from higher up, at a steadier angle.")
    return {"tracks": n_tracks, "median_track_s": round(med, 1), "lanes": int(len(lanes["lanes"])), "summary": summ}


def answer_lines(summ: dict) -> list[str]:
    """Plain-language answer per direction, as in the app."""
    out = []
    lanes = {r["lane"]: r for r in summ["whole_span"]}
    for a in summ["whole_span_announcement"]:
        d = a["direction"]
        def pick(t):
            lid = a.get(t) if isinstance(a.get(t), str) else a.get(f"{t}_candidate")
            r = lanes.get(lid)
            if not r:
                return None
            close = "" if isinstance(a.get(t), str) else " (close call)"
            return f"lane {r['number']} ({max(0.0, r['v_kmh']):.0f} km/h){close}"
        f, s = pick("fastest"), pick("slowest")
        out.append(f"{'→ Right-moving' if d == 'right' else '← Left-moving'} traffic: fastest {f or '–'}, slowest {s or '–'}")
    return out


# ---------------------------------------------------------------- publishing to the site (docs/)
def publish(cfg_path: Path, cfg: dict, paths: dict, docs: Path, rules: dict) -> dict:
    from lane_intel.calibrate import calib_path, load_road_frame
    from lane_intel.export_viewer import ViewerVariant, build_bundle
    from lane_intel.lane_speed import viewer_payload
    vid = cfg["uploaded"]["id"]
    out = docs / "clips" / vid
    if out.exists():
        shutil.rmtree(out)
    dur = cfg["lane_speed"]["windows"][0][1]
    build_bundle(cfg, [ViewerVariant(name=cfg["tracking"]["variant"], points=pd.read_parquet(paths["points"]), label="uploaded")],
                 out, 0.0, dur, write_page=False, lanes_file=ROOT / cfg["lanes_file"], stab=pd.read_parquet(paths["stab"]),
                 lane_speed=viewer_payload(paths["speeds"]), road_frame=load_road_frame(ROOT / calib_path(cfg_path)))
    reg_path = docs / "uploads.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else []
    reg = [r for r in reg if r["id"] != vid]
    reg.insert(0, {"id": vid, "label": cfg["label"], "dir": f"clips/{vid}/", "created_utc": cfg["uploaded"]["created_utc"]})
    keep, drop = reg[:rules["site"]["keep_newest"]], reg[rules["site"]["keep_newest"]:]
    for r in drop:
        shutil.rmtree(docs / r["dir"], ignore_errors=True)
    reg_path.write_text(json.dumps(keep, indent=1))
    merge_clips_json(docs)
    return {"dropped": [r["id"] for r in drop]}


def merge_clips_json(docs: Path) -> None:
    """docs/clips.json = the built-in clips (from the site build), then uploaded clips, newest first."""
    cj = docs / "clips.json"
    clips = json.loads(cj.read_text()) if cj.exists() else []
    reg = json.loads((docs / "uploads.json").read_text()) if (docs / "uploads.json").exists() else []
    up = {r["id"] for r in reg}
    builtin = [c for c in clips if c["id"] not in up and not c.get("uploaded")]
    cj.write_text(json.dumps(builtin + [{"id": r["id"], "label": r["label"], "dir": r["dir"], "uploaded": True} for r in reg], indent=1))


# ---------------------------------------------------------------- the whole thing
def run(video: str | Path, vid: str, label: str, publish_to: Path | None, skip_detect_sample: bool = False) -> dict:
    rules = json.loads(RULES.read_text())
    res = {"id": vid, "label": label, "status": "error"}
    out_dir = ROOT / "data" / "out" / "uploads" / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = probe(video)
        res["file"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items()}
        check_file(video, info, rules)
        norm = ROOT / "data" / "raw" / "uploads" / f"{vid}.mp4"
        normalise(video, norm, rules)
        ninfo = probe(norm)
        ps = picture_stats(norm, rules)
        res["picture"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in ps.items()}
        check_picture(ps, None, rules)
        vs = None if skip_detect_sample else vehicle_sample(norm, rules, ROOT / "data" / "models" / "yolo26s.pt")
        res["vehicles"] = vs
        check_picture(ps, vs, rules)
        band = [0, ninfo["height"]]          # whole frame: a band guessed from a few samples can cut off lanes
        cfg = make_config(vid, label, norm, ninfo, band, rules)
        cfg_path = ROOT / "configs" / "uploads" / f"{vid}.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=1))
        paths = analyse(cfg_path, cfg)
        chk = check_result(paths, rules, ninfo["fps"])
        res.update(status="good", tracks=chk["tracks"], lanes=chk["lanes"], answer=answer_lines(chk["summary"]))
        if publish_to is not None:
            res["published"] = publish(cfg_path, cfg, paths, Path(publish_to), rules)
    except NotGood as e:
        res.update(status="not_good", stage=e.stage, rule=e.rule, message=e.message, tip=e.tip)
    except Exception as e:                                       # noqa: BLE001 - reported, not hidden
        res.update(status="error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--id", default=None, help="short name for the clip (default: from the file name)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--publish", default=None, help="site folder to add the clip to (e.g. docs)")
    a = ap.parse_args()
    vid = slug(a.id or Path(a.video).stem)
    res = run(a.video, vid, clean_label(a.label or vid.replace("-", " ")), Path(a.publish) if a.publish else None)
    print(json.dumps({k: v for k, v in res.items() if k != "trace"}, indent=1, default=str))
    if res["status"] == "error":
        print(res.get("trace", ""), file=sys.stderr)
    sys.exit(0 if res["status"] in ("good", "not_good") else 1)


if __name__ == "__main__":
    main()
