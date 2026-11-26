# Plan: "Analyse your video" through GitHub

Status: plan, approved in principle by the owner. Nothing built yet.

Goal: upload a short traffic video from the phone or the browser, get the full lane analysis in the same app
(phone and desktop) a few minutes later. No separate server: **GitHub Issues** take the upload, **GitHub Actions**
does the computing, **GitHub Pages** shows the result. Only good videos are analysed; anything else gets a clear
"not a good video, because …" reply and is not processed further.

## 1. The user's journey

1. In the app (phone or desktop) tap **"Analyse your video"**. It opens GitHub's "New analysis" issue form.
2. Attach the video (drag it in, or "Attach files" in the GitHub mobile app / mobile browser), give it a name, submit.
3. Within about a minute the issue gets a reply: "Checking your video…". The checks then decide:
   * **Not a good video** → a reply that says which rule failed, in plain words, with a tip
     ("The video is 18 s long. Please send 31–59 s."). The issue is labelled `not-good` and closed. Nothing is
     published.
   * **Good video** → "Analysing… (about 10 minutes)". When done, a reply with the answer and the link:
     "→ Right-moving traffic: fastest lane 2 (54 km/h), slowest lane 4 (31 km/h) · open: …/#clip=<id>".
     The video appears in the app's video list on both phone and desktop. Label `done`, issue closed.
4. If the analysis itself fails (a bug, not a bad video): "Something went wrong on our side", label `error`, the
   log is linked for the owner. Nothing half-finished is published.

## 2. What counts as a good video (the gate)

Every rule is checked in order; the first failure stops everything. The thresholds are written into
`configs/upload_rules.json` before any upload is processed, as for the lane-speed rules.

**Stage A: the file (seconds, before anything heavy)**

| # | Rule | Why | Reply when it fails |
|---|---|---|---|
| A1 | Only the repository owner (or a collaborator) can trigger it | public repo: strangers must not use the compute or write to the site | (no reply; ignored) |
| A2 | Exactly one video attachment, mp4 / mov / webm | | "Attach one video file (mp4, mov or webm)." |
| A3 | Size ≤ 10 MB on a free GitHub account (25 MB via "Add file") | GitHub's own limits | "The file is 14 MB; please send ≤ 10 MB (shorter, or 720p)." |
| A4 | Duration **> 30 s and < 60 s** | enough traffic for lane averages, quick to process | "The video is 18 s long. Please send 31–59 s." |
| A5 | Resolution ≥ 960 px wide; frame rate ≥ 4 fps; decodes cleanly | small vehicles must be detectable | "The video is 640 px wide; please send at least 960 px (720p)." |

**Stage B: the camera and the picture (about 1 minute, a sample of 1 frame per second)**

| # | Rule | Why | Reply when it fails |
|---|---|---|---|
| B1 | **Fixed camera.** Stabilisation finds a steady background (≥ 150 inlier features per sampled frame) and the camera moves less than 5% of the frame width over the clip | lanes and speeds need a fixed view; a hovering drone's wobble is fine | "The camera moves too much (pans or zooms). Please hold the camera still or use a drone that hovers." |
| B2 | **One continuous shot**: no cuts (no sudden full-frame change between samples) | edited clips break tracking | "The video has cuts. Please send one continuous shot." |
| B3 | **Light and sharpness**: median brightness 40–220 (of 255), sharpness above a floor | night, glare or heavy blur make detection unreliable | "The video is too dark / too blurry." |
| B4 | **Traffic visible**: on 10 sampled frames, a median of ≥ 8 vehicles, median vehicle ≥ 20 px long | otherwise there is nothing to measure | "Only 3 vehicles are visible per frame. Please film a busier road or zoom in." |

**Stage C: the result itself (after the full analysis, before publishing)**

| # | Rule | Why | Reply when it fails |
|---|---|---|---|
| C1 | ≥ 30 vehicles tracked, median time in view ≥ 2 s | enough vehicles for averages | "Too few vehicles could be followed." |
| C2 | ≥ 2 lanes found in at least one direction, ≥ 85% of vehicles inside a lane | lane comparison needs lanes | "Lanes could not be recognised (is the camera looking along the road?)." |
| C3 | Common measuring stretch ≥ 40 m and ≥ 2 lanes with enough evidence (the lane-speed rules) | | "Not enough of the road is visible in every lane." |
| C4 | Speeds plausible: every lane 0–200 km/h; the car-length scale consistent across the frame (spread ≤ 30%) | catches a wrong calibration | "The distances could not be measured reliably." |

Only a video that passes A, B and C is published. That keeps the app's list to good analyses only.

## 3. The analysis (what runs on GitHub)

One automatic command, also usable on any computer:

```
python run.py analyze --video clip.mp4 --id <short-name>
```

| Step | What it does | Today | Needed |
|---|---|---|---|
| probe | duration, size, resolution, fps (ffprobe) | – | new `autocfg.probe` |
| gate A/B | rules above | stabilise.py exists | new `quality.py` (sampling, cuts, light, quick detection) |
| auto-config | writes `configs/uploads/<id>.json`: road band (rows that contain vehicles in the sample ± margin), detector size from the width (multiple of 32, ≤ 1920 on CPU), sample rate 10 fps, tracker, `vehicle_size` calibration (lane 3.6 m, car 4.5 m), one analysis window = whole clip | made by hand per camera | new `autocfg.py` |
| detect | YOLO26s at 10 frames per second (enough for speeds, 3× faster) | detect.py | frame step option |
| track | ByteTrack, low-confidence floor 0.15, 4 s memory (the setting chosen on the 4 fps phone clip, time-based so it works at 10 fps) | tracking package | – |
| stabilise, ground, lanes, calibrate, speeds | as for the two current videos | all exist | run per uploaded config |
| gate C | rules above | – | in `quality.py` |
| publish | phone-size proxy video (1280 px, ≈ 2–4 MB), tags, meta → `docs/clips/<id>/`, entry in `docs/clips.json` | viewer_site / pages | add an uploaded-clips list |

Expected time on GitHub's free machine (4 CPU cores, no GPU) for a 45 s 1080p clip: install about 3 min (cached
afterwards: about 1 min), checks about 1 min, detection about 5–10 min, the rest about 1 min. Total **about
10–15 minutes**. The job limit is 6 hours, so there is plenty of margin.

## 4. GitHub pieces

* `.github/ISSUE_TEMPLATE/analyze.yml`: issue form "Analyse a traffic video". Fields: name; video (drag in);
  "camera is fixed" checkbox; road / country (optional); speed limit (optional, enables the free-flow check).
* `.github/workflows/analyze.yml`:
  * triggers on issues `opened` / `edited` with the `analyze` label from the form, plus a manual run
    (`workflow_dispatch`) for re-runs;
  * **runs only if the issue author is the owner or a collaborator** (author association check);
  * one analysis at a time (`concurrency`), Python and the YOLO model cached;
  * downloads the attachment only from GitHub's own attachment host, size checked first;
  * runs `python run.py analyze`; on success commits `docs/clips/<id>/`, `docs/clips.json` and `configs/uploads/<id>.json`
    to the branch GitHub Pages serves (rebase and retry if another commit landed);
  * comments on the issue (github-script) with the answer, the link, and the label `done` / `not-good` / `error`.
* Repository settings (one-time, by the owner): Actions → Workflow permissions: **Read and write**; Pages source =
  the branch the workflow commits to; labels `analyze`, `done`, `not-good`, `error`.
* App: an **"Analyse your video"** button (phone: in the video chips row; desktop: next to the video selector) that
  opens the issue form; uploaded clips show in the same video list, newest first, with an "uploaded" badge.

## 5. Privacy and size

* The repository is public, so uploaded videos are public. The form says so.
* The original upload is **not** stored in the repository: only the compressed preview the app plays, the tags and
  the results. The issue attachment stays on GitHub until the issue is deleted.
* At most the 12 newest uploaded clips are kept on the site; older ones are removed from `docs/` by the workflow, so
  the repository does not grow without limit. A clip can also be removed by closing its issue with the label `remove`.

## 6. Phases and how each is checked

| Phase | Deliverable | Done when |
|---|---|---|
| U1 | `probe`, rule file, gate A | tests: right / wrong duration, size, resolution, format |
| U2 | gate B: sampling, stabilisation check, cuts, light, quick detection | test videos: steady MiTra cut (pass); synthetic pan (fail B1); clip with a cut (fail B2); dark (fail B3); empty road (fail B4) |
| U3 | auto-config + `run.py analyze` end to end + gate C | a 45 s MiTra cut and a 45 s cut of the uploaded clip analyse with no hand-written config; results match the existing hand-configured runs (lanes, fastest/slowest) |
| U4 | publish step: proxy video, clips list, newest-12 rule, "uploaded" badge, "Analyse your video" button | site builds locally with an uploaded clip on phone and desktop |
| U5 | issue form + workflow + comments; owner-only guard | workflow syntax checked; dry run with `workflow_dispatch` on a sample file |
| U6 | first real upload together | the owner uploads a 31–59 s clip from the phone; reply, link and app entry appear; one deliberately bad clip gets the right "not good" reply |

## 7. Honest limits

* No GPU on GitHub's free machines: about 10–15 minutes per video, not seconds.
* Upload size: 10 MB through an issue on a free account. A 45 s 1080p phone clip is often larger; the form explains how
  to share a smaller one (720p, or the phone's "reduce size" option). A paid GitHub plan raises the limit.
* Without ground truth, distances rely on the assumed lane width and car length (about 4% on MiTra). The ranking of
  lanes does not depend on it.
* The road must be a straight or gently curved motorway-like section seen from above or at an angle along it. A
  junction, a roundabout or city streets with parked cars are out of scope, and gate C will usually say so.
