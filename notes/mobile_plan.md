# Plan: a real mobile app for Lane Intel

Status: plan only, nothing built yet. The desktop Inspector stays as it is (it is the expert tool);
this is a second, phone-first app on the same data.

## 1. What is wrong today on a phone

* The phone gets the desktop Inspector squeezed into one column: about 15 panels to scroll through.
  The answer ("which lane?") and the video compete with diagnostics nobody needs on a phone
  (tracker quality, track register, break markers, keyboard hints).
* The MiTra video is a 6:1 strip. At phone width it is about 60 px tall, so it has to be zoomed,
  and panning is a plain sideways scroll with no sense of where you are on the road.
* Controls sit far from the thumb: play is under the video, zoom is in a "Layers" panel far below.
* Nothing invites touch: tapping a car works but opens a panel far down the page; there are no
  gestures, no feedback, nothing playful.

## 2. Design principles

1. **Video first.** The video is the first thing on screen and plays with one tap. Everything
   else is a layer on it or a sheet under it.
2. **Answer in one glance.** "Take lane 1 →" in big type, on the video, in the lane's colour.
3. **Thumb zone.** Every control sits in the bottom third; tap targets at least 44 px.
4. **Progressive disclosure.** First the answer, then why, then the details. The expert material
   stays in the desktop Inspector.
5. **Direct manipulation.** Tap a car, tap a lane, pinch, drag, scrub: no checkboxes.
6. **Honest.** "Close call" stays visible. The fun must never hide uncertainty.

## 3. The app, screen by screen

### 3.1 Opening (under 2 s to the first frame)
* Full-bleed video poster (first frame as an image, shown instantly) with a big ▶ in the middle.
* Over the video, a single answer chip: **"▲ Fastest: lane 1 · 42 km/h"**, in that lane's colour,
  with "close call" when it applies.
* A video switcher as swipeable cards at the top (MiTra, uploaded clip, future clips), each with
  a thumbnail and its headline answer.

### 3.2 Watch (the main screen)
* **Portrait:** the video sits at the top, sticky, cropped to about a 2.5:1 window of the road.
  * Above it, a **mini-map strip** shows the whole road with a frame you drag to pan, like a
    scrubber for space.
  * Drag on the video to pan and pinch to zoom; double-tap toggles 1× / 3×.
* **Landscape:** rotating the phone goes full screen, which suits the wide road best. The
  controls fade out and come back on a tap.
* **Bottom bar (thumb zone):**
  * a large ▶ / ❚❚ button;
  * frame-step buttons;
  * a speed toggle (0.5× / 1× / 2×);
  * a **timeline scrubber coloured by the fastest lane over time**, so you can see where the
    lead changes and jump there.
* **Lanes on the video:** each lane a soft translucent band in its own colour, labelled with
  number and km/h. The fastest band glows gently; the slowest is dimmed.

### 3.3 Tap a car → car card (bottom sheet)
* It slides up from the bottom; swipe down to close.
* Contents: class icon, **speed in km/h (large)**, lane, a "lane change" badge if it changed,
  distance travelled, time on screen.
* **Follow** button (or long-press the car): the camera keeps the car centred while playing.
* Light haptic tick when it opens.

### 3.4 Tap a lane → lane card
* That lane's band lights up and the others fade.
* The card shows:
  * the lane's speed with its range as a simple bar;
  * rank ("2nd of 3");
  * cars counted, trucks share, lane changes in / out;
  * a small sparkline of its speed over time, playhead synced with the video.

### 3.5 Lanes tab: the race
* Lanes as **horizontal bars racing over time** (an animated ranking that follows the video
  clock). The leader gets a crown; ties are shown as ties.
* The headline sentence underneath: "Lane 1 was fastest 78% of the time; lane 2 was slowest."
* "Why we're sure / not sure": one sentence plus a small interval bar, with a link to the method.

### 3.6 Fun layer (optional, honest)
* **Guess the lane:** before showing the answer, ask "Which lane do you think is fastest?". Then
  play the race and reveal, with a confetti burst only when the answer is clear, and "too close to
  call" otherwise.
* **Ghost race:** pick two cars and see which one got further (two dots racing on the mini-map).
* **Share card:** one image with the video still, the answer and the date, to share or post.

### 3.7 About tab
* The "How it works" story told as five swipeable cards (detect → track → lanes → metres →
  verdict), each with one picture. A link to the full desktop page.

## 4. Navigation

Bottom tab bar with 3 tabs: **Watch · Lanes · About**. The car and lane cards are bottom sheets
over Watch, not separate pages. The back gesture closes a sheet first, then the tab.

## 5. Technical plan

| Topic | Decision |
|---|---|
| Where | New `app/mobile/index.html` (plain HTML/JS, no framework, like the Inspector). `viewer_site.py` also writes `m/index.html`. Small screens are sent to it, with a "desktop view" link both ways. |
| Data | Same `clips.json` and `clips/<id>/meta.json` (lane speeds, lanes, tracks). Shared loading and drawing code moved into one small `common.js` used by both apps. |
| Video | A phone proxy per clip: 1280 px wide, a lower bitrate, and a poster image (`poster.jpg`). MiTra drops from 11 MB to about 3–4 MB. Streams with range requests instead of a full blob download, so playback starts at once. |
| Tags | Split `tags_*.json` into 5 s chunks, loaded around the playhead (MiTra tags are 2.9 MB today). |
| Gestures | Pointer Events for drag, pinch and double-tap; `touch-action` set so the page does not fight the video. |
| Feel | 60 fps canvas overlay; `prefers-reduced-motion` turns animations off; haptics through `navigator.vibrate` where supported. |
| Install | PWA: manifest, icons, offline cache of the app shell and the last opened clip; safe-area insets for notches. |
| Theme | Follows the phone; dark looks best over video. |

## 6. Phases

| Phase | Deliverable | Done when |
|---|---|---|
| M1 · Watch | Mobile page: poster, one-tap play, answer chip, bottom bar, portrait pan/zoom with mini-map, landscape full screen | Tested on iPhone and Pixel emulation: first frame under 2 s on a fast-3G throttle; play is one tap; no page sideways scroll |
| M2 · Touch | Car card and lane card as bottom sheets, Follow, lane bands in colour | Every tap target ≥ 44 px; sheets close by swipe and back gesture |
| M3 · Lanes | Racing bars synced to the video, headline sentence, honest uncertainty | Bars match `lane_speed` numbers exactly (automated check) |
| M4 · Speed | Phone video proxies, posters, chunked tags, PWA install and offline | MiTra opens with under 1 MB before the first frame; works offline after one visit |
| M5 · Fun | Guess the lane, ghost race, share card | Reveal respects "close call" (test) |
| M6 · Polish | About cards, reduced motion, accessibility pass, dark default | Lighthouse mobile ≥ 90 (performance, accessibility) |

**User test after M2 and M5:** 3–5 people, phone in hand, no instructions. Targets:
* find the fastest lane in under 10 s;
* play and pause without help;
* open a car's speed;
* say whether the result is certain.

## 7. Decisions (answered)

1. Separate mobile page (recommended) or one responsive page for both?
2. Fun layer (guess / ghost race / share): in or out, and which first?
3. Install as an app (PWA) needed?
4. Are lower-quality phone videos acceptable (much faster loading)?
5. Landscape full screen as the main way to watch MiTra, with portrait as the overview?

### Answers (owner)

1. **Same address.** One URL (https://noamari.github.io/BestLaneDetactor/). The page chooses the phone UI
   when the device is a phone: touch pointer (`pointer: coarse`) and a small screen (`max-width: 900px`).
   Everything else gets the desktop Inspector. A small "Desktop view / Phone view" switch is remembered in
   localStorage, so anyone can override the automatic choice. Implementation: `index.html` loads either the
   phone app (`m.js` + `m.css`) or the Inspector at start-up; the data files are shared.
2. **Fun: yes, with taste.** Guess-the-lane and the lane race first. Quiet motion, no sounds,
   confetti only for a clear answer and only once. Everything stays honest about "close call".
3. **Website only** for now (no install / PWA). Phone video proxies and chunked tags stay (speed).
4. **Lower-quality phone video: yes.**
5. **Wide road on a phone: "auto-framed portrait" plus "rotate for panorama"** (see below).

### 5. The chosen solution for the wide road video

The MiTra road is 6:1. A phone in portrait is 1:2. No single framing shows it well, so the app frames it
for you and lets you take over at any time:

* **Portrait (default):** the video fills the top ~45% of the screen, zoomed so cars are big. A
  **virtual camera** pans smoothly over the road:
  * Playing with no touch, it **follows a car in the fastest lane** ("Following a car in lane 1 ·
    42 km/h", a small label). This tells the story by itself: you ride along in the winning lane. When
    that car leaves, it picks the next one.
  * Tap any car to follow it instead; tap a lane to follow that lane's lead car.
  * Drag to pan by hand: the camera stops following ("Free view", with a ⟲ button to resume).
  * A **mini-map strip** above the video shows the whole road, with the camera window as a frame;
    tap or drag on it to jump.
* **Landscape:** rotating the phone shows the **whole road**, full screen, at once (the panorama). The
  controls fade and come back on tap. A one-time hint suggests rotating ("↻ Turn your phone to see the
  whole road").
* The same logic serves the uploaded clip (2.3:1): it needs less zoom, so the camera moves less.

Why this is best for both UI and UX:
* nothing to configure, because the video tells a story the moment you press play;
* the cars stay large enough to tap with a thumb;
* the whole road is always one glance away, on the mini-map or by rotating;
* following a car makes the speed difference between lanes something you *feel*. Following a car in
  lane 1 and then one in lane 3 is the answer, shown rather than stated.

## 8. Built: touch model (v2)

After the first phone test ("touch feels sticky") the interaction layer was rebuilt on standard mobile patterns:

| Problem in v1 | v2 |
|---|---|
| The video took every touch: a vertical swipe on it did not scroll the page | `touch-action: pan-y`: vertical always scrolls, horizontal pans the road |
| Any 8 px drag switched to "Free view" for good | Panning is temporary: back to the car 3.5 s after the last touch (while playing); "⟲ Back to the car" anytime |
| Taps waited 260 ms for a possible double-tap; a missed car paused the video | Instant taps; 18 px touch slop around cars; lane tags open lanes; empty road = play/pause with a visible flash |
| The video scrolled away under the cards | Pinned player (video, road map, transport) with the content scrolling beneath |
| Panning felt dead | Momentum and soft edges; two-finger pinch zoom; ⤢ / ⊕ button for whole road / zoom |
| The card fought the page | Bottom sheet with snap points (half, full, closed) by dragging the handle or pulling down at the top; tapping the video closes it |

Plus: a one-time swipe hint, haptic ticks where supported, "Guess the fastest lane" (confetti only for a clear answer),
About cards with page dots, reduced-motion support. Checked with real touch events on an emulated iPhone 13
(vertical swipe scrolls, horizontal swipe pans with momentum and returns, tap opens car card, sheet snaps, pinch zooms).
