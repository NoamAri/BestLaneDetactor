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

## 7. Decisions for you

1. Separate mobile page (recommended) or one responsive page for both?
2. Fun layer (guess / ghost race / share): in or out, and which first?
3. Install as an app (PWA) needed?
4. Are lower-quality phone videos acceptable (much faster loading)?
5. Landscape full screen as the main way to watch MiTra, with portrait as the overview?
