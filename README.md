# reframe

scene-aware video reframing. it watches a horizontal video, tracks the main
subject through cuts, and spits out a smooth crop path (where the crop window
should sit over time, as it pans and zooms). by default it does not touch pixels.
you get the crop path as json and apply it with a renderer (there is a bundled
ffmpeg one, or plug in your own).

why deciding is split from rendering: the crop path is plain json, so it is small,
inspectable, and cacheable. you can render it with whatever you already have, feed
it into another tool, or hand-edit a keyframe before rendering. the analysis runs
once; rendering is a separate, cheap step you can repeat at any size or quality.

## install (uv)

```bash
uv sync               # core only, pure python
uv sync --extra ml    # + detectors (yolo, mediapipe, opencv, scenedetect)
```

from another project: `uv add reframe` or `uv add "reframe[ml]"`, or
`pip install "reframe[ml]"`.

## core vs ml

the package splits in two:

- **core** (no extras): pure python, zero native deps. the camera/smoothing,
  ranker, tracker, presets, and path logic. it takes per-frame bounding boxes
  (from wherever) and turns them into a crop path. this is the brain.
- **ml** extra: the vision stack that looks at the pixels and produces those
  boxes (yolo, mediapipe, opencv, scenedetect). heavy, think torch and native
  libs, roughly a gb. this is the eyes.

so without ml you have the decision logic but nothing to generate boxes from a raw
mp4. with ml you get bundled local detectors that do it for you.

why it is split: you might already have boxes. if you feed detections from a cloud
vision api, you only need core and skip the whole torch/yolo install. if you want
it local and offline, install ml and use `YoloDetector`. same brain either way,
swappable eyes.

what works in each mode:

| | core only | with ml |
| --- | --- | --- |
| `build_crop_path`, camera, ranker, presets, json out | yes | yes |
| `ReplayDetector` (feed your own boxes) | yes | yes |
| `YoloDetector` (boxes straight from the video) | no | yes |
| `scene_starts` (scenedetect) | falls back to one scene | yes |
| cli end to end (`reframe video.mp4`) | no | yes |
| ffmpeg render | no | yes |

## use

cli (needs the ml extra):

```bash
reframe input.mp4 --preset talking_head --aspect 9:16 -o out.crop.json
reframe input.mp4 --render out.mp4        # also render the finished video (with audio)
reframe input.mp4 --asd lr-asd --render out.mp4   # use an active-speaker model (see below)
```

it always writes the crop path json; `--render` additionally produces the finished
mp4. so you can do either or both.

library:

```python
# crop path only (core, detector-agnostic, feed it boxes from anywhere)
from reframe import build_crop_path, VideoMeta
meta = VideoMeta(fps=30, width=1920, height=1080, source="clip.mp4")
path = build_crop_path(meta, frames, scene_starts=[0, 142, 410], preset_name="talking_head")
data = path.to_dict()

# full pipeline to a finished video (needs ml + ffmpeg)
from reframe import reframe_video, analyze_video, render_video
reframe_video("input.mp4", "out.mp4", preset="talking_head")   # analyze + render in one call
# or split it: path = analyze_video("input.mp4"); render_video(path, "input.mp4", "out.mp4")
```

## active speaker detection (optional)

in a multi-person shot, "who is the subject" is really "who is talking". the
built-in cue uses lip motion, which is noisy on small or low-res faces. you can
instead use a real audio-visual active-speaker model.

the models are hot-swappable behind one interface: each runs in its own
environment and emits the same per-frame speaking-score contract, so swapping one
for another never touches the camera, ranker, or path. supported names:

- `lr-asd`: lightweight (~1M params), cpu-friendly, nearly as accurate
- `talknet`: heavier, sharper separation

```bash
reframe input.mp4 --asd lr-asd
```

```python
from reframe import analyze_video, get_backend
path = analyze_video("input.mp4", asd_backend=get_backend("lr-asd"))
```

weights download automatically on first use, with a progress bar. they run on cuda
when available, otherwise cpu/mps. each backend points at its model installed in
its own venv (see `reframe/asd/backends.py`); for a deployed setup you can back the
same interface with an http service instead.

## tuning

`talking_head` is the only bundled preset, and the only one that's been tuned and
tested. it's the default. to support another content type (sports, pets, whatever),
add a `Preset` to `PRESETS` in `presets.py` (there's a note there) and select it by
name.

you do not have to edit `presets.py` to change a preset's feel. override fields right
from the call. overrides start from the named preset and patch only the fields you
name; an unknown field errors instead of being silently ignored:

```bash
reframe input.mp4 --set max_step_x=4 --set deadzone=0.06 --set switch_boost=0
```

```python
from reframe import analyze_video, resolve_preset
analyze_video("input.mp4", overrides={"max_step_x": 4.0, "deadzone": 0.06})
preset = resolve_preset("talking_head", {"max_step_x": 4.0})   # or build one and reuse
analyze_video("input.mp4", preset=preset)
```

### preset knobs (the camera feel)

- `classes`: which detection classes count as subjects, e.g. `("person",)`.
- `min_zoom` / `max_zoom`: punch-in range. 1.0 is the full-height crop for the target
  aspect; higher is tighter. `max_zoom` caps how tight it punches in.
- `max_step_x` / `max_step_y`: max pan speed in px per frame (x horizontal, y
  vertical). higher tracks faster and snappier, lower is calmer and slower. y sits
  lower because vertical bounce reads worse than horizontal.
- `motion_response`: spring stiffness toward the subject. higher catches up quicker,
  lower is lazier and smoother.
- `motion_damping`: how much pan velocity carries frame to frame. higher is heavier
  and smoother, lower is snappier.
- `target_alpha`: how fast the aim eases toward the raw detection. higher follows
  more eagerly, lower adds lag and smoothing to the goal itself.
- `deadzone`: fraction of frame size the subject can drift before the camera moves at
  all. higher holds stiller (ignores small movement), 0 always tracks.
- `switch_boost`: how a subject change is handled. 0 is a hard cut (instant); a
  positive number is that many frames of a fast eased whip to the new subject.
- `zoom_response` / `zoom_damping` / `zoom_alpha`: the same response, damping, and
  easing knobs, but for the zoom (punch-in) axis.

### selection knobs (who to follow)

these live on `SubjectSelector`, not the preset. build one and pass it to
`build_crop_path(selector=...)`:

- `turn_hold_frames`: how long a challenger must be the clear speaker before the lock
  switches. higher resists switching (steadier), lower follows turns sooner.
- `switch_margin`: how much a non-speaking challenger (bigger, more central, moving)
  must beat the locked subject to take it.
- `speaker_floor` / `speaker_switch_margin`: how clearly a challenger must be talking,
  and by how much more than the current subject, to take over.
- `min_hold_frames`: minimum frames on a subject before any switch is allowed.
- `reassoc_radius`: fraction of frame width within which a dropped or re-numbered
  track is treated as the same person (bridges tracker id churn).
- `warmup_frames`: frames at a scene start to acquire freely before committing.
- `motion_alpha` / `speaker_alpha`: smoothing windows for the motion and speaker
  signals.

## the output

```jsonc
{
  "version": 1,
  "fps": 30,
  "width": 1920, "height": 1080,
  "target_aspect": [9, 16],
  "preset": "talking_head",
  "keyframes": [
    { "t": 0.0, "cx": 612.4, "cy": 540.0, "zoom": 1.12 }
  ]
}
```

`cx`/`cy` is the crop center in source pixels, `zoom` is the punch-in (1.0 = the
full-height crop for the target aspect). a renderer interpolates between keyframes.

## how it works

```
video -> scene_starts (pyscenedetect)
      -> detector (yolo+bytetrack | cloud | replay) -> per-frame boxes
      -> speaker cue (lip motion, or an active-speaker model) -> who is talking
      -> selector (rank + lock/hold/switch hysteresis) -> the subject
      -> focus + desired zoom
      -> camera (damped spring, snaps at cuts) -> crop center + zoom
      -> crop path
```

- `smooth.py` is the virtual camera. a damped spring per axis (ease the target,
  then advance with capped, damped velocity), weighted by detection confidence,
  with a deadzone so slight movement does not pan the frame. this is where it
  feels smooth instead of jittery.
- `rank.py` picks which subject to follow with a small linear scorer, and holds
  onto it so it does not flicker between people (no llm, no sam). a real turn
  change cuts; track-id churn does not.
- `presets.py` has the tuned knobs (just `talking_head` today; add more as needed).
- `asd/` is the active-speaker backends and the scores contract.
- `path.py` ties it together, cuts the camera at scene and speaker changes, and
  holds the last framing when the subject is briefly lost.
- `detect.py` / `scenes.py` are the swappable ml backends (lazy imported).
- `render.py` applies a crop path to the source and writes a finished mp4 (ffmpeg,
  with audio). `pipeline.py` has the one-shot `analyze_video` / `reframe_video`
  wrappers. all optional, ml only.
