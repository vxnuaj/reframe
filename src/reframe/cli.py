from __future__ import annotations

import argparse
import json
import sys

from .presets import DEFAULT_PRESET, PRESETS


def _coerce(v: str):
    """Best-effort scalar type for a CLI --set value: int, then float, else str."""
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reframe", description="Reframe a video to a subject-tracking crop path (and optionally render it).")
    p.add_argument("video")
    p.add_argument("-o", "--out", default=None, help="crop-path JSON output (default: <video>.crop.json)")
    p.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET)
    p.add_argument("--aspect", default="9:16", help="target aspect W:H (default 9:16)")
    p.add_argument("--no-face", action="store_true", help="skip MediaPipe face refinement")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
        help="override a preset field without editing code, e.g. "
        "--set max_step_x=4 --set deadzone=0.06 --set switch_boost=0 (repeatable)",
    )
    p.add_argument(
        "--asd", default=None, metavar="MODEL",
        help="active-speaker model for the speaker cue (e.g. lr-asd, talknet); "
        "runs the model in its own env and replaces the built-in lip heuristic",
    )
    # full render
    p.add_argument("--render", default=None, metavar="OUT.mp4", help="also render the final reframed video")
    p.add_argument("--out-height", type=int, default=1920, help="render height (default 1920)")
    p.add_argument("--crf", type=int, default=20, help="render quality (lower = better, default 20)")
    p.add_argument("--x264-preset", default="veryfast", help="ffmpeg encode preset (default veryfast)")
    p.add_argument("--no-audio", action="store_true", help="render without the source audio")
    args = p.parse_args(argv)

    aspect = tuple(int(x) for x in args.aspect.split(":"))
    overrides = {}
    for kv in args.overrides:
        key, _, val = kv.partition("=")
        overrides[key.strip()] = _coerce(val.strip())

    # lazy so `reframe --help` works without the ml extra installed
    from .pipeline import analyze_video

    backend = None
    if args.asd:
        from .asd import get_backend

        backend = get_backend(args.asd)
        print(f"active-speaker model: {args.asd} ({backend.python_exe})")

    path = analyze_video(
        args.video, preset=args.preset, aspect=aspect, use_face=not args.no_face,
        asd_backend=backend, overrides=overrides or None,
    )

    out = args.out or f"{args.video}.crop.json"
    with open(out, "w") as f:
        json.dump(path.to_dict(), f, indent=2)
    print(f"wrote {out} ({len(path.keyframes)} keyframes, preset={args.preset})")

    if args.render:
        from .render import render_video

        render_video(
            path, args.video, args.render,
            out_height=args.out_height, crf=args.crf, preset=args.x264_preset, with_audio=not args.no_audio,
        )
        print(f"rendered {args.render}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
