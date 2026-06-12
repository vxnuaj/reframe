"""ASD backend interface + a subprocess transport.

A backend turns a video into a `SpeakingScores`. The interface is tiny on purpose
(`run(video) -> SpeakingScores`) so any model fits behind it. `SubprocessASDBackend`
is the local transport for the bake-off: it shells out to a model's *runner script*
executed by that model's own venv python (the repo on PYTHONPATH), so the model's
pinned deps never touch reframe's. The runner emits the scores contract as JSON;
we load it back. A production `HttpASDBackend` would implement the same `run`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .contract import SpeakingScores


@runtime_checkable
class ASDBackend(Protocol):
    name: str

    def run(self, video_path: str) -> SpeakingScores: ...


@dataclass
class SubprocessASDBackend:
    """Run a model's runner script in its own environment and read back the contract.

    runner: a script (living under asd/runners/) that imports the model from
    `repo_path`, scores the video, and writes the SpeakingScores JSON to --out.
    python_exe: the model venv's python (where the model's deps are installed).
    weights: model files to fetch on first use — list of {path, gdrive, label,
    arg?}. Each is downloaded (with a visible progress bar) only if missing.
    weights_dir: where weights live and download to (default: the repo's own
    paths). offline: never download; use weights already present and error if a
    required one is missing (for baked/mounted weights with no runtime network).
    """

    name: str
    python_exe: str
    runner: str
    repo_path: str
    weights: list[dict] = field(default_factory=list)
    weights_dir: str | None = None
    offline: bool = False
    extra_args: list[str] = field(default_factory=list)
    timeout: int = 3600

    def _dest(self, w: dict) -> str:
        """Where a weight lives: weights_dir (by basename) if set, else repo/path."""
        if self.weights_dir:
            return os.path.join(self.weights_dir, os.path.basename(w["path"]))
        return os.path.join(self.repo_path, w["path"])

    def _weight_args(self) -> list[str]:
        """Tell the runner where to load each weight from (so weights_dir works)."""
        args: list[str] = []
        for w in self.weights:
            if w.get("arg"):
                args += [w["arg"], self._dest(w)]
        return args

    def ensure_weights(self) -> None:
        """Make sure every weight is present. Missing ones download (streaming
        gdown's progress bar) unless offline, in which case it's an error."""
        gdown = os.path.join(os.path.dirname(self.python_exe), "gdown")
        for w in self.weights:
            dest = self._dest(w)
            if os.path.exists(dest):
                continue
            if self.offline:
                raise RuntimeError(
                    f"backend '{self.name}': {w['label']} not found at {dest} and offline=True. "
                    f"put the file there, or set offline=False to download it."
                )
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            print(f"[reframe] {self.name}: downloading {w['label']} (one-time) -> {dest}", flush=True)
            # no capture_output -> gdown's progress bar shows live in the terminal
            proc = subprocess.run([gdown, w["gdrive"], "-O", dest], cwd=self.repo_path)
            if proc.returncode != 0 or not os.path.exists(dest):
                raise RuntimeError(
                    f"backend '{self.name}': failed to download {w['label']} "
                    f"(gdrive {w['gdrive']} -> {dest})"
                )
            print(f"[reframe] {self.name}: {w['label']} ready.", flush=True)

    def run(self, video_path: str) -> SpeakingScores:
        self.ensure_weights()
        video_path = os.path.abspath(video_path)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "scores.json")
            cmd = [
                self.python_exe, self.runner,
                "--video", video_path,
                "--out", out,
                "--model", self.name,
                *self._weight_args(),
                *self.extra_args,
            ]
            env = {**os.environ, "PYTHONPATH": self.repo_path + os.pathsep + os.environ.get("PYTHONPATH", "")}
            # Stream the model's stdout/stderr to ours (don't capture) so its per-step
            # progress shows up live in the logs — the run is minutes long on CPU, so
            # silence is the enemy. The result comes from the --out file, not stdout.
            proc = subprocess.run(cmd, cwd=self.repo_path, env=env, timeout=self.timeout)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ASD backend '{self.name}' failed (exit {proc.returncode}); see the model logs above. "
                    f"cmd: {' '.join(cmd)}"
                )
            if not os.path.exists(out):
                raise RuntimeError(
                    f"ASD backend '{self.name}' produced no scores file; see the model logs above."
                )
            return SpeakingScores.load(out)
