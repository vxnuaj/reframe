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
    weights: model files to fetch on first use — list of {path, gdrive, label}.
    Each is downloaded (with a visible progress bar) only if missing, so a user
    never has to fetch weights by hand.
    """

    name: str
    python_exe: str
    runner: str
    repo_path: str
    weights: list[dict] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
    timeout: int = 3600

    def ensure_weights(self) -> None:
        """Download any missing weights, streaming gdown's progress bar to the
        terminal. Reuses the model venv's gdown so reframe stays dep-light."""
        gdown = os.path.join(os.path.dirname(self.python_exe), "gdown")
        for w in self.weights:
            dest = os.path.join(self.repo_path, w["path"])
            if os.path.exists(dest):
                continue
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            print(f"[reframe] {self.name}: downloading {w['label']} (one-time)...", flush=True)
            # no capture_output -> gdown's progress bar shows live in the terminal
            proc = subprocess.run([gdown, w["gdrive"], "-O", dest], cwd=self.repo_path)
            if proc.returncode != 0 or not os.path.exists(dest):
                raise RuntimeError(
                    f"backend '{self.name}': failed to download {w['label']} "
                    f"(gdrive {w['gdrive']} -> {w['path']})"
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
                *self.extra_args,
            ]
            env = {**os.environ, "PYTHONPATH": self.repo_path + os.pathsep + os.environ.get("PYTHONPATH", "")}
            proc = subprocess.run(
                cmd, cwd=self.repo_path, env=env, timeout=self.timeout,
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ASD backend '{self.name}' failed (exit {proc.returncode}).\n"
                    f"cmd: {' '.join(cmd)}\nstderr tail:\n{proc.stderr[-2000:]}"
                )
            if not os.path.exists(out):
                raise RuntimeError(
                    f"ASD backend '{self.name}' produced no scores file.\nstderr tail:\n{proc.stderr[-2000:]}"
                )
            return SpeakingScores.load(out)
