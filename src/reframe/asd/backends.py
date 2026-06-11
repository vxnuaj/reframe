"""Registry of concrete ASD backends (the hot-swap surface).

Each entry builds a backend pointed at a model's repo + its own venv. Locally these
are SubprocessASDBackends (run the model's runner in its venv); in production a
backend could instead be an HttpASDBackend to a deployed service — same contract.
Repo locations default to the sibling clones under `comet/` but are overridable.
"""

from __future__ import annotations

import os

from .base import SubprocessASDBackend

_ASD_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNNERS = os.path.join(_ASD_DIR, "runners")
# comet/reframe/src/reframe/asd -> up 4 -> comet/
_COMET = os.path.abspath(os.path.join(_ASD_DIR, "..", "..", "..", ".."))

# Google-Drive ids for the model weights (auto-fetched on first use). `arg` is the
# runner flag that tells the model where to load it from, so a custom weights_dir
# actually takes effect (not just where it downloads).
_S3FD = {"path": "model/faceDetector/s3fd/sfd_face.pth", "gdrive": "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt",
         "label": "S3FD face detector (~90MB)", "arg": "--s3fd-weight"}
_TALKSET = {"path": "pretrain_TalkSet.model", "gdrive": "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea",
            "label": "TalkNet weights (~63MB)", "arg": "--weight"}


def _subprocess(name, repo, runner, repo_path, python_exe, weights, **kw):
    repo_path = repo_path or os.path.join(_COMET, repo)
    python_exe = python_exe or os.path.join(repo_path, ".venv", "bin", "python")
    return SubprocessASDBackend(
        name=name,
        python_exe=python_exe,
        runner=os.path.join(_RUNNERS, runner),
        repo_path=repo_path,
        weights=weights,
        **kw,
    )


def lr_asd(repo_path: str | None = None, python_exe: str | None = None, **kw) -> SubprocessASDBackend:
    # LR-ASD ships its own model weight in-repo; only the S3FD detector is fetched.
    return _subprocess("lr-asd", "LR-ASD", "lrasd_runner.py", repo_path, python_exe, [_S3FD], **kw)


def talknet(repo_path: str | None = None, python_exe: str | None = None, **kw) -> SubprocessASDBackend:
    return _subprocess("talknet", "TalkNet-ASD", "talknet_runner.py", repo_path, python_exe, [_S3FD, _TALKSET], **kw)


# name -> factory. fast-asd slots in here next (same family, productionised TalkNet).
BACKENDS = {
    "lr-asd": lr_asd,
    "talknet": talknet,
}


def get_backend(name: str, **kw) -> SubprocessASDBackend:
    if name not in BACKENDS:
        raise KeyError(f"unknown ASD backend '{name}'. available: {sorted(BACKENDS)}")
    return BACKENDS[name](**kw)
