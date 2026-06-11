#!/usr/bin/env python
"""LR-ASD backend runner. Thin: provides the model load + per-track scoring; the
shared TalkNet-family pipeline lives in _common. Runs in the LR-ASD venv with the
repo on PYTHONPATH and cwd = repo root."""

import math

import numpy as np
import torch

import _common
from loss import lossAV
from model.Model import ASD_Model


def load_model(weight, dev):
    sd = torch.load(weight, map_location=dev)
    model = ASD_Model().to(dev)
    la = lossAV().to(dev)
    model.load_state_dict(_common.split_state(sd, "model"))
    la.load_state_dict(_common.split_state(sd, "lossAV"))
    model.eval()
    la.eval()
    return model, la


def score_track(ctx, vfeat, afeat, dev):
    model, la = ctx
    length = min((afeat.shape[0] - afeat.shape[0] % 4) / 100, vfeat.shape[0])
    if length < 1:
        return np.zeros(vfeat.shape[0])
    afeat = afeat[: int(round(length * 100)), :]
    vfeat = vfeat[: int(round(length * _common.PROC_FPS)), :, :]
    all_score = []
    for duration in (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6):
        batch = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batch):
                ia = torch.FloatTensor(afeat[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).to(dev)
                iv = torch.FloatTensor(vfeat[i * duration * 25:(i + 1) * duration * 25, :, :]).unsqueeze(0).to(dev)
                out = model.forward_audio_visual_backend(
                    model.forward_audio_frontend(ia), model.forward_visual_frontend(iv))
                scores.extend(la.forward(out, labels=None))
        all_score.append(scores)
    return np.mean(np.array(all_score), axis=0)


if __name__ == "__main__":
    _common.run(load_model, score_track, default_weight="weight/finetuning_TalkSet.model")
