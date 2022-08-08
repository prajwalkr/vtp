import os, torch, random, scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import subprocess
import torch.distributed as dist

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size=512, factor=1, warmup=8000, steps=0, 
                optimizer_state=None, reset_optimizer=False):
        self.optimizer = optimizer

        self._step = steps
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = self.rate() if steps != 0 else 0
        
    def update_lr(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def load_state_dict(self, state_dict, steps):
        self.optimizer.load_state_dict(state_dict)
        self._step = steps
        self._rate = self.rate()

    def state_dict(self):
        return self.optimizer.state_dict()

def load(model, ckpt_path, ignore_face_encoder=False, device='cuda',
            face_encoder_ckpt=None, strict=True, only_face_encoder=False):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if face_encoder_ckpt is not None:
        face_encoder_ckpt = torch.load(face_encoder_ckpt, map_location=device)

    s = checkpoint["state_dict"]
    new_s = {}
    if face_encoder_ckpt is not None:
        fs = face_encoder_ckpt["state_dict"]
        for k, v in fs.items():
            if 'face_encoder' in k:
                new_s[k.replace('module.', '')] = v

    for k, v in s.items():
        if ignore_face_encoder and 'face_encoder' in k: continue
        new_s[k.replace('module.', '')] = v

    model.load_state_dict(new_s, strict=strict)

    optimizer_state = checkpoint["optimizer"]
    if not ignore_face_encoder and face_encoder_ckpt is not None:
        vis_optimizer_state = face_encoder_ckpt["vis_optimizer"]
    elif "vis_optimizer" in checkpoint:
        vis_optimizer_state = checkpoint["vis_optimizer"]
    else:
        vis_optimizer_state = None
    epoch = checkpoint['global_epoch']

    if 'steps' in checkpoint:
        return model, optimizer_state, vis_optimizer_state, epoch, checkpoint['steps']

    return model, optimizer_state, vis_optimizer_state, epoch

def load_cnn(cnn_module, ckpt_path, device='cuda'):
    checkpoint = torch.load(ckpt_path, map_location=device)

    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        if 'conv' not in k: continue
        new_s[k.replace('module.', '').replace('face_encoder.', '')] = v

    cnn_module.load_state_dict(new_s, strict=False)
    return cnn_module

def levenshtein(a, b):
  """Calculates the Levenshtein distance between a and b.
  The code was taken from: http://hetland.org/coding/python/levenshtein.py
  """
  n, m = len(a), len(b)
  if n > m:
    # Make sure n <= m, to use O(min(n,m)) space
    a, b = b, a
    n, m = m, n
  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)
  return current[n]
