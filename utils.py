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

#=========  DDP utils from https://github.com/pytorch/vision/blob/4cbe71401fc6e330e4c4fb40d47e814911e63399/references/video_classification/utils.py 

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, save_path):
    if is_main_process():
        print('Saving checkpoint: {}'.format(save_path))
        torch.save(state, save_path)


def init_distributed_mode(args):

    args.distributed = False

    if args.ngpu > 1 and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # -- job started with torch.distributed.launch

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.global_rank = int(os.environ['RANK'])
        args.n_gpu_per_node = torch.cuda.device_count()

        args.job_id = args.global_rank 
        args.device = args.local_rank 

        args.n_nodes = args.world_size // args.n_gpu_per_node
        args.node_id = args.global_rank // args.n_gpu_per_node

        args.distributed = True


    elif args.ngpu > 1 and 'SLURM_PROCID' in os.environ:
        # -- SLURM JOB
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        # # job ID
        # args.job_id = os.environ['SLURM_JOB_ID']

        args.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        args.node_id = int(os.environ['SLURM_NODEID'])

        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.global_rank = int(os.environ['SLURM_PROCID'])

        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.n_gpu_per_node = args.world_size // args.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames',
            os.environ['SLURM_JOB_NODELIST']])
        args.master_addr = hostnames.split()[0].decode('utf-8')
        args.master_port = 14000
        assert 10001 <= args.master_port <= 20000 or args.world_size == 1
        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = str(args.master_addr)
        os.environ['MASTER_PORT'] = str(args.master_port)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.global_rank)


        if args.world_size >= 1:
            args.distributed = True
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(args.gpu)

    if args.distributed:
        print('| distributed init (rank {}, world_size {}): {}'.format(
            args.rank, args.world_size, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                            world_size=args.world_size, rank=args.rank)
        setup_for_distributed(args.rank == 0)

#================================================================================================

def random_frame_drop(frames, max_fraction):
    drop_prob = random.uniform(0, max_fraction)

    to_keep = np.random.uniform(size=len(frames)) > drop_prob

    return frames[to_keep]

def random_frame_duplicate(frames, max_fraction):
    dupl_prob = random.uniform(0, max_fraction)

    to_dupl = np.random.uniform(size=len(frames)) < dupl_prob

    duplicated_frames = []
    for f, dp in zip(frames, to_dupl):
        duplicated_frames.append(f)
        if dp:
            duplicated_frames.append(f)

    return np.array(duplicated_frames)