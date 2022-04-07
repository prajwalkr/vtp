import torch, os, pickle, cv2, random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.autograd import Variable
import pandas as pd
from utils import random_frame_drop, random_frame_duplicate

from torch.utils import data as data_utils

import numpy as np
from config import load_args

args = load_args()

from decord import VideoReader
from glob import glob

# GPU-based augmentation utils:
import kornia.augmentation as K
import torchvision.transforms.functional as TF

from config import load_args, pad_token, bos_token, eos_token, unk_token
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir='checkpoints/tokenizers', 
		bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token, 
		use_fast=True)

class AugmentationPipeline(torch.nn.Module):
	def __init__(self, args):
		super(AugmentationPipeline, self).__init__()
		self.frame_size = args.frame_size
		self.img_size = args.img_size
		self.crop_offset = args.rand_crop

		self.affine = K.VideoSequential(
							K.RandomHorizontalFlip(),
							K.RandomRotation(degrees=args.rot),
							same_on_frame=True, data_format="BCTHW"
						)

	##### deterministic test-time augmentation
	def horizontal_flip(self, frames):
		return torch.flip(frames, dims=[4])

	##### random augmentations during training
	def random_crop(self, frames):
		size = self.frame_size - self.crop_offset

		x = random.randint(0, frames.size(4) - size)
		y = random.randint(0, frames.size(3) - size)

		return frames[:, :, :, y : y + size, x : x + size]

	def center_crop(self, frames):		
		crop_x = (frames.size(3) - self.img_size) // 2
		crop_y = (frames.size(4) - self.img_size) // 2
		faces = frames[:, :, :, crop_x:crop_x + self.img_size , 
							crop_y:crop_y + self.img_size]

		return faces

	def random_brightness(self, faces, factor=.1):
		factor = (2. * factor) * torch.rand(faces.size(0), 1, 
						faces.size(2), 1, 1).to(faces) + (1 - factor)
		faces = torch.clamp(faces * factor, 0, 1)
		return faces

	def forward(self, x, train_mode):
		# x : (B, C, T, H, W)
		if train_mode:
			x = self.affine(x)
			x = self.random_crop(x)

		faces = self.center_crop(x)
		if train_mode:
			faces = self.random_brightness(faces)

		return faces

class VideoDataset(object):
	def __init__(self, args, inference=False, mode='train'):
		self.img_size = args.img_size
		self.frame_size = args.frame_size
		self.normalize = args.normalize
		print('Normalize face: {}'.format(bool(self.normalize)))

		self.vocab_size = tokenizer.vocab_size

		if inference:
			return

	def input_transform(self, frames, augment=True):
		# frames := (T, H, W, C)

		if augment:
			# temporal augmentations
			frames = random_frame_drop(frames, max_fraction=0.03)
			frames = random_frame_duplicate(frames, max_fraction=0.03)

		if self.normalize: faces = frames / 255.
		else: faces = frames

		return faces 

	def read_video(self, fpath, start=0, end=None):
		start = max(start - 4, 0)
		if end is not None:
			end += 4 # to read til end + 3
		else:
			end = 1000000000000000000000000000000000 # some large finite num

		with open(fpath, 'rb') as f:
			video_stream = VideoReader(f, width=self.frame_size, height=self.frame_size)

			end = min(end, len(video_stream))

			frames = video_stream.get_batch(list(range(start, end))).asnumpy()

		return frames
