import torch, os, pickle, cv2, random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.autograd import Variable
import pandas as pd

import numpy as np
from config import load_args

args = load_args()

from decord import VideoReader
from glob import glob

from config import load_args, pad_token, bos_token, eos_token, unk_token
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir='checkpoints/tokenizers', 
		bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token, 
		use_fast=True)

class AugmentationPipeline(torch.nn.Module):
	def __init__(self, args):
		super(AugmentationPipeline, self).__init__()
		self.img_size = args.img_size

	##### deterministic test-time augmentation
	def horizontal_flip(self, frames):
		return torch.flip(frames, dims=[4])

	def center_crop(self, frames):		
		crop_x = (frames.size(3) - self.img_size) // 2
		crop_y = (frames.size(4) - self.img_size) // 2
		faces = frames[:, :, :, crop_x:crop_x + self.img_size , 
							crop_y:crop_y + self.img_size]

		return faces

	def forward(self, x, flip=False):
		x = x.permute(0, 4, 1, 2, 3)
		
		# x : (B, C, T, H, W)
		faces = self.center_crop(x)

		return faces

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

class VideoDataset(object):
	def __init__(self, args):
		self.frame_size = args.frame_size
		self.normalize = args.normalize
		print('Normalize face: {}'.format(bool(self.normalize)))

		self.vocab_size = tokenizer.vocab_size

	def read_video(self, fpath, start=0, end=None):
		start = max(start - 4, 0)
		if end is not None:
			end += 4 # to read til end + 3
		else:
			end = 1000000000000000000000000000000000 # some large finite num

		with open(fpath, 'rb') as f:
			video_stream = VideoReader(f, width=self.frame_size, height=self.frame_size)

			end = min(end, len(video_stream))

			frames = video_stream.get_batch(list(range(start, 
							end))).asnumpy().astype(np.float32)

		frames /= 255.

		return frames

	def to_ids(self, text):
		return [t - 1 for t in tokenizer('<bos> ' + text + ' <eos>')['input_ids']]
		
	def to_tokens(self, ids):
		return tokenizer.convert_tokens_to_string(\
				tokenizer.convert_ids_to_tokens([t + 1 for t in ids])).replace(\
				'<bos> ', '').replace(' <eos>', 
				'').strip().lower().replace('[cls] ', '').replace(' [sep]', '').strip()
		