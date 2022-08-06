import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import load_args
from models import builders

from dataloader import AugmentationPipeline, VideoDataset
from utils import load

from glob import glob

from torch.cuda.amp import autocast

def init(args):
	data_util = VideoDataset(args)

	model = builders[args.builder](data_util.vocab_size + 1, args.feat_dim, N=args.num_blocks, 
							d_model=args.hidden_units, 
							h=args.num_heads, dropout=args.dropout_rate)

	return model.to(args.device).eval(), data_util

def dump_feats(vid_paths, feat_paths, model, data_util):
	def save_feat(vidpath, featpath, model):
		if os.path.exists(featpath): return

		src = torch.FloatTensor(data_util.read_video(vidpath)).unsqueeze(0)

		with torch.no_grad():
			src = augmentor(src).detach()
			src_mask = torch.ones((1, 1, src.size(2)))
			with autocast():
				src = src.cuda()
				src_mask = src_mask.cuda()
				outs = []
				chunk_size = 512 # increase or decrease based on how much GPU memory you have
				i = 0
				while i < src.size(2):
					s = src[:, :, i : i + chunk_size]
					m = src_mask[:, :, i : i + chunk_size]

					outs.append(model.face_encoder(s, m)[0].cpu())

					i += chunk_size

				out = torch.cat(outs, dim=0)

				np.save(featpath, out.cpu().numpy().astype(np.float16))

	for v, f in tqdm(zip(vid_paths, feat_paths), total=len(vid_paths)):
		try:
			save_feat(v, f, model)
		except Exception as e:
			print(e)

def main(args):
	args.device = 'cuda'
	model, data_util = init(args)
	if args.ckpt_path is not None:
		print('Resuming from: {}'.format(args.ckpt_path))
		model = load(model, args.ckpt_path)[0]
	else:
		raise SystemError('Need a checkpoint to dump feats')

	vidpaths = sorted(list(glob('{}/{}'.format(args.videos_root, args.file_list))))

	print('################# INFO ####################')
	print('Will be dumping to: {}'.format(args.feats_root))
	print('Found {} videos'.format(len(vidpaths)))
	# input('Press Enter to continue. Or Press Ctrl + C to abort')

	featpaths = [f.replace(args.videos_root, args.feats_root)[:-3] + '.npy' for f in vidpaths]
	all_folders = np.unique([os.path.dirname(f) for f in featpaths])
	for f in all_folders:
		os.makedirs(f, exist_ok=True)

	print('Created the directory structure')

	print('Sorted {} videos'.format(len(vidpaths)))

	part = args.part
	num_parts = args.num_parts
	part_size = len(vidpaths) // num_parts
	input('Doing for part {}, press Enter to continue.'.format(part))
	cur_vid_paths = vidpaths[part * part_size :] if part == num_parts - 1 else \
				vidpaths[part * part_size : (part + 1) * part_size]
	cur_feat_paths = featpaths[part * part_size :] if part == num_parts - 1 else \
				featpaths[part * part_size : (part + 1) * part_size]

	dump_feats(cur_vid_paths, cur_feat_paths, model, data_util)


if __name__ == '__main__':
	args = load_args()
	augmentor = AugmentationPipeline(args)
	main(args)