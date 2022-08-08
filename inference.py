import numpy as np
import torch, os, cv2, pickle, sys

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from glob import glob

from dataloader import VideoDataset, AugmentationPipeline

from config import load_args, start_symbol, end_symbol
from models import builders
from utils import load

from torch.cuda.amp import autocast
from search import beam_search

args = load_args()
augmentor = AugmentationPipeline(args)

def forward_pass(model, src, src_mask):
	encoder_output, src_mask = model.encode(src, src_mask)

	beam_outs, beam_scores = beam_search(
			decoder=model,
			bos_index=start_symbol,
			eos_index=end_symbol,
			max_output_length=args.max_decode_len,
			pad_index=0,
			encoder_output=encoder_output,
			src_mask=src_mask,
			size=args.beam_size,
			alpha=args.beam_len_alpha,
			n_best=args.beam_size,
		)

	return beam_outs, beam_scores

def get_lm_score(lm, lm_tokenizer, texts):
	logloss = nn.CrossEntropyLoss()
	tokens_tensor = lm_tokenizer.batch_encode_plus(texts, 
								return_tensors="pt", padding=True)
	logits = lm(tokens_tensor['input_ids'], 
					attention_mask=tokens_tensor['attention_mask'])[0]
	losses = []
	for logits, m, labels in zip(logits, tokens_tensor['attention_mask'], 
										tokens_tensor['input_ids']):
		loss = logloss(logits[:m.sum() - 1], labels[1:m.sum()])
		losses.append(loss.item())

	losses = 1./np.exp(np.array(losses)) # higher should be treated as better
	return losses

def minmax_normalize(values):
	v = np.array(values)
	v = (v - v.min()) / (v.max() - v.min())
	return v

def run(vidpath, dataloader, model, lm=None, lm_tokenizer=None, display=True):
	frames = torch.FloatTensor(dataloader.read_video(vidpath)).unsqueeze(0)
	frames = augmentor(frames).detach()

	if args.ss is not None or args.es is not None:
		if args.ss is None:
			ss = 0
		else:
			ss = int(25. * args.ss)
		if args.es is not None:
			es = int(25 * args.es)
			frames = frames[:, :, ss:es]
		else:
			frames = frames[:, :, ss:]

	chunk_frames = args.chunk_size * 25
	preds = []
	for i in range(0, frames.size(2), chunk_frames):
		cur_src = frames[:, :, i : i + chunk_frames].to(args.device)
		cur_src_mask = torch.ones((1, 1, cur_src.size(2))).to(args.device)

		with torch.no_grad():
			with autocast():
				beam_outs, beam_scores = forward_pass(model, cur_src, cur_src_mask)
				beam_outs_f, beam_scores_f = forward_pass(model, 
								augmentor.horizontal_flip(cur_src), cur_src_mask)

				beam_outs = beam_outs[0] + beam_outs_f[0]
				beam_scores = np.array(beam_scores[0] + beam_scores_f[0])

				if lm is not None:
					pred_texts = [dataloader.to_tokens(o.cpu().numpy().tolist()) \
									for o in beam_outs]

					lm_scores = get_lm_score(lm, lm_tokenizer, pred_texts)
					lm_scores = minmax_normalize(lm_scores)
					beam_scores = minmax_normalize(beam_scores)

					beam_scores = args.lm_alpha * lm_scores + \
									(1 - args.lm_alpha) * beam_scores

		best_pred_idx = beam_scores.argmax()

		out = beam_outs[best_pred_idx]
		pred = dataloader.to_tokens(out.cpu().numpy().tolist())

		preds.append(pred)

	pred = ' '.join(preds)
	if display: print(pred)
	return pred

def main(args):
	video_loader = VideoDataset(args)

	model = builders[args.builder](video_loader.vocab_size + 1, args.feat_dim, 
							N=args.num_blocks, d_model=args.hidden_units, 
							h=args.num_heads, 
							dropout=args.dropout_rate).to(args.device).eval()

	assert args.ckpt_path is not None, 'Specify a trained checkpoint!'
	assert args.cnn_ckpt_path is not None, 'Specify a trained visual backbone checkpoint!'
	print('Loading checkpoints...')
	model = load(model, args.ckpt_path, face_encoder_ckpt=args.cnn_ckpt_path, device=args.device)[0]

	if args.lm_alpha > 0.:
		from transformers import GPT2LMHeadModel, GPT2Tokenizer
		lm = GPT2LMHeadModel.from_pretrained('gpt2').eval()
		lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=True)
		lm_tokenizer.pad_token = '<pad>'
	else:
		lm, lm_tokenizer = None, None

	return model, video_loader, lm, lm_tokenizer

if __name__ == '__main__':
	model, video_loader, lm, lm_tokenizer = main(args)
	run(args.fpath, video_loader, model, lm, lm_tokenizer)