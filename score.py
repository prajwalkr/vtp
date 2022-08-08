import numpy as np
import torch, os, cv2, pickle

from tqdm import tqdm
from glob import glob
import pandas as pd

from utils import load, levenshtein
from inference import main as inference_initializer
from inference import run as get_prediction
from inference import args

def main(args):
	model, video_loader, lm, lm_tokenizer = inference_initializer(args)

	total_wer, total_cer, total_tokens, total_chars = 0., 0., 0., 0.

	df = pd.read_csv(args.fpath)
	fnames = df[list(df.columns.values)[0]].tolist()
	texts = df['transcript'].tolist()

	prog_bar = tqdm(list(zip(fnames, texts)))
	for data in prog_bar:
		fname = data[0]
		if fname.endswith('.wav'): fname = fname[:-4]
		gt = data[1]

		fpath = f"{args.videos_root}/{fname}.mp4"
		pred = get_prediction(fpath, video_loader, model, 
								lm, lm_tokenizer, display=False)
		
		wer = levenshtein(gt.split(), pred.split())
		cer = levenshtein(list(gt), list(pred))

		total_wer += wer
		total_cer += cer
		total_tokens += len(gt.split())
		total_chars += len(list(gt))

		prog_bar.set_description('WER: {}, CER: {}'.format(
								total_wer / total_tokens, total_cer / total_chars))

if __name__ == '__main__':
	main(args)