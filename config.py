import configargparse

pad_token = '<pad>'
bos_token = '<bos>'
eos_token = '<eos>'
unk_token = '<unk>'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_args():

  parser = configargparse.ArgumentParser(description = "main")

  parser.add_argument('--device', type=str, default='cuda')

  parser.add_argument('--builder', default='base')
  parser.add_argument('--ckpt_path', default=None)
  parser.add_argument('--fp16', default=True, type=str2bool)

  # Data
  parser.add_argument('--feat_dim', type=int, default=512, help='Video features dimension - used if loading features directly (featurizer=False)')
  parser.add_argument('--videos_root', type=str)
  parser.add_argument('--feats_root', type=str)

  # Transformer config
  parser.add_argument('--num_blocks', type=int, default=6, help='# of transformer blocks')
  parser.add_argument('--hidden_units', type=int, default=512, help='Transformer model size')
  parser.add_argument('--num_heads', type=int, default=8, help='# of attention heads')
  parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout probability')

  # Preprocessing
  parser.add_argument('--img_size', type=int, default=96, help='Resize the input face frames to this resolution')
  parser.add_argument('--frame_size', type=int, default=160, help='Resize the input video frames to this resolution')
  parser.add_argument('--normalize', type=str2bool, default=True, help='Normalize by dividing by 255')

  # Data aug params
  parser.add_argument('--rot', type=float, default=10., help='Rotation degrees')
  parser.add_argument('--rand_crop', type=int, default=3, help='Rand Crop with this offset')

  # feature extraction config
  parser.add_argument('--file_list', type=str, help='(List of) video file paths relative to videos_folder, can also be regex')
  parser.add_argument('--num_parts', type=int, default=1, help='Partwise feature dumping, how many parts?')
  parser.add_argument('--part', type=int, default=0, help='Partwise feature dumping, which part?')

  args = parser.parse_args()

  return args

