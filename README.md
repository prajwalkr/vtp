# Sub-word level Lip reading with Visual Attention

This is the official implementation of the paper. The code has been tested with Python version 3.6.8. Pre-trained checkpoints are also released below. 

## Setup
- `pip install -r requirements.txt`
- Download the necessary checkpoints, links are available in the table below.
  - `cd checkpoints/`
  - `wget <link_to_ckpt>`

## Checkpoints

|Training data|Link                         |
|-------------------------------|-----------------------------|
|LRS2 + LRS3|[Feature extractor](https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/public_train_data/feature_extractor.pth); [FT-LRS2](https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/public_train_data/ft_lrs2.pth); [FT-LRS3](https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/public_train_data/ft_lrs3.pth)|
|LRS2 + LRS3 + MVLRS + LRS3v2| [Feature extractor](https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data/feature_extractor.pth); [FT-LRS2](https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data/ft_lrs2.pth); [FT-LRS3](https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data/ft_lrs3.pth)|

## Running inference on a video of your choice

The first step to do is to extract and preprocess face tracks using the `run_pipeline.py` script from this repo: [syncnet_python](https://github.com/joonson/syncnet_python). Once you have the face track, you can run the following script to perform lip-reading. 

`python inference.py --builder vtp24x24 --ckpt_path <path_to_a_ft_ckpt> --cnn_ckpt_path <path_to_corresponging_feat_extractor> --beam_size 30 --max_decode_len 35 --fpath <path_to_face_track>`

## Feature extraction

After downloading the feature extractor checkpoint, run the following from the project root folder:

`python extract_feats.py --builder vtp24x24 --ckpt_path <path_to_a_feat_extractor> --videos_root <video_data_root> --file_list */*.mp4 --feats_root <feature_extraction_dest_root>`

The file_list argument can be a regex (to extract for a list of files) or a single file. `*/*.mp4` is an example regex. 

## Reproduce the paper's best scores on the standard test sets

Download the test csv files by running the `download.sh` script in the `test_csvs` folder. 

### Models trained on public data
**LRS2 test set** (WER=28.9)
`python score.py --builder vtp24x24 --ckpt_path <path_to_lrs2_ft_ckpt> --cnn_ckpt_path <path_to_feat_extractor> --beam_size 40 --max_decode_len 25 --fpath test_csvs/lrs2-test.csv --videos_root <path_to_main_folder_of_lrs2> --lm_alpha 0.35`

**LRS3 test set** (WER=40.6)
`python score.py --builder vtp24x24 --ckpt_path <path_to_lrs3_ft_ckpt> --cnn_ckpt_path <path_to_feat_extractor> --beam_size 30 --max_decode_len 35 --fpath test_csvs/lrs3-test.csv --videos_root <path_to_test_folder_of_lrs3> --lm_alpha 0.35`

### Models trained on a larger internal dataset (LRS3v2)
**LRS2 test set** (WER=22.3)
`python score.py --builder vtp24x24 --ckpt_path <path_to_lrs2_ft_ckpt> --cnn_ckpt_path <path_to_feat_extractor> --beam_size 30 --max_decode_len 25 --fpath <path_to_lrs2_test_csv> --videos_root <path_to_main_folder_of_lrs2> --lm_alpha 0.3`

**LRS3 test set** (WER=30.7)
`python score.py --builder vtp24x24 --ckpt_path <path_to_lrs3_ft_ckpt> --cnn_ckpt_path <path_to_feat_extractor> --beam_size 30 --max_decode_len 35 --fpath <path_to_lrs3_test_csv> --videos_root <path_to_test_folder_of_lrs3> --lm_alpha 0.4`