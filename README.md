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
|LRS2 + LRS3|[Feature extractor](https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/public_train_data/feature_extractor.pth); FT-LRS2; FT-LRS3          |
|LRS2 + LRS3 + MVLRS + LRS3v2| To be released.

## Feature extraction

After downloading the feature extractor checkpoint, run the following from the project root folder:

`python extract_feats.py --builder vtp24x24 --ckpt_path checkpoints/feature_extractor.pth --videos_root video_data_root --file_list \*/*.mp4 --feats_root feature_extraction_dest_root`

The file_list argument can be a regex (to extract for a list of files) or a single file. `*/*.mp4` is an example regex. 