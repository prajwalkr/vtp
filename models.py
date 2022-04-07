import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import EncoderDecoder, PositionwiseFeedForward, PositionalEncoding, EncoderLayer, \
        DecoderLayer, MultiHeadedAttention, Encoder, Decoder, \
        Generator, Embeddings

from modules import CNN_3d_featextractor, CNN_3d, VTP, VTP_wrapper

def CNN_Baseline(vocab, visual_dim, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, 
                    backbone=True):
    c = copy.deepcopy

    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        (CNN_3d(visual_dim) if backbone else None),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),

        nn.Sequential(c(position)),
        nn.Sequential(Embeddings(d_model, vocab), c(position)),
        Generator(d_model, vocab))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# Lip-reading VTP model
def VTP_24x24(vocab, visual_dim, N=6, d_model=512, 
                d_ff=2048, h=8, dropout=0.1, backbone=True):
    c = copy.deepcopy

    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    linear_pooler = VTP(num_layers=[3, 3], dims=[256, 512], heads=[8, 8], 
                                patch_sizes=[1, 2], initial_resolution=24, initial_dim=128)

    model = EncoderDecoder(
        (VTP_wrapper(CNN_3d_featextractor(d_model, till=24), linear_pooler, in_dim=128, 
                                out_dim=512) if backbone else None),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),

        nn.Sequential(c(position)),
        nn.Sequential(Embeddings(d_model, vocab), c(position)),
        Generator(d_model, vocab))
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

###### VSD model
def Silencer_VTP_24x24(visual_dim, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, 
                    backbone=False):
    c = copy.deepcopy

    from modules import Silencer
    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    linear_pooler = VTP(num_layers=[3, 3], dims=[256, 512], heads=[8, 8], 
                            patch_sizes=[1, 2], initial_resolution=24, initial_dim=128)

    model = Silencer(
        (VTP_wrapper(CNN_3d96_featextractor(d_model, till=24), linear_pooler, in_dim=128, 
                                out_dim=512) if backbone else None), c(position),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Linear(d_model, 1))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

builders = {
    'cnn_baseline' : CNN_Baseline, 
    'vtp24x24' : VTP_24x24,

    # vsd
    'silencer_vtp24x24' : Silencer_VTP_24x24,
}
