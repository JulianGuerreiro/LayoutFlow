from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, pack, unpack

from src.models.BaseGenModel import PositionalEncoding, TimePositionalEncoding


class SimpleTransformerFlex(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        tr_enc_only: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 8,
        dropout: float = 0.1,
        fixed_ratio: bool = False,
        use_pos_enc: bool = False,
        use_time_enc: bool = False,
        num_cat: int = 6,
        attr_encoding = 'continuous',
        activation = '',
        stacked = True,
        time_emb = 'pure_positional',
        apply_cond_mask = 'pre_stack',
    ):
        super().__init__()
        self.geom_dim = 4 if not fixed_ratio else 3
        self.num_cat = num_cat
        self.attr_encoding = attr_encoding
        self.stacked = stacked

        self.use_pose_enc = use_pos_enc
        self.use_time_enc = use_time_enc
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=200)
        self.time_enc = TimePositionalEncoding(d_model, dropout)
        self.time_emb = time_emb
        if time_emb != 'pure_positional':
            self.time_transform = nn.Sequential(nn.Linear(d_model, d_model))
        self.cond_enc = nn.Embedding(6, latent_dim if (stacked and apply_cond_mask=='pre_stack') else 2*latent_dim)
        self.apply_cond_mask = apply_cond_mask

        attr_dim = int(np.ceil(np.log2(num_cat))) if attr_encoding == 'AnalogBit' else 1
        if attr_encoding == 'discrete':
            self.type_embed = nn.Embedding(num_cat, latent_dim)
            self.geom_embed = nn.Linear(self.geom_dim, latent_dim)
        else:
            self.type_embed = nn.Linear(attr_dim, latent_dim if stacked else 2*latent_dim)
            self.geom_embed = nn.Linear(self.geom_dim, latent_dim if stacked else 2*latent_dim)
        self.elem_embed = nn.Linear(2*latent_dim, d_model)

        self.num_layers = num_layers
        decoder_layer = nn.TransformerEncoderLayer (
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.tr_enc_only = tr_enc_only
        if tr_enc_only:
            self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        else:
            self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, batch_first=True)
        self.linear = nn.Linear(d_model if stacked else 2*d_model, self.geom_dim+attr_dim)

        self.activation = activation in ['sigmoid', 'tanh']
        if activation == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activation == 'tanh':
            self.activ = nn.Tanh()
        else:
            self.activ = nn.Identity()


    def forward(self, geom, attr, cond_flags, t=0, batch=None) -> Any:
        '''
        geom (N, S, 4)
        attr (N, S, 1)
        cond_floags (N, S, 1)
        '''

        if self.attr_encoding == 'discrete':
            geom = self.geom_embed(geom)
            attr = self.type_embed(attr.squeeze())
            x = torch.cat([geom, attr], dim=-1)
        else:
            geom = self.geom_embed(geom)
            attr = self.type_embed(attr)
            if self.apply_cond_mask == 'pre_stack':
                geom += self.cond_enc(cond_flags[:,:,:self.geom_dim].sum(-1))
                attr += self.cond_enc(cond_flags[:,:,-1])
            if self.stacked:
                x, _ = pack([geom, attr], 'b s *')
            else:
                x, _ = pack([geom, attr], 'b * d')
            if self.apply_cond_mask != 'pre_stack':
               x += self.cond_enc(cond_flags[:,:,:self.geom_dim+1].sum(-1)) 

        x = self.elem_embed(x)
        if self.use_pose_enc:
            x += self.pos_enc(x)
        if self.use_time_enc:
            t_emb = self.time_enc(t) if self.time_emb=='pure_positional' else self.time_transform(self.time_enc(t))
            if self.time_emb == 'tokenize':
                x, ps = pack([t_emb, x], 'b * d')
            else:
                x += t_emb

        if self.tr_enc_only:
            x = self.transformer(x)
        else:
            x = self.transformer(x, x)
        
        if not self.stacked:
            x = rearrange(x, 'b (s k) d -> b s (k d)', k=2)
        if self.time_emb == 'tokenize':
            x, _ = unpack(x, ps, 'b * d') # remove time embedding token 
        x = self.linear(x)
        x = self.activ(x)

        return x