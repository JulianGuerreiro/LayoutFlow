# Implement TransformerEncoder that can consider timesteps as optional args for Diffusion.
import copy
import numpy as np
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from einops import rearrange, pack, unpack

from src.models.BaseGenModel import PositionalEncoding

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _gelu2(x):
    return x * F.sigmoid(1.702 * x)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu2":
        return _gelu2
    else:
        raise RuntimeError(
            "activation should be relu/gelu/gelu2, not {}".format(activation)
        )

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: Tensor):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        d_model=1024,
        nhead=16,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()

        assert norm_first  # minGPT-based implementations are designed for prenorm only
        layer_norm_eps = 1e-5  # fixed

        self.norm_first = norm_first

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = torch.nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = AdaLayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Tensor = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = self.norm1(x, timestep)
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x, timestep)
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    """
    Close to torch.nn.TransformerEncoder, but with timestep support for diffusion
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Tensor = None,
    ) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                timestep=timestep,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class LayoutDMBackbone(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        tr_enc_only: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 8,
        dropout: float = 0.1,
        use_pos_enc: bool = False,
        num_cat: int = 6,
        attr_encoding = 'continuous',
        seq_type = 'stacked',
    ):
        super().__init__()
        self.geom_dim = 4
        self.num_cat = num_cat
        self.attr_encoding = attr_encoding
        self.seq_type = seq_type

        self.use_pose_enc = use_pos_enc
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=200)

        self.cond_enc = nn.Embedding(6, latent_dim if seq_type=='stacked' else 2*latent_dim)

        attr_dim = int(np.ceil(np.log2(num_cat))) if attr_encoding == 'AnalogBit' else 1
        if attr_encoding == 'discrete':
            self.type_embed = nn.Embedding(num_cat, latent_dim)
            self.geom_embed = nn.Linear(self.geom_dim, latent_dim)
        else:
            self.type_embed = nn.Linear(attr_dim, latent_dim if seq_type=='stacked' else 2*latent_dim)
            if seq_type == 'seq':
                self.geom_enc = nn.ModuleList([nn.Linear(1, 2*latent_dim) for i in range(4)])
            else:
                self.geom_embed = nn.Linear(self.geom_dim, latent_dim if seq_type=='stacked' else 2*latent_dim)
        self.elem_embed = nn.Linear(2*latent_dim, d_model)

        self.num_layers = num_layers
        decoder_layer = Block(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            norm_first= True,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.tr_enc_only = tr_enc_only
        if tr_enc_only:
            self.transformer = TransformerEncoder(decoder_layer, num_layers=num_layers, norm=torch.nn.LayerNorm(d_model))
        else:
            self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, batch_first=True)
        self.linear = nn.Linear(d_model, self.geom_dim+attr_dim)
        if seq_type!='stacked':
            k = 2 if seq_type=='seq_cond' else 5
            self.to_attrdim = nn.Linear(k*(self.geom_dim+attr_dim), self.geom_dim+attr_dim)

    def forward(self, 
                geom: Tensor, 
                attr: Tensor, 
                cond_flags: Tensor, 
                t: Tensor = 0) -> Any:
        '''
        geom (N, S, 4)
        attr (N, S, 1)
        cond_flags (N, S, 1)
        '''

        if self.attr_encoding == 'discrete':
            geom = self.geom_embed(geom)
            attr = self.type_embed(attr.squeeze())
            x = torch.cat([geom, attr], dim=-1)
        else:
            if self.seq_type == 'stacked':
                geom = self.geom_embed(geom) + self.cond_enc(cond_flags[:,:,:self.geom_dim].sum(-1))
                attr = self.type_embed(attr) + self.cond_enc(cond_flags[:,:,-1])
                x, ps = pack([geom, attr], 'b s *')
            elif self.seq_type == 'seq_cond':
                geom = self.geom_embed(geom) + self.cond_enc(cond_flags[:,:,:self.geom_dim].sum(-1))
                attr = self.type_embed(attr) + self.cond_enc(cond_flags[:,:,-1])
                x, ps = pack([geom, attr], 'b * d')
            elif self.seq_type == 'seq':
                geom = [self.geom_enc[i](geom[:,:,i,None]) + self.cond_enc(cond_flags[:,:,i]) for i in range(4)]
                attr = self.type_embed(attr) + self.cond_enc(cond_flags[:,:,-1])
                x, ps = pack(geom + [attr], 'b * d')
            else:
                print('Wrong sequence type. Choose from [stacked, seq_cond, seq]')

        x = self.elem_embed(x)
        if self.use_pose_enc:
            x += self.pos_enc(x)

        if self.tr_enc_only:
            x = self.transformer(x, timestep=t)
        else:
            x = self.transformer(x, x)
        
        x = self.linear(x)
        if not self.seq_type=='stacked':
            x = unpack(x, ps, 'b * d')
            x = rearrange(x, 'k b s d -> b s (k d)')
            x = self.to_attrdim(x)

        return x