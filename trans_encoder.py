import torch
import torch.nn as nn
from x_transformers import Encoder
import torch.utils.checkpoint as checkpoint


class XTEncoder(nn.Module):
    """
    6-Layer Transformer Encoder with Rotary Position Embeddings, RMSNorm, and Flash Attention.
    """
    def __init__(self, dim=384, depth=6, heads=6, gradient_checkpointing=False):
        super().__init__()
        self.dim = dim
        self.gradient_checkpointing = gradient_checkpointing
        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads if heads is not None else max(1, dim // 64),
            ff_swish=True,
            ff_mult=4,
            attn_flash=True,
            rotary_pos_emb=True,
            use_rmsnorm=True,
            ff_no_bias=True,
            attn_dropout=0.0,
            ff_dropout=0.0,
            layer_dropout=0.0,
        )

    def _encoder_step(self, x, mask=None):
        return self.encoder(x, mask=mask)

    def forward(self, x, mask=None):
        if self.training and self.gradient_checkpointing:
            return checkpoint.checkpoint(self._encoder_step, x, mask, use_reentrant=False)
        return self.encoder(x, mask=mask)