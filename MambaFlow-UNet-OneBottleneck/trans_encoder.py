import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mamba_ssm import Mamba2


def reverse_valid_sequence(x, mask=None):
    """
    Reverses sequence x along time dimension (dim=1) only for valid non-padded tokens.
    Vectorized tensor gather without Python loops for consistent CUDA execution and gradients.
    """
    if mask is None:
        return torch.flip(x, dims=[1])
    
    B, T, D = x.shape
    m_bool = mask.bool() if mask.dim() == 2 else mask.squeeze(-1).bool()
    valid_lengths = m_bool.sum(dim=1)
    
    t_range = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
    valid_mask = t_range < valid_lengths.unsqueeze(1)
    rev_indices = torch.clamp(valid_lengths.unsqueeze(1) - 1 - t_range, min=0)
    indices = torch.where(valid_mask, rev_indices, t_range)
    
    idx_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(x, dim=1, index=idx_expanded)


class ConvNeXt1DBlock(nn.Module):
    """
    ConvNeXt-1D Large-Kernel Local Refinement Block:
    - Depthwise Conv1D with large kernel (e.g. k=9, 17, 31)
    - RMSNorm
    - Inverted Bottleneck: Linear (dim -> 4*dim) -> SiLU -> Linear (4*dim -> dim)
    - Residual connection
    """
    def __init__(self, dim=192, kernel_size=17, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.norm = nn.RMSNorm(dim)
        self.pw1 = nn.Linear(dim, mlp_ratio * dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.pw2 = nn.Linear(mlp_ratio * dim, dim, bias=False)

    def forward(self, x, mask=None):
        # x: [B, T, dim]
        res = x
        x_t = x.transpose(1, 2)
        if mask is not None:
            m = mask.transpose(1, 2) if mask.dim() == 3 else mask.unsqueeze(1)
            x_t = x_t * m.float()
            m_c = mask if mask.dim() == 3 else mask.unsqueeze(-1)
        else:
            m_c = None

        h = self.dwconv(x_t).transpose(1, 2)
        if m_c is not None:
            h = h * m_c.float()

        h = self.norm(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.pw2(h)

        if m_c is not None:
            h = h * m_c.float()

        return res + h


class BiMambaConvNeXtLayer(nn.Module):
    """
    Single Bi-Mamba + ConvNeXt Encoder Layer:
                x
                │
             RMSNorm
                │
       ┌────────┴────────┐
       ↓                 ↓
  Mamba forward     Mamba backward
       │                 │
       └───────┬─────────┘
               ↓
         gated fusion
               ↓
          residual + x
               │
            RMSNorm
               │
          ConvNeXt (large kernel)
               │
          residual + x
    """
    def __init__(
        self,
        dim=192,
        kernel_size=17,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.RMSNorm(dim)

        # Forward and backward Mamba2 streams
        self.mamba_fwd = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.mamba_bwd = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        # Gated bidirectional fusion network
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * dim, dim, bias=False),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Linear(2 * dim, dim, bias=False)

        # Second sub-block: RMSNorm + ConvNeXt with large kernel
        self.norm2 = nn.RMSNorm(dim)
        self.convnext = ConvNeXt1DBlock(
            dim=dim,
            kernel_size=kernel_size,
            mlp_ratio=4,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        # -------------------------------------------------------------
        # Sub-block 1: RMSNorm -> Bi-Mamba2 (Fwd & Bwd) -> Gated Fusion -> Residual (+x)
        # -------------------------------------------------------------
        h = self.norm1(x)
        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            h = h * m
        else:
            m = None

        # Forward scan (left -> right)
        h_fwd = self.mamba_fwd(h)
        if m is not None:
            h_fwd = h_fwd * m

        # Backward scan (right -> left)
        h_rev = reverse_valid_sequence(h, mask=m)
        h_bwd_rev = self.mamba_bwd(h_rev)
        h_bwd = reverse_valid_sequence(h_bwd_rev, mask=m)
        if m is not None:
            h_bwd = h_bwd * m

        # Gated bidirectional fusion
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        gate = self.gate_proj(h_cat)
        h_fused = self.fusion_proj(h_cat)
        h_mamba = gate * h_fused

        if m is not None:
            h_mamba = h_mamba * m

        x_1 = x + h_mamba

        # -------------------------------------------------------------
        # Sub-block 2: RMSNorm -> ConvNeXt-1D Large Kernel -> Residual (+x_1)
        # -------------------------------------------------------------
        h_conv_in = self.norm2(x_1)
        if m is not None:
            h_conv_in = h_conv_in * m

        x_2 = self.convnext(h_conv_in, mask=mask)

        if m is not None:
            x_2 = x_2 * m

        return x_2


class BiMambaConvNeXtEncoder(nn.Module):
    """
    6-Layer Bidirectional Mamba2 + Large-Kernel ConvNeXt Text Encoder:
    Combines global bidirectional SSM context with multi-scale Conv1D local refinement.
    """
    def __init__(
        self,
        dim=192,
        depth=6,
        kernel_sizes=(9, 17, 31, 9, 17, 31),
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        dropout=0.0,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.gradient_checkpointing = gradient_checkpointing

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * depth
        elif len(kernel_sizes) < depth:
            kernel_sizes = list(kernel_sizes) * ((depth // len(kernel_sizes)) + 1)
            kernel_sizes = kernel_sizes[:depth]

        self.layers = nn.ModuleList([
            BiMambaConvNeXtLayer(
                dim=dim,
                kernel_size=kernel_sizes[i],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                dropout=dropout,
            )
            for i in range(depth)
        ])
        self.norm_final = nn.RMSNorm(dim)

    def forward(self, x, mask=None):
        m = None
        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = checkpoint.checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask=mask)

        out = self.norm_final(x)
        if m is not None:
            out = out * m
        return out


# =====================================================================
# Previous Transformer-based XTEncoder (Commented for reference as requested)
# =====================================================================
# from x_transformers import Encoder
#
# class XTEncoderLegacy(nn.Module):
#     """
#     6-Layer Transformer Encoder with Rotary Position Embeddings, RMSNorm, and Flash Attention.
#     """
#     def __init__(
#         self,
#         dim=192,
#         depth=6,
#         heads=6,
#         attn_dropout=0.0,
#         ff_dropout=0.0,
#         layer_dropout=0.0,
#         gradient_checkpointing=False,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.gradient_checkpointing = gradient_checkpointing
#         self.encoder = Encoder(
#             dim=dim,
#             depth=depth,
#             heads=heads if heads is not None else max(1, dim // 32),
#             ff_swish=True,
#             ff_mult=4,
#             attn_flash=True,
#             rotary_pos_emb=True,
#             use_rmsnorm=True,
#             ff_no_bias=True,
#             attn_dropout=attn_dropout,
#             ff_dropout=ff_dropout,
#             layer_dropout=layer_dropout,
#         )
#
#     def _encoder_step(self, x, mask=None):
#         return self.encoder(x, mask=mask)
#
#     def forward(self, x, mask=None):
#         if self.training and self.gradient_checkpointing:
#             return checkpoint.checkpoint(self._encoder_step, x, mask, use_reentrant=False)
#         return self.encoder(x, mask=mask)


# Export alias for seamless integration across codebase
XTEncoder = BiMambaConvNeXtEncoder
TextEncoder = BiMambaConvNeXtEncoder