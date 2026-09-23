import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2


def reverse_valid_sequence(x, mask=None):
    """
    Reverses sequence x along time dimension (dim=1) exclusively for valid non-padded tokens.
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


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for 1D sequence tokens/frames.
    """
    def __init__(self, dim=256, max_len=10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        T = x.size(1)
        if T > self.pe.size(1):
            device = x.device
            position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.dim))
            pe = torch.zeros(1, T, self.dim, device=device, dtype=x.dtype)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            return pe
        return self.pe[:, :T, :].to(device=x.device, dtype=x.dtype)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt 1D Block with FiLM adaptive conditioning:
    - 1D Depthwise Convolution (kernel_size=7) for local acoustic continuity
    - RMSNorm
    - Inverted Bottleneck: Linear (dim -> mult*dim) -> SiLU -> Linear (mult*dim -> dim), bias=False
    - FiLM conditioning modulation: gamma, beta, gate (alpha)
    - Strictly mask-aware
    """
    def __init__(self, dim=384, kernel_size=7, mult=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True),
        )
        nn.init.xavier_uniform_(self.cond_mlp[1].weight, gain=0.1)
        nn.init.zeros_(self.cond_mlp[1].bias)

        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )
        self.norm = nn.RMSNorm(dim)
        self.pw1 = nn.Linear(dim, mult * dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.pw2 = nn.Linear(mult * dim, dim, bias=False)

    def forward(self, x, cond, mask=None):
        film = self.cond_mlp(cond)
        gamma, beta, gate = film.chunk(3, dim=-1)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        if gate.dim() == 2:
            gate = gate.unsqueeze(1)
        alpha = torch.sigmoid(gate)

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            x = x * m
        else:
            m = None

        # 1. Depthwise conv
        h = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        if m is not None:
            h = h * m

        # 2. Modulated norm
        h_norm = self.norm(h)
        h_mod = h_norm * (1.0 + gamma) + beta
        if m is not None:
            h_mod = h_mod * m

        # 3. Inverted bottleneck
        h_pw = self.pw1(h_mod)
        h_pw = self.act(h_pw)
        h_pw = self.dropout(h_pw)
        update = self.pw2(h_pw)
        if m is not None:
            update = update * m

        out = x + alpha * update
        if m is not None:
            out = out * m
        return out


class BiMamba2PointwiseMLPBlock(nn.Module):
    """
    Bidirectional Mamba-2 + Pointwise Linear MLP Block (d_model=384):
    - FiLM adaptive conditioning: gamma, beta, gate (alpha) from timestep conditioning alone
    - Modulation: RMSNorm(x) * (1 + gamma) + beta
    - Sinusoidal token positional encoding
    - Forward Mamba-2 SSM scan (384-dim)
    - Valid-reversal backward Mamba-2 SSM scan (384-dim)
    - Concatenation: [h_fwd, h_bwd] -> 768-dim
    - Pointwise Linear MLP (bias=False, SiLU activation):
      Linear(768 -> 1024) -> SiLU -> Linear(1024 -> 384)
      (default mlp_hidden_dim=1024)
    - Adaptive residual scaling: x + alpha * update
    - Strict mask awareness at every intermediate step
    """
    def __init__(
        self,
        d_model=384,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        mlp_hidden_dim=1024,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.mlp_hidden_dim = mlp_hidden_dim

        # Timestep FiLM adaptive conditioning projection (gamma, beta, gate)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model, bias=True),
        )
        nn.init.xavier_uniform_(self.cond_mlp[1].weight, gain=0.1)
        nn.init.zeros_(self.cond_mlp[1].bias)

        self.norm = nn.RMSNorm(d_model)
        self.pos_emb = SinusoidalPositionalEncoding(d_model)

        # Forward & Backward Mamba-2 SSMs
        self.mamba_fwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.mamba_bwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        # Pointwise Linear MLP without bias: 2*d_model (512) -> mlp_hidden_dim (768) -> d_model (256)
        self.mlp_pw1 = nn.Linear(2 * d_model, mlp_hidden_dim, bias=False)
        self.mlp_act = nn.SiLU()
        self.mlp_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.mlp_pw2 = nn.Linear(mlp_hidden_dim, d_model, bias=False)

        nn.init.xavier_uniform_(self.mlp_pw1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.mlp_pw2.weight, gain=0.1)

    def forward(self, x, cond, mask=None):
        # 1. Conditioning FiLM modulation (timestep alone)
        film = self.cond_mlp(cond)
        gamma, beta, gate = film.chunk(3, dim=-1)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        if gate.dim() == 2:
            gate = gate.unsqueeze(1)
        alpha = torch.sigmoid(gate)

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            x = x * m
        else:
            m = None

        # 2. Modulated norm
        h_norm = self.norm(x)
        h_mod = h_norm * (1.0 + gamma) + beta
        if m is not None:
            h_mod = h_mod * m

        # 3. Positional encoding
        pos = self.pos_emb(h_mod)
        h_pos = h_mod + pos
        if m is not None:
            h_pos = h_pos * m

        # 4. Bidirectional SSM Scans
        # Forward scan (left -> right)
        h_fwd = self.mamba_fwd(h_pos)
        if m is not None:
            h_fwd = h_fwd * m

        # Backward scan (right -> left, reversing valid sequence tokens only)
        h_rev = reverse_valid_sequence(h_pos, mask=m)
        h_bwd_rev = self.mamba_bwd(h_rev)
        if m is not None:
            h_bwd_rev = h_bwd_rev * m
        h_bwd = reverse_valid_sequence(h_bwd_rev, mask=m)
        if m is not None:
            h_bwd = h_bwd * m

        # 5. Concatenation: [h_fwd, h_bwd] -> [B, T, 2 * d_model] (e.g. 512)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        if m is not None:
            h_cat = h_cat * m

        # 6. Pointwise Linear MLP without bias
        h_mlp = self.mlp_pw1(h_cat)
        h_mlp = self.mlp_act(h_mlp)
        h_mlp = self.mlp_dropout(h_mlp)
        update = self.mlp_pw2(h_mlp)
        if m is not None:
            update = update * m

        # 7. Adaptive residual connection
        out = x + alpha * update
        if m is not None:
            out = out * m
        return out


class Mamba2ForwardBlock(nn.Module):
    """
    Backward-compatibility Forward-Only Mamba-2 Block.
    """
    def __init__(
        self,
        d_model=256,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
    ):
        super().__init__()
        self.d_model = d_model

        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model, bias=True),
        )
        nn.init.xavier_uniform_(self.cond_mlp[1].weight, gain=0.1)
        nn.init.zeros_(self.cond_mlp[1].bias)

        self.norm = nn.RMSNorm(d_model)
        self.pos_emb = SinusoidalPositionalEncoding(d_model)

        self.mamba_fwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cond, mask=None):
        film = self.cond_mlp(cond)
        gamma, beta, gate = film.chunk(3, dim=-1)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        if gate.dim() == 2:
            gate = gate.unsqueeze(1)
        alpha = torch.sigmoid(gate)

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            x = x * m
        else:
            m = None

        h_norm = self.norm(x)
        h_mod = h_norm * (1.0 + gamma) + beta
        if m is not None:
            h_mod = h_mod * m

        pos = self.pos_emb(h_mod)
        h_pos = h_mod + pos
        if m is not None:
            h_pos = h_pos * m

        h_ssm = self.mamba_fwd(h_pos)
        if m is not None:
            h_ssm = h_ssm * m

        update = self.out_proj(h_ssm)
        if m is not None:
            update = update * m

        out = x + alpha * update
        if m is not None:
            out = out * m
        return out
