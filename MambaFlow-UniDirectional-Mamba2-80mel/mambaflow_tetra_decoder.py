import os
import sys
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2
from x_transformers import Attention
from x_transformers.x_transformers import RotaryEmbedding

from mambaflow_mamba_convnext_decoder import Mamba2ForwardBlock, ConvNeXtBlock


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal timestep embedding for continuous time t in [0, 1].
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * 1000.0 * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """
    Compact MLP-based timestep embedding projection.
    """
    def __init__(self, d_t=256):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(d_t)
        self.mlp = nn.Sequential(
            nn.Linear(d_t, d_t, bias=True),
            nn.SiLU(),
            nn.Linear(d_t, d_t, bias=True),
        )

    def forward(self, t):
        t = t.view(-1, 1)
        emb = self.sin_emb(t)
        return self.mlp(emb)


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


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for 1D sequence tokens/frames (positions 0, 1, ..., T-1).
    Provides sequence position awareness to bidirectional Mamba-2 SSM layers.
    """
    def __init__(self, dim, max_len=10000):
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


class CNNFeedForward(nn.Module):
    """
    ConvNeXt-style 1D Depthwise-Separable CNN FeedForward Network:
    - 1D Depthwise Convolution (kernel_size=5) for local temporal acoustic correlation
    - Pointwise Inverted Bottleneck Linear projection: dim -> mult * dim
    - Non-linear Activation (SiLU)
    - Pointwise Linear projection: mult * dim -> dim
    - Sequence masking to ensure padded tokens remain strictly zeroed
    """
    def __init__(self, dim, mult=4, kernel_size=5, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )
        self.pw1 = nn.Linear(dim, mult * dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.pw2 = nn.Linear(mult * dim, dim, bias=False)

    def forward(self, x, mask=None):
        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            x = x * m
        else:
            m = None

        h = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        if m is not None:
            h = h * m

        h = self.pw1(h)
        h = self.act(h)
        h = self.dropout(h)
        out = self.pw2(h)
        if m is not None:
            out = out * m

        return out


class TransformerBlock(nn.Module):
    """
    Hybrid Transformer Block (dim=384, 6 heads, head_dim=64):
    - Non-causal Flash Self-Attention with Rotary Position Embedding (RoPE)
    - ConvNeXt-style 1D Depthwise Temporal CNN FeedForward (kernel_size=5)
    - Layer-specific FiLM adaptive conditioning: gamma, beta, gate = cond_mlp(condition).chunk(3, dim=-1)
    - Modulation: h_mod = RMSNorm(x) * (1 + gamma) + beta
    - Bounded sigmoid gate: h = x + sigmoid(gate) * block_update
    """
    def __init__(self, dim=384, n_heads=6, kernel_size=5, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        dim_head = dim // n_heads

        # Layer-specific conditioning projection producing gamma, beta, gate
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True),
        )
        nn.init.xavier_uniform_(self.cond_mlp[1].weight, gain=0.1)
        nn.init.zeros_(self.cond_mlp[1].bias)

        self.norm1 = nn.RMSNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim=min(32, dim_head))
        self.attn = Attention(
            dim=dim,
            heads=n_heads,
            dim_head=dim_head,
            dropout=dropout,
            causal=False,
            flash=True,
            zero_init_output=False,
        )
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = CNNFeedForward(
            dim=dim,
            mult=mlp_ratio,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x, cond, mask=None):
        film = self.cond_mlp(cond)
        gamma, beta, gate = film.chunk(3, dim=-1)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        if gate.dim() == 2:
            gate = gate.unsqueeze(1)
        gate = torch.sigmoid(gate)

        if mask is not None:
            m_bool = mask.bool() if mask.dim() == 2 else mask.squeeze(-1).bool()
            m_c = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
        else:
            m_bool = None
            m_c = None

        # 1. Modulated Self-Attention
        h_norm1 = self.norm1(x)
        h_mod = h_norm1 * (1.0 + gamma) + beta
        if m_c is not None:
            h_mod = h_mod * m_c

        seq_len = x.shape[1]
        rotary_pos = self.rotary_emb(torch.arange(seq_len, device=x.device))
        attn_out = self.attn(h_mod, mask=m_bool, rotary_pos_emb=rotary_pos)
        if m_c is not None:
            attn_out = attn_out * m_c

        # 2. CNN FeedForward
        h_mid = self.norm2(x + attn_out)
        mlp_out = self.mlp(h_mid, mask=m_bool)
        if m_c is not None:
            mlp_out = mlp_out * m_c

        block_update = attn_out + mlp_out
        out = x + gate * block_update
        if m_c is not None:
            out = out * m_c

        return out


class BiMamba2GatedFusionBlock(nn.Module):
    """
    Bidirectional Mamba-2 Block (d_model=384, expand=2, d_state=64, d_conv=4, headdim=64):
    - Sinusoidal Positional Encoding for sequence frame awareness
    - Forward scan + Valid-reversal backward scan
    - Gated bidirectional fusion
    - Layer-specific FiLM adaptive conditioning: gamma, beta, gate = cond_mlp(condition).chunk(3, dim=-1)
    - Modulation: h_mod = RMSNorm(x) * (1 + gamma) + beta
    - Bounded sigmoid gate: h = x + sigmoid(gate) * block_update
    """
    def __init__(
        self,
        d_model=384,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
    ):
        super().__init__()
        self.d_model = d_model

        # Layer-specific conditioning projection producing gamma, beta, gate
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
        self.mamba_bwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Sigmoid(),
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
        gate = torch.sigmoid(gate)

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
        else:
            m = None

        # Modulation before temporal processing
        h_norm = self.norm(x)
        h_mod = h_norm * (1.0 + gamma) + beta
        if m is not None:
            h_mod = h_mod * m

        # Positional Encoding for Mamba SSM
        h_pos = h_mod + self.pos_emb(h_mod)
        if m is not None:
            h_pos = h_pos * m

        # 1. Forward scan (left -> right)
        h_fwd = self.mamba_fwd(h_pos)
        if m is not None:
            h_fwd = h_fwd * m

        # 2. Backward scan (right -> left)
        h_rev = reverse_valid_sequence(h_pos, mask=m)
        h_bwd_rev = self.mamba_bwd(h_rev)
        h_bwd = reverse_valid_sequence(h_bwd_rev, mask=m)
        if m is not None:
            h_bwd = h_bwd * m

        # 3. Gated bidirectional fusion
        h_sum = h_fwd + h_bwd
        fusion_gate = self.gate_proj(h_sum)
        fused = h_sum * fusion_gate
        block_update = self.out_proj(fused)
        if m is not None:
            block_update = block_update * m

        out = x + gate * block_update
        if m is not None:
            out = out * m

        return out


# Backward compatibility alias
TransformerAdaLNBlock = TransformerBlock


class MambaFlowTetraDecoder(nn.Module):
    """
    Decoupled Optimal Transport CFM Velocity Decoder with 4-Layer Interleaved Architecture:
    - Independent Main Flow Pathway: x_proj(x_t) [100 -> 384]. NO direct concatenation of [x_t, mu].
    - Independent Conditioning Pathway with Coarse-to-Fine Mu Routing:
        * mu_proj(mu) [100 -> 384]
        * low = MaskAwareAvgPool1d(k=5, s=1, p=2)(mu_hidden)
        * high = mu_hidden - low
        * bandwidth = t ** 2
        * routed_mu = low + bandwidth * high
        * cond = cond_proj(routed_mu) [384 -> 384]
    - Timestep Conditioning:
        * t_emb = TimestepEmbedding(256)(t)
        * t_cond = t_proj(t_emb) [256 -> 384]
        * condition = cond + t_cond [B, T, 384]
    - Layer-Specific FiLM Adaptive Modulation at every layer:
        * Layer 1: TransformerBlock (dim=384, heads=6, Flash Attention, RoPE, CNN-FFN k=5)
        * Layer 2: BiMamba2GatedFusionBlock (d_model=384, expand=2, d_state=64, d_conv=4, headdim=64)
        * Layer 3: TransformerBlock (dim=384, heads=6, Flash Attention, RoPE, CNN-FFN k=5)
        * Layer 4: BiMamba2GatedFusionBlock (d_model=384, expand=2, d_state=64, d_conv=4, headdim=64)
    - Output Head: RMSNorm(384) -> Linear(384, 100) predicting CFM velocity v.
    - Pure CFM formulation: training target is flow_target (full mel), inference is continuous ODE integration.
    """
    def __init__(
        self,
        d_in=100,
        d_cond=100,
        d_out=100,
        d_model=384,
        d_t=256,
        n_heads=6,
        sigma_min=1e-4,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_cond = d_cond
        self.d_out = d_out
        self.d_model = d_model
        self.d_t = d_t
        self.n_heads = n_heads
        self.sigma_min = sigma_min
        self.gradient_checkpointing = gradient_checkpointing

        # 1. Main Flow Pathway (Input Highway)
        self.x_proj = nn.Linear(d_in + d_cond, d_model, bias=False)

        # 3. Timestep Conditioning
        self.t_embed = TimestepEmbedding(d_t=d_t)
        self.t_proj = nn.Linear(d_t, d_model, bias=True)

        # 4. Exact 4-Layer Interleaved Stack: Transformer -> BiMamba2 -> Transformer -> BiMamba2
        # self.layer1 = TransformerBlock(dim=d_model, n_heads=n_heads, kernel_size=5)
        # self.layer2 = BiMamba2GatedFusionBlock(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64)
        # self.layer3 = TransformerBlock(dim=d_model, n_heads=n_heads, kernel_size=5)
        # self.layer4 = BiMamba2GatedFusionBlock(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64)

        self.layer1 = Mamba2ForwardBlock(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64)
        self.layer2 = ConvNeXtBlock(dim=d_model, kernel_size=7, mult=4)
        self.layer3 = Mamba2ForwardBlock(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64)
        self.layer4 = ConvNeXtBlock(dim=d_model, kernel_size=7, mult=4)

        # 5. Output Velocity Head
        self.final_norm = nn.RMSNorm(d_model)
        self.final_out_proj = nn.Linear(d_model, d_out, bias=False)
        nn.init.xavier_uniform_(self.final_out_proj.weight, gain=0.1)



    def _run_block(self, block, h, cond, mask):
        if self.training and self.gradient_checkpointing:
            return checkpoint(block, h, cond, mask, use_reentrant=False)
        return block(h, cond, mask=mask)

    def predict_velocity(self, x, t, mu, mask=None):
        """
        Predicts CFM velocity v_theta at time t given noisy state x and acoustic prior mu.
        - Flow state: h = x_proj(x)
        - Condition: condition = cond_proj(routed_mu) + t_proj(t_emb)
        - 4 layers modulated independently
        """
        B, T, _ = x.size()

        # 1. Align sequence lengths if needed
        if mu.size(1) != T:
            min_T = min(T, mu.size(1))
            x = x[:, :min_T, :]
            mu = mu[:, :min_T, :]
            if mask is not None:
                mask = mask[:, :min_T]
            T = min_T

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            x = x * m
        else:
            m = None

        # 2. Main flow representation (concatenated with mu)
        x_in = torch.cat([x, mu], dim=-1)
        h = self.x_proj(x_in)
        if m is not None:
            h = h * m

        # 3. Timestep embedding projection
        t_emb = self.t_embed(t)               # [B, d_t]
        condition = self.t_proj(t_emb).unsqueeze(1)  # [B, 1, d_model]
        if m is not None:
            condition = condition * m

        # 5. Exact 4-Layer Interleaved Processing
        h = self._run_block(self.layer1, h, condition, mask)
        h = self._run_block(self.layer2, h, condition, mask)
        h = self._run_block(self.layer3, h, condition, mask)
        h = self._run_block(self.layer4, h, condition, mask)

        # 6. Final output velocity projection
        h_norm = self.final_norm(h)
        v = self.final_out_proj(h_norm)
        if m is not None:
            v = v * m

        return v

    @torch.no_grad()
    def solve_euler(self, mu, mask=None, n_timesteps=50, temperature=0.667, solver="midpoint", t_end=1.0):
        """
        Continuous ODE integration solving dx/dt = v_theta(x, t, mu) from t=0 to t=1.
        Initial state x_0 is Gaussian noise scaled by temperature.
        """
        B, T, _ = mu.size()
        device = mu.device
        dtype = mu.dtype

        # Initial state from pure Gaussian noise
        x = torch.randn_like(mu) * temperature

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            x = x * m
        else:
            m = None

        dt = float(t_end) / float(n_timesteps)

        for step in range(n_timesteps):
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)

            # Predict velocity at current flow state
            v = self.predict_velocity(x, t_tensor, mu, mask=mask)

            if solver == "midpoint":
                if step == n_timesteps - 1:
                    x = x + dt * v
                else:
                    x_mid = x + (dt / 2.0) * v
                    if m is not None:
                        x_mid = x_mid * m
                    t_mid = torch.full((B,), t_val + dt / 2.0, device=device, dtype=dtype)
                    v_mid = self.predict_velocity(x_mid, t_mid, mu, mask=mask)
                    x = x + dt * v_mid
            else:
                x = x + dt * v

            if m is not None:
                x = x * m

        return x

    def forward(self, mu, mask=None, target=None, n_timesteps=50, temperature=0.667, solver="midpoint", t_end=1.0):
        B, T, _ = mu.size()

        if target is not None:
            # Training Mode: Direct Full Optimal Transport Flow Matching
            if target.size(1) != T:
                min_len = min(T, target.size(1))
                target = target[:, :min_len, :]
                mu = mu[:, :min_len, :]
                if mask is not None:
                    mask = mask[:, :min_len]

            t = torch.rand(B, device=target.device, dtype=target.dtype)
            x_0 = torch.randn_like(target)

            t_expand = t.view(B, 1, 1)
            x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x_0 + t_expand * target
            u_target = target - (1.0 - self.sigma_min) * x_0
            v_pred = self.predict_velocity(x_t, t, mu, mask=mask)

            return v_pred, u_target
        else:
            # Inference Mode: Continuous ODE integration from noise to mel
            return self.solve_euler(
                mu=mu,
                mask=mask,
                n_timesteps=n_timesteps,
                temperature=temperature,
                solver=solver,
                t_end=t_end,
            )


# Backward and export aliases
decoder = MambaFlowTetraDecoder
MambaFlowParallelDecoder = MambaFlowTetraDecoder
