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

from mambaflow_mamba_convnext_decoder import (
    BiMamba2PointwiseMLPBlock,
    ConvNeXtBlock,
    SinusoidalPositionalEncoding,
    reverse_valid_sequence,
    Mamba2ForwardBlock,
)


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


class MambaFlowTetraDecoder(nn.Module):
    """
    Optimal Transport CFM Velocity Decoder with 4-Layer Alternating Architecture:
    - Main Flow Highway Pathway: x_proj(concat([x_t, mu])) [80 + 80 = 160 -> 384].
    - Timestep Conditioning Alone:
        * t_emb = TimestepEmbedding(256)(t)
        * condition = t_proj(t_emb).unsqueeze(1) [B, 1, 384]
        * Condition modulates each layer via FiLM (gamma, beta, alpha).
    - 4 Alternating Layers at native 384-dim resolution:
        * Layer 1: BiMamba2PointwiseMLPBlock (384-dim fwd/bwd SSM -> concat 768 -> MLP 768->1024->384)
        * Layer 2: ConvNeXtBlock (dim=384, k=7, mult=4: 384 -> 1536 -> 384)
        * Layer 3: BiMamba2PointwiseMLPBlock (384-dim fwd/bwd SSM -> concat 768 -> MLP 768->1024->384)
        * Layer 4: ConvNeXtBlock (dim=384, k=7, mult=4: 384 -> 1536 -> 384)
    - Output Head: RMSNorm(384) -> Linear(384, 80, bias=False) predicting CFM velocity v.
    - Strictly pad-token mask aware across all operations.
    """
    def __init__(
        self,
        d_in=80,
        d_cond=80,
        d_out=80,
        d_model=384,
        d_t=256,
        mlp_hidden_dim=1024,
        sigma_min=1e-4,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_cond = d_cond
        self.d_out = d_out
        self.d_model = d_model
        self.d_t = d_t
        self.mlp_hidden_dim = mlp_hidden_dim
        self.sigma_min = sigma_min
        self.gradient_checkpointing = gradient_checkpointing

        # 1. Main Flow Highway Pathway: concat([x_t, mu]) [80 + 80 = 160 -> 256]
        self.x_proj = nn.Linear(d_in + d_cond, d_model, bias=False)

        # 2. Timestep Conditioning Alone (no acoustic prior in FiLM)
        self.t_embed = TimestepEmbedding(d_t=d_t)
        self.t_proj = nn.Linear(d_t, d_model, bias=True)

        # 3. 4-Layer Alternating Stack: BiMamba2+MLP -> ConvNeXt -> BiMamba2+MLP -> ConvNeXt
        self.layer1 = BiMamba2PointwiseMLPBlock(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.layer2 = ConvNeXtBlock(
            dim=d_model,
            kernel_size=7,
            mult=4,
        )
        self.layer3 = BiMamba2PointwiseMLPBlock(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.layer4 = ConvNeXtBlock(
            dim=d_model,
            kernel_size=7,
            mult=4,
        )

        # 4. Output Velocity Head
        self.final_norm = nn.RMSNorm(d_model)
        self.final_out_proj = nn.Linear(d_model, d_out, bias=False)
        nn.init.xavier_uniform_(self.final_out_proj.weight, gain=0.1)

    def _run_block(self, block, h, cond, mask):
        if self.training and self.gradient_checkpointing:
            return checkpoint(block, h, cond, mask, use_reentrant=False)
        return block(h, cond, mask=mask)

    def predict_velocity(self, x, t, mu, mask=None):
        """
        Predicts CFM velocity v_theta at time t given flow state x and acoustic prior mu.
        - Flow state: h = x_proj(concat([x, mu]))
        - Timestep conditioning alone: condition = t_proj(t_emb)
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
            mu = mu * m
        else:
            m = None

        # 2. Main flow representation (highway concatenation with mu)
        x_in = torch.cat([x, mu], dim=-1)
        h = self.x_proj(x_in)
        if m is not None:
            h = h * m

        # 3. Timestep embedding projection alone for conditioning
        t_emb = self.t_embed(t)               # [B, d_t]
        condition = self.t_proj(t_emb).unsqueeze(1)  # [B, 1, d_model]

        # 4. 4-Layer Alternating Processing (BiMamba2+MLP -> ConvNeXt -> BiMamba2+MLP -> ConvNeXt)
        h = self._run_block(self.layer1, h, condition, mask)
        h = self._run_block(self.layer2, h, condition, mask)
        h = self._run_block(self.layer3, h, condition, mask)
        h = self._run_block(self.layer4, h, condition, mask)

        # 5. Final output velocity projection
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
MambaFlowDecoder = MambaFlowTetraDecoder
