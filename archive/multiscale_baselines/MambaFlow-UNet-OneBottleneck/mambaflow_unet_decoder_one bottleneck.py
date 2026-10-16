import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2


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
    MLP-based timestep embedding projection.
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


class BiMamba2ConformerBlock(nn.Module):
    """
    Bidirectional Mamba2 + Large-Kernel Depthwise Convolution Block with AdaLN-Zero Modulation:
    - Global temporal context via Bidirectional Mamba2 (forward & backward streams with gated fusion)
    - Local harmonic formant continuity via Depthwise Conv1D with large kernel (k=17)
    - AdaLN-Zero modulation driven by timestep t (predicts scale gamma, shift beta, gate alpha)
    """
    def __init__(self, d_model=256, d_t=256, d_state=64, d_conv=4, expand=2, headdim=64, kernel_size=17, dropout=0.1):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)

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
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Linear(2 * d_model, d_model, bias=False)

        # Depthwise Conv1D with large kernel for harmonic formant continuity across mel channels
        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.act = nn.SiLU()
        self.pw_conv = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # AdaLN-Zero: predicts scale (gamma), shift (beta), and residual gate (alpha)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t, 3 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x, t_emb=None, mask=None):
        # x is [B, T, d_model]
        res = x

        # AdaLN modulation
        if t_emb is not None:
            gamma, beta, alpha = self.adaln(t_emb).unsqueeze(1).chunk(3, dim=-1)
            h = (1.0 + gamma) * self.norm(x) + beta
        else:
            alpha = 1.0
            h = self.norm(x)

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            h = h * m
        else:
            m = None

        # 1. Forward scan (left -> right)
        h_fwd = self.mamba_fwd(h)
        if m is not None:
            h_fwd = h_fwd * m

        # 2. Backward scan (right -> left)
        h_rev = reverse_valid_sequence(h, mask=m)
        h_bwd_rev = self.mamba_bwd(h_rev)
        h_bwd = reverse_valid_sequence(h_bwd_rev, mask=m)
        if m is not None:
            h_bwd = h_bwd * m

        # 3. Gated bidirectional fusion
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        gate = self.gate_proj(h_cat)
        fused = self.fusion_proj(h_cat) * gate

        # 4. Large-Kernel Depthwise Convolution for harmonic formant continuity
        h_conv = self.dw_conv(fused.transpose(1, 2)).transpose(1, 2)
        if m is not None:
            h_conv = h_conv * m
        h_conv = self.act(self.norm2(h_conv))
        h_conv = self.pw_conv(h_conv)
        h_conv = self.dropout(h_conv)
        if m is not None:
            h_conv = h_conv * m

        return res + alpha * h_conv


class ConvNeXtAdaLNBlock(nn.Module):
    """
    ConvNeXt-1D Large-Kernel Local Refinement Block with AdaLN-Zero Timestep Conditioning:
    - Depthwise Conv1D with large kernel (e.g. k=31 or 17) for local temporal & harmonic continuity
    - AdaLN-Zero modulation: predicts scale (gamma), shift (beta), and residual gate (alpha) from t_emb
    - Inverted Bottleneck: Linear (dim -> 4*dim) -> SiLU -> Dropout -> Linear (4*dim -> dim)
    - Zero-initialized AdaLN-Zero projection guarantees smooth identity start at initialization
    """
    def __init__(self, dim=192, d_t=192, kernel_size=17, mlp_ratio=4, dropout=0.0):
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

        # AdaLN-Zero: predicts scale (gamma), shift (beta), and residual gate (alpha)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t, 3 * dim, bias=True),
        )
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x, t_emb=None, mask=None):
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

        if t_emb is not None:
            gamma, beta, alpha = self.adaln(t_emb).unsqueeze(1).chunk(3, dim=-1)
            h = (1.0 + gamma) * self.norm(h) + beta
        else:
            alpha = 1.0
            h = self.norm(h)

        if m_c is not None:
            h = h * m_c.float()

        h = self.pw1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.pw2(h)

        if m_c is not None:
            h = h * m_c.float()

        return res + alpha * h


class Downsample1D(nn.Module):
    """
    Symmetric 1D Downsampling layer (stride 2) using Conv1D:
    Halves temporal resolution while doubling receptive field smoothly.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # x: [B, T, C] -> transpose to [B, C, T] -> conv -> transpose back to [B, T//2, C_out]
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class Upsample1D(nn.Module):
    """
    Symmetric 1D Upsampling layer (stride 2) using ConvTranspose1D:
    Doubles temporal resolution smoothly with exact length matching to avoid phase shift.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, target_len=None):
        # x: [B, T, C] -> transpose to [B, C, T] -> upsample -> transpose back
        h = self.conv_transpose(x.transpose(1, 2)).transpose(1, 2)
        if target_len is not None and h.size(1) != target_len:
            if h.size(1) > target_len:
                h = h[:, :target_len, :]
            else:
                h = F.pad(h, (0, 0, 0, target_len - h.size(1)), value=0.0)
        return h


class MambaFlowUNetDecoder(nn.Module):
    """
    MambaFlow-UNet Decoder:
    Multi-Scale Temporal U-Net with Bidirectional Mamba2 SSMs, Multi-Scale AdaLN-Zero ConvNeXt layers,
    and 192 -> 256 -> 192 Channel Scaling.

    Key Architectural Advantages:
    1. Multi-Scale Temporal Pyramid (1x -> 1/2x -> 1/4x -> 1/2x -> 1x):
       Local harmonic formant continuity via ConvNeXt blocks at 1x (k=31) and 1/2x (k=17),
       and deep global context via Bidirectional Mamba2 at 1/4x (k=17).
    2. Dynamic Channel Scaling (192 -> 256 -> 192):
       Expands internal representation capacity during downsampling and restores it smoothly
       during upsampling, preventing high-frequency metallic artifacts and phase distortion.
    3. Zero Phase Distortion (Anti-Dual Audio Guarantee):
       Symmetric multiple-of-4 padding ensures exact mathematical alignment across all skip connections.
    4. Pure Linear-Time Global Modeling:
       Replaces quadratic O(T^2) self-attention with linear O(T) Bidirectional Mamba2 SSMs.
    5. Full Mel Velocity Flow Matching:
       Targets full ground truth mel directly, eliminating additive residual leaks.
    """
    def __init__(
        self,
        d_in=100,
        d_cond=100,
        d_out=100,
        d_model_1=192,
        d_model_2=256,
        d_model_mid=256,
        d_t=192,
        kernel_size_1=31,
        kernel_size_2=17,
        kernel_size_mid=17,
        dropout=0.1,
        sigma_min=1e-4,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_cond = d_cond
        self.d_out = d_out
        self.d_t = d_t
        self.sigma_min = sigma_min
        self.gradient_checkpointing = gradient_checkpointing

        # Timestep embedding network (scalar t in [0, 1] -> d_t vector)
        self.t_embed = TimestepEmbedding(d_t=d_t)

        # Level 1 (Resolution 1x): Input projection + Large Kernel ConvNeXt (k=31)
        self.in_proj = nn.Linear(d_in + d_cond, d_model_1, bias=False)
        self.down_conv1 = ConvNeXtAdaLNBlock(dim=d_model_1, d_t=d_t, kernel_size=kernel_size_1, dropout=dropout)

        # Downsample 1: 1x -> 1/2x (scale channels from 192 -> 256)
        self.down1 = Downsample1D(d_model_1, d_model_2)

        # Level 2 (Resolution 1/2x): Large Kernel ConvNeXt (k=17, dim=256)
        self.down_conv2 = ConvNeXtAdaLNBlock(dim=d_model_2, d_t=d_t, kernel_size=kernel_size_2, dropout=dropout)

        # Downsample 2: 1/2x -> 1/4x (256 -> 256)
        self.down2 = Downsample1D(d_model_2, d_model_mid)

        # Bottleneck (Resolution 1/4x): Deep global context + Large Kernel (k=17)
        self.mid_blocks = nn.ModuleList([
            BiMamba2ConformerBlock(d_model=d_model_mid, d_t=d_t, kernel_size=kernel_size_mid, dropout=dropout)
            for _ in range(1)
        ])

        # Upsample 2: 1/4x -> 1/2x (256 -> 256) with skip connection fusion + ConvNeXt (k=17)
        self.up2 = Upsample1D(d_model_mid, d_model_2)
        self.dec_level2_fuse = nn.Linear(d_model_2 + d_model_2, d_model_2, bias=False)
        self.dec_level2_act = nn.SiLU()
        self.up_conv2 = ConvNeXtAdaLNBlock(dim=d_model_2, d_t=d_t, kernel_size=kernel_size_2, dropout=dropout)

        # Upsample 1: 1/2x -> 1x (scale channels from 256 back to 192) with skip connection fusion + ConvNeXt (k=31)
        self.up1 = Upsample1D(d_model_2, d_model_1)
        self.dec_level1_fuse = nn.Linear(d_model_1 + d_model_1, d_model_1, bias=False)
        self.dec_level1_act = nn.SiLU()
        self.up_conv1 = ConvNeXtAdaLNBlock(dim=d_model_1, d_t=d_t, kernel_size=kernel_size_1, dropout=dropout)

        # Final projection to output velocity v_theta
        self.final_norm = nn.RMSNorm(d_model_1)
        self.final_out_proj = nn.Linear(d_model_1, d_out, bias=False)
        nn.init.zeros_(self.final_out_proj.weight)

    def _run_block(self, block, h, t_emb, mask):
        if self.training and self.gradient_checkpointing:
            return checkpoint(block, h, t_emb, mask, use_reentrant=False)
        return block(h, t_emb=t_emb, mask=mask)

    def predict_velocity(self, x, t, mu, mask=None):
        """
        Evaluates the learned velocity field v_theta(x, t, mu).
        x:    [B, T, 100] (noisy state at timestep t)
        t:    [B]         (timestep scalar in [0, 1])
        mu:   [B, T, 100] (acoustic prior conditioning from text encoder)
        mask: [B, T]      (boolean audio mask)
        """
        B, T, _ = x.size()

        # =====================================================================
        # Symmetric Multiple-of-4 Padding (Anti-Dual Audio Guarantee):
        # Eliminates odd-length rounding discrepancies that cause time-shifted
        # skip-connection misalignment and phase cancellation/dual audio artifacts.
        # =====================================================================
        pad = (4 - (T % 4)) % 4
        if pad > 0:
            x_in_pad = F.pad(x, (0, 0, 0, pad), value=0.0)
            mu_pad = F.pad(mu, (0, 0, 0, pad), value=0.0)
            if mask is not None:
                mask_pad = F.pad(mask, (0, pad), value=False)
            else:
                mask_pad = None
        else:
            x_in_pad = x
            mu_pad = mu
            mask_pad = mask

        # Timestep embedding
        t_emb = self.t_embed(t)

        # Initial feature concatenation & projection [B, T_pad, d_model_1]
        cat_in = torch.cat([x_in_pad, mu_pad], dim=-1)
        h1 = self.in_proj(cat_in)
        if mask_pad is not None:
            m1 = mask_pad.unsqueeze(-1).float()
            h1 = h1 * m1
        else:
            m1 = None

        # Level 1 Down (1x resolution): ConvNeXt layer
        h1 = self._run_block(self.down_conv1, h1, t_emb, mask_pad)
        skip1 = h1
        len_lvl1 = h1.size(1)

        # Downsample 1: 1x -> 1/2x (channel dimension increases from 192 to 256)
        h2 = self.down1(h1)
        len_lvl2 = h2.size(1)
        mask_lvl2 = mask_pad[:, ::2] if mask_pad is not None else None
        if mask_lvl2 is not None:
            h2 = h2 * mask_lvl2.unsqueeze(-1).float()

        # Level 2 Down (1/2x resolution): ConvNeXt layer (256-dim)
        h2 = self._run_block(self.down_conv2, h2, t_emb, mask_lvl2)
        skip2 = h2

        # Downsample 2: 1/2x -> 1/4x (256 -> 256)
        h_mid = self.down2(h2)
        mask_mid = mask_lvl2[:, ::2] if mask_lvl2 is not None else None
        if mask_mid is not None:
            h_mid = h_mid * mask_mid.unsqueeze(-1).float()

        # Bottleneck (1/4x resolution): Bidirectional Mamba2 SSM
        for blk in self.mid_blocks:
            h_mid = self._run_block(blk, h_mid, t_emb, mask_mid)

        # Upsample 2: 1/4x -> 1/2x (256 -> 256) with skip connection fusion & ConvNeXt
        h2_up = self.up2(h_mid, target_len=len_lvl2)
        h2_fused = self.dec_level2_act(self.dec_level2_fuse(torch.cat([h2_up, skip2], dim=-1)))
        if mask_lvl2 is not None:
            h2_fused = h2_fused * mask_lvl2.unsqueeze(-1).float()
        h2_fused = self._run_block(self.up_conv2, h2_fused, t_emb, mask_lvl2)

        # Upsample 1: 1/2x -> 1x (256 -> 192) with skip connection fusion & ConvNeXt
        h1_up = self.up1(h2_fused, target_len=len_lvl1)
        h1_fused = self.dec_level1_act(self.dec_level1_fuse(torch.cat([h1_up, skip1], dim=-1)))
        if mask_pad is not None:
            h1_fused = h1_fused * m1
        h1_fused = self._run_block(self.up_conv1, h1_fused, t_emb, mask_pad)

        # Final projection to velocity
        h_out = self.final_norm(h1_fused)
        v = self.final_out_proj(h_out)
        v = self.final_out_proj(h_out)

        # Unpad back to exact original sequence length T
        if pad > 0:
            v = v[:, :T, :]

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            v = v * m

        return v

    @torch.no_grad()
    def solve_euler(self, mu, mask=None, n_timesteps=30, temperature=1.0, solver="euler", t_end=1.0):
        """
        Integrates ODE dx_t/dt = v_theta(x_t, t, mu) from t=0 to t=t_end.
        Initial condition: standard Gaussian noise x_0 ~ N(0, (temperature)^2 * I).
        Supports:
          - solver="euler": 1st-Order standard Euler ODE solver (1 evaluation per step)
          - solver="midpoint": 2nd-Order Heun's Runge-Kutta Midpoint ODE solver (2 evaluations per step)
        """
        B, T, _ = mu.size()
        device = mu.device
        dtype = mu.dtype

        # Initial state: pure standard normal noise scaled by temperature
        x = torch.randn_like(mu) * temperature

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            x = x * m
        else:
            m = None

        dt = t_end / float(n_timesteps)

        for step in range(n_timesteps):
            t_val = step * t_end / float(n_timesteps)
            t = torch.full((B,), t_val, device=device, dtype=dtype)

            # Velocity at current step
            v = self.predict_velocity(x, t, mu, mask=mask)

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

        # Returns the predicted full mel spectrogram directly
        return x

    def forward(self, mu, mask=None, target=None, n_timesteps=30, temperature=1.0, solver="euler", t_end=1.0):
        """
        Unified forward interface:
        - Training Mode (target is not None):
            Computes conditional flow matching velocity target and prediction on full mel.
        - Inference Mode (target is None):
            Solves the ODE from pure noise to generate full mel spectrogram directly.
        """
        B, T, _ = mu.size()

        if target is not None:
            # Training Mode (Full Mel Continuous Flow Matching)
            if target.size(1) != T:
                min_len = min(T, target.size(1))
                target = target[:, :min_len, :]
                mu = mu[:, :min_len, :]
                if mask is not None:
                    mask = mask[:, :min_len]

            # Sample continuous uniform timestep t in [0, 1]
            t = torch.rand(B, device=target.device, dtype=target.dtype)

            # Standard Gaussian initial noise distribution x_0 ~ N(0, I)
            x_0 = torch.randn_like(target)

            # Straight-line flow matching trajectory:
            # x_t = (1 - (1 - sigma_min)*t) * x_0 + t * target
            t_expand = t.view(B, 1, 1)
            x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x_0 + t_expand * target

            # Analytical target velocity: u_t = target - (1 - sigma_min) * x_0
            u_target = target - (1.0 - self.sigma_min) * x_0

            # Predict velocity
            v_pred = self.predict_velocity(x_t, t, mu, mask=mask)

            return v_pred, u_target
        else:
            # Inference Mode: Solve ODE to generate complete mel
            return self.solve_euler(
                mu=mu,
                mask=mask,
                n_timesteps=n_timesteps,
                temperature=temperature,
                solver=solver,
                t_end=t_end,
            )


# Export aliases
MambaUNetDecoder = MambaFlowUNetDecoder
decoder = MambaFlowUNetDecoder
