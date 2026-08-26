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
    def __init__(self, d_t=128):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(d_t)
        self.mlp = nn.Sequential(
            nn.Linear(d_t, d_t),
            nn.SiLU(),
            nn.Linear(d_t, d_t),
        )

    def forward(self, t):
        t = t.view(-1, 1)
        emb = self.sin_emb(t)
        return self.mlp(emb)


def reverse_valid_sequence(x, mask=None):
    """
    Reverses sequence x along time dimension (dim=1) only for valid non-padded tokens.
    Uses pure tensor gather without in-place slice mutation for fast autograd backward execution.
    """
    if mask is None:
        return torch.flip(x, dims=[1])
    
    B, T, D = x.shape
    m_bool = mask.bool() if mask.dim() == 2 else mask.squeeze(-1).bool()
    valid_lengths = m_bool.sum(dim=1)
    
    indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1).clone()
    for b in range(B):
        length = valid_lengths[b].item()
        if length > 1:
            indices[b, :length] = torch.arange(length - 1, -1, -1, device=x.device)
            
    idx_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(x, dim=1, index=idx_expanded)


class ConvMLPInputBlock(nn.Module):
    """
    Masked 1D Convolutional MLP input stage for concatenated inputs:
    [noise (72), mu_latent (72), spk_emb (64), time_emb (128)] -> Conv1D -> Mask -> SiLU -> Conv1D -> Mask -> Residual + RMSNorm.
    """
    def __init__(self, in_channels, d_model, hidden_dim=512, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv1d(hidden_dim, d_model, kernel_size=kernel_size, padding=padding, bias=False)
        self.res_proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x, mask=None):
        # x: [B, T, in_channels]
        x_t = x.transpose(1, 2)
        if mask is not None:
            m = mask.unsqueeze(1).float() if mask.dim() == 2 else mask.float()
            m_c = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            x_t = x_t * m
        else:
            m = None
            m_c = None

        res = self.res_proj(x_t)
        h = self.conv1(x_t)
        if m is not None:
            h = h * m
        h = self.act1(h)
        h = self.conv2(h)
        if m is not None:
            h = h * m

        out = (h + res).transpose(1, 2)
        out = self.norm(out)
        if m_c is not None:
            out = out * m_c
        return out


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style purely unconditioned spatial mixing block.
    Uses depthwise 1D conv -> RMSNorm -> 4x pointwise -> SiLU -> pointwise.
    """
    def __init__(self, d_model, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, 
            padding=padding, dilation=dilation, groups=d_model, bias=False
        )
        self.norm = nn.RMSNorm(d_model)
        self.pw1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.act = nn.SiLU()
        self.pw2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x is [B, T, D]
        res = x
        x_t = x.transpose(1, 2)  # [B, D, T]
        
        if mask is not None:
            # mask from BiMamba2Block is already [B, T, 1]
            # Convert to [B, 1, T] to broadcast over [B, D, T]
            m = mask.transpose(1, 2) if mask.dim() == 3 else mask.unsqueeze(1)
            x_t = x_t * m

        x_t = self.dwconv(x_t)
        
        x_out = x_t.transpose(1, 2)  # [B, T, D]
        x_out = self.norm(x_out)
        x_out = self.pw1(x_out)
        x_out = self.act(x_out)
        x_out = self.pw2(x_out)
        
        if mask is not None:
            # mask is [B, T, 1], broadcasts fine over [B, T, D]
            m_c = mask if mask.dim() == 3 else mask.unsqueeze(-1)
            x_out = x_out * m_c

        return res + x_out


class BiMamba2Block(nn.Module):
    """
    True Bidirectional Mamba2 Block with 3-Way AdaLN Conditioning:
    - Forward Mamba2: processes sequence left-to-right
    - Backward Mamba2: processes sequence right-to-left
    - Combined by residual addition: x + h_fwd + h_bwd
    - Per-layer AdaLN-Zero for:
      1) Timestep embedding (t_emb)
      2) Conditional Latent (cond / mu)
    """
    def __init__(self, d_model=256, d_t=128, d_cond=100, d_state=128, d_conv=4, expand=2, headdim=64, kernel_size=5, dilation=1):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        
        # Interleaved ConvNeXt Block for spatial mixing
        self.conv_next = ConvNeXtBlock(d_model=d_model, kernel_size=kernel_size, dilation=dilation)
        
        # Forward and backward Mamba2 streams
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

        # AdaLN-Zero for timestep
        self.adaln_t = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t, 2 * d_model),
        )
        nn.init.zeros_(self.adaln_t[1].weight)
        nn.init.zeros_(self.adaln_t[1].bias)

        # AdaLN-Zero for conditional latent
        self.adaln_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 2 * d_model),
        )
        nn.init.zeros_(self.adaln_cond[1].weight)
        nn.init.zeros_(self.adaln_cond[1].bias)

        # Mixing layer for forward and backward streams (commented out for epoch 104)
        # self.mixer = nn.Linear(2 * d_model, d_model)

    def forward(self, x, t_emb=None, cond=None, mask=None):
        h = self.norm(x)
        
        # AdaLN timestep conditioning
        if t_emb is not None:
            scale_t, shift_t = self.adaln_t(t_emb).unsqueeze(1).chunk(2, dim=-1)
            h = h * (1.0 + scale_t) + shift_t

        # AdaLN conditional latent conditioning
        if cond is not None:
            scale_c, shift_c = self.adaln_cond(cond).chunk(2, dim=-1)
            h = h * (1.0 + scale_c) + shift_c

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

        # Combine streams directly with residual x
        # h_concat = torch.cat([h_fwd, h_bwd], dim=-1)
        # h_mixed = self.mixer(h_concat)
        # if m is not None:
        #     h_mixed = h_mixed * m
        # h_mamba = x + h_mixed
        h_mamba = x + h_fwd + h_bwd        
        # Unconditioned local spatial mixing
        return self.conv_next(h_mamba, mask=mask)


class decoder(nn.Module):
    """
    Conditional Flow Matching (CFM) Velocity Predictor with True Bidirectional Mamba2.
    Predicts 100-dim continuous normalized mel velocity vectors.
    """
    def __init__(
        self,
        d_in=100,
        d_cond=100,
        d_out=100,
        d_model=256,
        d_t=128,
        n_layers=6,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        sigma_min=1e-4,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_cond = d_cond
        self.d_out = d_out
        self.d_model = d_model
        self.d_t = d_t
        self.n_layers = n_layers
        self.sigma_min = sigma_min
        self.gradient_checkpointing = gradient_checkpointing

        # Time embedding
        self.time_emb = TimestepEmbedding(d_t=d_t)

        # Masked Conv-1D MLP input stage: [x_t (100) + mu (100) + time_emb (128)] = 328 -> d_model
        in_channels = d_in + d_cond + d_t
        self.input_block = ConvMLPInputBlock(
            in_channels=in_channels,
            d_model=d_model,
            hidden_dim=512,
            kernel_size=3,
        )

        # Stack of clean True Bidirectional Mamba2 blocks with 3-way AdaLN and ConvNeXt
        # Using dilations 1 to prevent gridding/electric buzzing artifacts, and larger kernels to compensate receptive field
        dilations = [1, 1, 1, 1, 1, 1]
        kernel_sizes = [7, 7, 7, 9, 9, 9]
        
        self.layers = nn.ModuleList([
            BiMamba2Block(
                d_model=d_model,
                d_t=d_t,
                d_cond=d_cond,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                kernel_size=kernel_sizes[i % len(kernel_sizes)],
                dilation=dilations[i % len(dilations)],
            )
            for i in range(n_layers)
        ])

        self.norm_f = nn.RMSNorm(d_model)

        # Output projection: d_model -> d_out (100)
        self.final_out_proj = nn.Linear(d_model, d_out, bias=False)

    def predict_velocity(self, x_t, t, mu, mask=None):
        B, T, _ = x_t.size()

        # Time embedding
        t_emb = self.time_emb(t)
        t_emb_seq = t_emb.unsqueeze(1).expand(-1, T, -1)

        # Align length of mu if needed
        if mu.size(1) != T:
            if mu.size(1) > T:
                mu = mu[:, :T, :]
            else:
                mu = F.pad(mu, (0, 0, 0, T - mu.size(1)), value=0.0)

        # Concat: [100 + 100 + 128 = 328] -> Pass through masked Conv MLP block
        x_concat = torch.cat([x_t, mu, t_emb_seq], dim=-1)
        h = self.input_block(x_concat, mask=mask)

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            if m.size(1) > T:
                m = m[:, :T, :]
            elif m.size(1) < T:
                m = F.pad(m, (0, 0, 0, T - m.size(1)), value=0.0)
            h = h * m
        else:
            m = None

        # Pass through bidirectional Mamba2 layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, t_emb, mu, m, use_reentrant=False)
            else:
                h = layer(h, t_emb=t_emb, cond=mu, mask=m)

        h = self.norm_f(h)
        if m is not None:
            h = h * m

        v = self.final_out_proj(h)
        if m is not None:
            v = v * m

        return v

    @torch.no_grad()
    def solve_euler(self, mu, mask=None, n_timesteps=10, temperature=1.0):
        """
        Vectorized Euler ODE solver integrating dx_t/dt = v_theta from t=0 to t=1.
        Initial state x(0) starts at prior mu perturbed by scaled temperature noise (Matcha-TTS style).
        """
        B, T, _ = mu.size()
        device = mu.device
        dtype = mu.dtype

        # Prior-centered initial state x_0 ~ N(mu, temperature^2 * I)
        x = mu + torch.randn_like(mu) * temperature

        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            if m.size(1) > T:
                m = m[:, :T, :]
            elif m.size(1) < T:
                m = F.pad(m, (0, 0, 0, T - m.size(1)), value=0.0)
            x = x * m
        else:
            m = None

        dt = 1.0 / n_timesteps

        for step in range(n_timesteps):
            t_val = step / float(n_timesteps)
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.predict_velocity(x, t, mu, mask=mask)
            x = x + dt * v
            if m is not None:
                x = x * m

        return x

    def forward(self, mu, mask=None, target=None, n_timesteps=10, temperature=1.0):
        B, T, _ = mu.size()

        if target is not None:
            # Training Mode (Prior-Guided OT-CFM)
            if target.size(1) != T:
                min_len = min(T, target.size(1))
                mu = mu[:, :min_len]
                target = target[:, :min_len]
                if mask is not None:
                    mask = mask[:, :min_len]
                T = min_len

            t = torch.rand(B, device=mu.device, dtype=mu.dtype)
            t_expand = t.view(B, 1, 1)

            # Prior-centered source distribution x_0 ~ N(mu, I)
            x_0 = mu + torch.randn_like(target)
            x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x_0 + t_expand * target
            u_target = target - (1.0 - self.sigma_min) * x_0

            v_pred = self.predict_velocity(x_t, t, mu, mask=mask)
            return v_pred, u_target
        else:
            # Inference Mode (ODE Solver starting from acoustic prior mu)
            return self.solve_euler(mu, mask=mask, n_timesteps=n_timesteps, temperature=temperature)