import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2
from x_transformers import Encoder


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
    def __init__(self, d_model=384, d_t=256, d_state=64, d_conv=4, expand=2, headdim=64, kernel_size=17, dropout=0.1):
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

        # Depthwise Conv1D with large kernel for harmonic formant continuity
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


class XTEncoderAdaLNBlock(nn.Module):
    """
    x_transformers Encoder Block with AdaLN-Zero Timestep Conditioning.
    Provides exact global attention using Flash Attention and Rotary Embeddings.
    """
    def __init__(self, dim=256, d_t=256, depth=1, heads=4, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_swish=True,
            ff_mult=4,
            attn_flash=True,
            rotary_pos_emb=True,
            use_rmsnorm=True,
            attn_dropout=dropout,
            ff_dropout=dropout
        )
        self.norm = nn.RMSNorm(dim)
        
        # AdaLN-Zero: predicts scale (gamma), shift (beta), and residual gate (alpha)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t, 3 * dim, bias=True),
        )
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x, t_emb=None, mask=None):
        if t_emb is not None:
            gamma, beta, alpha = self.adaln(t_emb).unsqueeze(1).chunk(3, dim=-1)
            h = (1.0 + gamma) * self.norm(x) + beta
        else:
            alpha = 1.0
            h = self.norm(x)
            
        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            h = h * m
            bool_mask = mask.bool()
            if bool_mask.dim() == 3:
                bool_mask = bool_mask.squeeze(-1)
        else:
            m = None
            bool_mask = None
            
        # The encoder internally computes: out = h + F(h)
        out = self.encoder(h, mask=bool_mask)
        
        # Extract the pure transformer delta: F(h) = out - h
        delta = out - h
        
        # Apply AdaLN-Zero scaling and add to original residual x
        res = x + alpha * delta
        
        if m is not None:
            res = res * m
            
        return res


class ConvNeXtAdaLNBlock(nn.Module):
    """
    ConvNeXt-1D Large-Kernel Local Refinement Block with AdaLN-Zero Timestep Conditioning:
    - Depthwise Conv1D with large kernel (e.g. k=31 or 17) for local temporal & harmonic continuity
    - AdaLN-Zero modulation: predicts scale (gamma), shift (beta), and residual gate (alpha) from t_emb
    - Inverted Bottleneck: Linear (dim -> 4*dim) -> SiLU -> Dropout -> Linear (4*dim -> dim)
    - Zero-initialized AdaLN-Zero projection guarantees smooth identity start at initialization
    """
    def __init__(self, dim=256, d_t=256, kernel_size=17, mlp_ratio=4, dropout=0.0):
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


class Downsample4D(nn.Module):
    """
    Symmetric 1D 4x Downsampling layer (stride 4) using Conv1D:
    Reduces temporal resolution from T to T//4 while expanding receptive field.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=8, stride=4, padding=2)

    def forward(self, x):
        # x: [B, T, C] -> transpose to [B, C, T] -> conv -> transpose back to [B, T//4, C_out]
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class Upsample4D(nn.Module):
    """
    Symmetric 1D 4x Direct Upsampling layer (stride 4) using ConvTranspose1D:
    Expands temporal resolution directly from T//4 to T with exact target length matching.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=8, stride=4, padding=2)

    def forward(self, x, target_len=None):
        # x: [B, T//4, C] -> transpose to [B, C, T//4] -> upsample -> transpose back to [B, T, C_out]
        h = self.conv_transpose(x.transpose(1, 2)).transpose(1, 2)
        if target_len is not None and h.size(1) != target_len:
            if h.size(1) > target_len:
                h = h[:, :target_len, :]
            else:
                h = F.pad(h, (0, 0, 0, target_len - h.size(1)), value=0.0)
        return h


class MambaFlowStaircasePyramidDecoder(nn.Module):
    """
    MambaFlow ResNet-Style Staircase Feature Pyramid Decoder:

    Architecture Overview:
    1. Input at full resolution T: Concatenates [x_t, mu] and projects to d_base (192).
       Initial local temporal refinement via ConvNeXt-AdaLN (dim=192, k=31).
    2. Level 1: Coarsest Temporal Scale (T/4, 256-dim)
       - Direct Downsample: T -> T/4 (Downsample4D: 192 -> 256)
       - Global Prosody Bottleneck: Bidirectional Mamba-2 SSM (k=17) + ConvNeXt-AdaLN (k=17)
       - Upsample: T/4 -> T/2 (Upsample1D: 256 -> 256) -> P_1
    3. Level 2: Intermediate Temporal Scale (T/2, 256-dim)
       - Direct Downsample: T -> T/2 (Downsample1D: 192 -> 256)
       - Staircase Residual Addition: H_L2_in + P_1
       - Temporal Smoothing with AdaLN: ConvNeXt-AdaLN (dim=256, k=17)
       - Syllabic/Formant Bottleneck: Bidirectional Mamba-2 SSM (k=17) + ConvNeXt-AdaLN (k=17)
       - Upsample: T/2 -> T (Upsample1D: 256 -> 192) -> P_2
    4. Level 3: Full-Resolution Fine Detail Scale (T, 192-dim)
       - Staircase Residual Addition: H_T + P_2
       - Temporal Smoothing with AdaLN: ConvNeXt-AdaLN (dim=192, k=31)
       - Harmonic Formant Refinement (Pure ConvNeXt, eliminates SSM low-pass blur):
         * ConvNeXt-AdaLN (dim=192, k=31)
         * ConvNeXt-AdaLN (dim=192, k=17)
    5. Output: RMSNorm -> Zero-initialized Linear projection to 100-dim mel velocity v_theta.
    6. Anti-Dual Audio / Anti-Phase-Distortion Guarantee:
       Multiple-of-4 symmetric padding at entry + exact target_len matching at every upsampling layer
       + clean exit unpadding ensures 100% exact frame alignment.
    """
    def __init__(
        self,
        d_in=100,
        d_cond=100,
        d_out=100,
        d_base=192,
        d_mid=256,
        d_t=192,
        kernel_size_coarse=17,
        kernel_size_fine=31,
        dropout=0.1,
        sigma_min=1e-4,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__()
        # Backward compatibility for legacy dimension arguments if passed
        d_base = kwargs.get("d_model_1", kwargs.get("d_model", d_base))
        d_mid = kwargs.get("d_model_2", kwargs.get("d_model_mid", d_mid))
        d_t = kwargs.get("d_t", d_t)

        self.d_in = d_in
        self.d_cond = d_cond
        self.d_out = d_out
        self.d_base = d_base
        self.d_mid = d_mid
        self.d_t = d_t
        self.sigma_min = sigma_min
        self.gradient_checkpointing = gradient_checkpointing

        # Timestep embedding network (scalar t in [0, 1] -> d_t vector)
        self.t_embed = TimestepEmbedding(d_t=d_t)

        # Level 3 Base (Resolution 1x, dim 192): Input projection & initial ConvNeXt refinement
        self.in_proj = nn.Linear(d_in + d_cond, d_base, bias=False)
        self.in_conv = ConvNeXtAdaLNBlock(dim=d_base, d_t=d_t, kernel_size=kernel_size_fine, dropout=dropout)

        # =====================================================================
        # LEVEL 1: Coarsest Scale (T -> T/4 -> T/2)
        # =====================================================================
        self.down_to_l1 = Downsample4D(d_base, d_mid)
        self.l1_xt = XTEncoderAdaLNBlock(
            dim=d_mid,
            d_t=d_t,
            depth=1,
            heads=max(1, d_mid // 64),
            dropout=dropout,
        )
        self.l1_conv = ConvNeXtAdaLNBlock(dim=d_mid, d_t=d_t, kernel_size=kernel_size_coarse, dropout=dropout)
        self.l1_refine = nn.Linear(d_mid, d_mid)
        self.l1_up = Upsample1D(d_mid, d_mid)  # T/4 -> T/2 (256 -> 256)

        # =====================================================================
        # LEVEL 2: Intermediate Scale (T -> T/2 -> T) with Early Staircase Fusion
        # =====================================================================
        self.down_to_l2 = Downsample1D(d_base, d_mid)
        self.l2_smooth = ConvNeXtAdaLNBlock(dim=d_mid, d_t=d_t, kernel_size=kernel_size_coarse, dropout=dropout)
        self.l2_xt = XTEncoderAdaLNBlock(
            dim=d_mid,
            d_t=d_t,
            depth=1,
            heads=max(1, d_mid // 64),
            dropout=dropout,
        )
        self.l2_conv = ConvNeXtAdaLNBlock(dim=d_mid, d_t=d_t, kernel_size=kernel_size_coarse, dropout=dropout)
        self.l2_refine = nn.Linear(d_mid, d_mid)
        self.l2_up = Upsample1D(d_mid, d_base)  # T/2 -> T (256 -> 192)

        # =====================================================================
        # LEVEL 3: Full-Resolution Scale (T) with Early Staircase Fusion
        # =====================================================================
        self.l3_smooth = ConvNeXtAdaLNBlock(dim=d_base, d_t=d_t, kernel_size=kernel_size_fine, dropout=dropout)
        self.l3_xt = XTEncoderAdaLNBlock(
            dim=d_base,
            d_t=d_t,
            depth=1,
            heads=max(1, d_base // 64),
            dropout=dropout,
        )
        self.l3_conv1 = ConvNeXtAdaLNBlock(dim=d_base, d_t=d_t, kernel_size=kernel_size_fine, dropout=dropout)
        self.l3_conv2 = ConvNeXtAdaLNBlock(dim=d_base, d_t=d_t, kernel_size=kernel_size_coarse, dropout=dropout)
        self.l3_refine = nn.Linear(d_base, d_base)

        # Output Projection
        self.final_norm = nn.RMSNorm(d_base)
        self.final_out_proj = nn.Linear(d_base, d_out, bias=False)
        nn.init.zeros_(self.final_out_proj.weight)

    def _run_block(self, block, h, t_emb, mask):
        if self.training and self.gradient_checkpointing:
            return checkpoint(block, h, t_emb, mask, use_reentrant=False)
        return block(h, t_emb=t_emb, mask=mask)

    def predict_velocity(self, x, t, mu, mask=None):
        B, T, _ = x.size()

        # Symmetric Multiple-of-4 Padding:
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

        T_pad = x_in_pad.size(1)
        target_len_l2 = T_pad // 2
        target_len_t = T_pad

        # Timestep embedding [B, d_t]
        t_emb = self.t_embed(t)

        # Initial projection & refinement at full resolution T
        cat_in = torch.cat([x_in_pad, mu_pad], dim=-1)
        h_base = self.in_proj(cat_in)
        if mask_pad is not None:
            m_pad = mask_pad.unsqueeze(-1).float() if mask_pad.dim() == 2 else mask_pad.float()
            h_base = h_base * m_pad
        else:
            m_pad = None

        h_t = self._run_block(self.in_conv, h_base, t_emb, mask_pad)

        # Downsampled masks
        mask_l2 = mask_pad[:, ::2] if mask_pad is not None else None
        m_l2 = mask_l2.unsqueeze(-1).float() if mask_l2 is not None else None

        mask_l1 = mask_pad[:, ::4] if mask_pad is not None else None
        m_l1 = mask_l1.unsqueeze(-1).float() if mask_l1 is not None else None

        # =====================================================================
        # LEVEL 1: Coarsest Scale (T -> T/4 -> T/2)
        # =====================================================================
        h_l1_in = self.down_to_l1(h_t)
        if m_l1 is not None:
            h_l1_in = h_l1_in * m_l1

        h_l1 = self._run_block(self.l1_xt, h_l1_in, t_emb, mask_l1)
        h_l1 = self._run_block(self.l1_conv, h_l1, t_emb, mask_l1)
        
        # Dedicated Layer 1 Residual Pathway + Refinement
        h_l1 = h_l1 + h_l1_in
        h_l1 = self.l1_refine(h_l1)

        p1 = self.l1_up(h_l1, target_len=target_len_l2)
        if m_l2 is not None:
            p1 = p1 * m_l2

        # =====================================================================
        # LEVEL 2: Intermediate Scale (T -> T/2 -> T) with Early Staircase Fusion
        # =====================================================================
        h_l2_in = self.down_to_l2(h_t)
        if m_l2 is not None:
            h_l2_in = h_l2_in * m_l2

        # Early Staircase Fusion: inject Level 1 coarse prosody before Level 2 SSM
        h_l2_fused = h_l2_in + p1
        h_l2_fused = self._run_block(self.l2_smooth, h_l2_fused, t_emb, mask_l2)

        h_l2 = self._run_block(self.l2_xt, h_l2_fused, t_emb, mask_l2)
        h_l2 = self._run_block(self.l2_conv, h_l2, t_emb, mask_l2)
        
        # Dedicated Layer 2 Residual Pathway + Refinement
        h_l2 = h_l2 + h_l2_fused
        h_l2 = self.l2_refine(h_l2)

        p2 = self.l2_up(h_l2, target_len=target_len_t)
        if m_pad is not None:
            p2 = p2 * m_pad

        # =====================================================================
        # LEVEL 3: Full-Resolution Scale (T) with Early Staircase Fusion
        # =====================================================================
        # Early Staircase Fusion: inject Level 2 mid-scale features before Level 3 blocks
        h_l3_fused = h_t + p2
        h_l3_in = self._run_block(self.l3_smooth, h_l3_fused, t_emb, mask_pad)

        # Process through XTEncoder and ConvNeXt blocks
        h_l3 = self._run_block(self.l3_xt, h_l3_in, t_emb, mask_pad)
        h_l3 = self._run_block(self.l3_conv1, h_l3, t_emb, mask_pad)
        h_l3 = self._run_block(self.l3_conv2, h_l3, t_emb, mask_pad)
        
        # Dedicated Layer 3 Residual Pathway + Refinement
        h_l3 = h_l3 + h_l3_in
        h_l3 = self.l3_refine(h_l3)

        # =====================================================================
        # Output Projection & Exact Length Recovery
        # =====================================================================
        h_out = self.final_norm(h_l3)
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

        dt = float(t_end) / float(n_timesteps)

        for step in range(n_timesteps):
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)

            # Predict velocity at current point
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

        # Returns the predicted full mel spectrogram directly
        return x

    def forward(self, mu, mask=None, target=None, n_timesteps=30, temperature=1.0, solver="euler", t_end=1.0):
        B, T, _ = mu.size()

        if target is not None:
            # Training Mode
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
            # Inference Mode
            return self.solve_euler(
                mu=mu,
                mask=mask,
                n_timesteps=n_timesteps,
                temperature=temperature,
                solver=solver,
                t_end=t_end,
            )


# Export aliases for full backward and module compatibility
MambaFlowStaircasePyramidDecoder = MambaFlowStaircasePyramidDecoder
MambaFlowTwoStageDecoder = MambaFlowStaircasePyramidDecoder
MambaTwoStageDecoder = MambaFlowStaircasePyramidDecoder
MambaFlowUNetDecoder = MambaFlowStaircasePyramidDecoder
MambaUNetDecoder = MambaFlowStaircasePyramidDecoder
decoder = MambaFlowStaircasePyramidDecoder
