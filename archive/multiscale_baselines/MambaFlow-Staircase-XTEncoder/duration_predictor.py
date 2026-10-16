import torch
import torch.nn as nn
from preprocessing.monotonic_align import maximum_path as default_mas


class ConvNeXt1DDurationBlock(nn.Module):
    """
    ConvNeXt-1D Duration Block with large kernel (e.g. k=9, 17, 31):
    - Depthwise Conv1D (groups=dim) with large kernel: huge receptive field with tiny parameter count
    - LayerNorm on channel dimension
    - Inverted Bottleneck: Linear (dim -> 2*dim) -> SiLU -> Dropout -> Linear (2*dim -> dim)
    - Residual Connection
    """
    def __init__(self, dim=256, kernel_size=17, mlp_ratio=2, dropout_p=0.0):
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
        self.norm = nn.LayerNorm(dim, bias=False)
        self.pw1 = nn.Linear(dim, mlp_ratio * dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_p)
        self.pw2 = nn.Linear(mlp_ratio * dim, dim, bias=False)

    def forward(self, x, mask=None):
        # x: [B, T, dim]
        res = x

        x_t = x.transpose(1, 2)
        if mask is not None:
            m = mask.unsqueeze(1).float() if mask.dim() == 2 else mask.float()
            x_t = x_t * m
            m_c = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
        else:
            m = None
            m_c = None

        # 1. Depthwise Convolution with Large Kernel
        h = self.dwconv(x_t).transpose(1, 2)
        if m_c is not None:
            h = h * m_c

        # 2. LayerNorm + Inverted Bottleneck MLP
        h = self.norm(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.pw2(h)

        out = res + h
        if m_c is not None:
            out = out * m_c
        return out


class duration(nn.Module):
    """
    ConvNeXt-1D Large-Kernel Duration Predictor & MAS Alignment Module:
    - Predictor: Multi-stage ConvNeXt-1D stack with large kernels (k=9, 17, 31) over detached 100-d acoustic prior features.
    - Large receptive field (>35 phonemes) with depthwise convolutions (lightweight, zero overfitting, no dilation needed).
    - Alignment: Direct pairwise negative Euclidean distance between 100-dim acoustic prior (mu_text) and 100-dim normalized target.
    """
    def __init__(
        self,
        dims=100,
        hidden_dim=256,
        target_dim=100,
        kernel_sizes=(9, 17, 31),
        mlp_ratio=2,
        dropout_p=0.0,
        mas=None,
    ):
        super().__init__()
        self.dims = dims
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        # Input projection
        self.in_proj = nn.Linear(dims, hidden_dim, bias=False)
        self.in_norm = nn.LayerNorm(hidden_dim, bias=False)
        self.in_act = nn.SiLU()

        # Multi-scale ConvNeXt-1D layers (k=9, 17, 31)
        self.conv_blocks = nn.ModuleList([
            ConvNeXt1DDurationBlock(
                dim=hidden_dim,
                kernel_size=k,
                mlp_ratio=mlp_ratio,
                dropout_p=dropout_p,
            )
            for k in kernel_sizes
        ])

        # Output linear projection -> [B, T_text, 1] -> squeeze(-1) -> [B, T_text]
        self.out_norm = nn.LayerNorm(hidden_dim, bias=False)
        self.linear = nn.Linear(hidden_dim, 1, bias=False)

        # Monotonic Alignment Search
        self.mas = mas if mas is not None else default_mas

    def forward(self, mu_text, latent=None, mask=None, audio_mask=None, length_scale=1.2, mas_prior=None):
        """
        Args:
            mu_text:      [B, T_text, dims] - 100-dim Base Acoustic Prior (or text representation)
            latent:       [B, T_audio, 100] - Ground-truth target mel (training only)
            mask:         [B, T_text]       - Text token boolean/float mask
            audio_mask:   [B, T_audio]      - Audio frame boolean/float mask
            length_scale: float             - Duration scaling factor during inference
            mas_prior:    [B, T_text, 100]  - Explicit detached 100-dim acoustic prior for MAS distance
        """
        # Always detach acoustic prior to prevent duration gradients from pulling acoustic latents
        x_det = mu_text.detach()
        if mas_prior is None:
            mas_prior = x_det
        else:
            mas_prior = mas_prior.detach()

        # Input projection
        h = self.in_proj(x_det)
        h = self.in_norm(h)
        h = self.in_act(h)

        if mask is not None:
            m_c = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            h = h * m_c

        # ConvNeXt-1D large-kernel stack
        for block in self.conv_blocks:
            h = block(h, mask=mask)

        h = self.out_norm(h)
        duration_pred = self.linear(h).squeeze(-1)

        if mask is not None:
            m_b = mask.bool() if mask.dim() == 2 else mask.squeeze(-1).bool()
            duration_pred = duration_pred.masked_fill(~m_b, 0.0)

        # ==================================================
        # Training Mode (MAS Alignment between 100-dim to_latent output and Target Mel)
        # ==================================================
        if latent is not None:
            # Strictly use the detached 100-dim acoustic prior for MAS alignment
            prior = mas_prior

            # Negative squared Euclidean distance: -0.5 * ||prior (100-dim) - target (100-dim)||^2
            prior_sq = (prior ** 2).sum(dim=-1, keepdim=True)        # [B, T_text, 1]
            latent_sq = (latent ** 2).sum(dim=-1).unsqueeze(1)       # [B, 1, T_latent]
            cross = torch.bmm(prior, latent.transpose(1, 2))         # [B, T_text, T_latent]
            score = -0.5 * (prior_sq - 2.0 * cross + latent_sq).float()

            if mask is None:
                text_mask = torch.ones(mu_text.size(0), mu_text.size(1), 1, dtype=torch.bool, device=mu_text.device)
            elif mask.dim() == 2:
                text_mask = mask.unsqueeze(-1).bool()
            else:
                text_mask = mask.bool()

            if audio_mask is not None:
                latent_mask = audio_mask.unsqueeze(1).bool() if audio_mask.dim() == 2 else audio_mask.bool()
            else:
                latent_mask = torch.ones(
                    latent.size(0),
                    1,
                    latent.size(1),
                    dtype=torch.bool,
                    device=latent.device
                )

            mas_mask = text_mask & latent_mask

            alignment = self.mas(score, mas_mask)
            # Return raw 3D alignment matrix instead of aligned_target, TTS_model expects this for bmm
            return alignment, duration_pred, score

        # ==================================================
        # Inference Mode
        # ==================================================
        else:
            # Scale durations by length_scale (default 1.2 slows down speech pace for cleaner articulation)
            dur_int = torch.clamp((torch.exp(duration_pred) - 1.0) * float(length_scale), min=0.0)
            dur_int = torch.round(dur_int)
            dur_int = torch.clamp(dur_int, min=1).long()
            if mask is not None:
                m_b = mask.bool() if mask.dim() == 2 else mask.squeeze(-1).bool()
                dur_int = dur_int.masked_fill(~m_b, 0)

            return dur_int, duration_pred, None