import torch
import torch.nn as nn
from preprocessing.monotonic_align import maximum_path as default_mas



class duration(nn.Module):
    """
    Duration Predictor & MAS Alignment Module:
    - Predictor: Conv1D (k=3) network with LayerNorm and SiLU over detached 384-d encoder features.
    - Alignment: Direct pairwise negative Euclidean distance between 100-dim acoustic prior (mu_text) and 100-dim normalized target.
    """
    def __init__(self, dims=384, hidden_dim=256, target_dim=100, mas=None):
        super().__init__()
        self.dims = dims
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        # Duration predictor Conv1D stack
        self.conv1 = nn.Conv1d(dims, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim, bias=False)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.LayerNorm(hidden_dim, bias=False)
        self.act2 = nn.SiLU()

        self.linear = nn.Linear(hidden_dim, 1, bias=False)

        # Monotonic Alignment Search
        self.mas = mas if mas is not None else default_mas

    def forward(self, x, mu_text=None, latent=None, mask=None, audio_mask=None, length_scale=1.2):
        # Always detach encoder representations for duration predictor to prevent duration gradients from distorting phonetic embeddings
        x_det = x.detach()

        # Hidden Layer 1
        h = x_det.transpose(1, 2)
        if mask is not None:
            m = mask.unsqueeze(1).float() if mask.dim() == 2 else mask.float()
            h = h * m
        h = self.conv1(h)
        if mask is not None:
            h = h * m
        h = self.norm1(h.transpose(1, 2))
        h = self.act1(h)

        # Hidden Layer 2
        h = h.transpose(1, 2)
        if mask is not None:
            m = mask.unsqueeze(1).float() if mask.dim() == 2 else mask.float()
            h = h * m
        h = self.conv2(h)
        if mask is not None:
            h = h * m
        h = self.norm2(h.transpose(1, 2))
        h = self.act2(h)

        if mask is not None:
            m_c = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            h = h * m_c

        # Output linear projection -> [B, T_text, 1] -> squeeze(-1) -> [B, T_text]
        duration_pred = self.linear(h).squeeze(-1)
        if mask is not None:
            m_b = mask.bool() if mask.dim() == 2 else mask.squeeze(-1).bool()
            duration_pred = duration_pred.masked_fill(~m_b, 0.0)

        # ==================================================
        # Training Mode (MAS Alignment with Target Codec Latent)
        # ==================================================
        if latent is not None:
            # Use 72-dim acoustic prior mu_text for alignment
            prior = mu_text.detach() if mu_text is not None else x_det

            # Negative squared Euclidean distance: -0.5 * ||prior - latent||^2
            prior_sq = (prior ** 2).sum(dim=-1, keepdim=True)        # [B, T_text, 1]
            latent_sq = (latent ** 2).sum(dim=-1).unsqueeze(1)       # [B, 1, T_latent]
            cross = torch.bmm(prior, latent.transpose(1, 2))         # [B, T_text, T_latent]
            score = -0.5 * (prior_sq - 2.0 * cross + latent_sq).float()

            if mask is None:
                text_mask = torch.ones(x.size(0), x.size(1), 1, dtype=torch.bool, device=x.device)
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
            # Scale durations by length_scale (default 1.2 slows down speech pace by 20% for cleaner articulation)
            dur_int = torch.clamp((torch.exp(duration_pred) - 1.0) * float(length_scale), min=0.0)
            dur_int = torch.round(dur_int)
            dur_int = torch.clamp(dur_int, min=1).long()
            if mask is not None:
                m_b = mask.bool() if mask.dim() == 2 else mask.squeeze(-1).bool()
                dur_int = dur_int.masked_fill(~m_b, 0)

            return dur_int, duration_pred, None