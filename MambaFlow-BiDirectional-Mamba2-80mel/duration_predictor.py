import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    """
    3-Layer High-Capacity ResNet-Conv1D Duration Predictor with LayerNorm and Residuals:
    - Layer 1: Conv1D (k=5, padding=2) -> LayerNorm -> SiLU -> Dropout(0.1)
    - Layer 2: Conv1D (k=5, padding=2) -> LayerNorm -> SiLU -> Dropout(0.1) + Residual
    - Layer 3: Conv1D (k=5, padding=2) -> LayerNorm -> SiLU -> Dropout(0.1) + Residual
    - Output Head: Linear(hidden_dim, 1) predicting log(duration)
    - Receptive Field = 13 phonemes (covers phrase boundaries, word stress, commas, and pauses)
    - Pure Duration Regression Head: MAS is completely eliminated.
    """
    def __init__(
        self,
        dims=192,
        hidden_dim=256,
        target_dim=100,
        kernel_size=5,
        dropout_p=0.1,
        **kwargs,
    ):
        super().__init__()
        self.dims = dims
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.dropout_p = dropout_p

        # Layer 1: 1D Conv -> LayerNorm -> SiLU -> Dropout
        self.conv1 = nn.Conv1d(
            dims,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, bias=False)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_p)

        # Layer 2: 1D Conv -> LayerNorm -> SiLU -> Dropout
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, bias=False)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_p)

        # Layer 3: 1D Conv -> LayerNorm -> SiLU -> Dropout
        self.conv3 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.norm3 = nn.LayerNorm(hidden_dim, bias=False)
        self.act3 = nn.SiLU()
        self.drop3 = nn.Dropout(dropout_p)

        # Output linear projection -> [B, T_text, 1] -> squeeze(-1)
        self.linear = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, mu_text, latent=None, mask=None, audio_mask=None, length_scale=1.0, mas_prior=None):
        """
        Pure Duration Regression forward pass.
        Args:
            mu_text:      [B, T_text, dims] - Feature representation (detached enc_out)
            mask:         [B, T_text]       - Text token boolean/float mask
            length_scale: float             - Duration scaling factor during inference
        Returns:
            dur_int:       [B, T_text]      - Clamped integer frame counts for inference
            duration_pred: [B, T_text]      - Continuous log(duration) predictions for Huber loss
            None:          Backward-compatibility placeholder for unused MAS score
        """
        x_det = mu_text.detach()
        m_c = mask.unsqueeze(-1).float() if (mask is not None and mask.dim() == 2) else (mask.float() if mask is not None else None)

        # Layer 1: dims -> hidden_dim
        h = x_det
        if m_c is not None:
            h = h * m_c
        h = self.conv1(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.drop1(h)
        if m_c is not None:
            h = h * m_c

        # Layer 2: hidden_dim -> hidden_dim with residual connection
        res2 = h
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.drop2(h)
        h = h + res2
        if m_c is not None:
            h = h * m_c

        # Layer 3: hidden_dim -> hidden_dim with residual connection
        res3 = h
        h = self.conv3(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm3(h)
        h = self.act3(h)
        h = self.drop3(h)
        h = h + res3
        if m_c is not None:
            h = h * m_c

        # Output projection -> log duration
        duration_pred = self.linear(h).squeeze(-1)

        if mask is not None:
            m_b = mask.bool() if mask.dim() == 2 else mask.squeeze(-1).bool()
            duration_pred = duration_pred.masked_fill(~m_b, 0.0)

        # Convert log1p(w) to integer frame counts for inference expansion
        dur_int = torch.clamp(torch.round(torch.expm1(duration_pred.clamp(min=0.0)) * float(length_scale)), min=0.0).long()
        if mask is not None:
            dur_int = dur_int.masked_fill(~m_b, 0)

        return dur_int, duration_pred, None


# Backward compatibility aliases
duration = DurationPredictor
DurationPredictor3Layer = DurationPredictor
DurationPredictor2Layer = DurationPredictor