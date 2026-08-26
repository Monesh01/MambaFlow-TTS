import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from trans_encoder import XTEncoder
from duration_predictor import duration
from ssm_decoder import decoder


class MambaFlowTTSModel(nn.Module):
    """
    MambaFlow-TTS Architecture (Single Speaker Mel-Spectrogram):
    - Text -> 6-Layer Transformer Encoder -> Detached state to Duration Predictor
    - MAS Alignment on 100-dim Normalized Mel Latent
    - Repeat-Interleaved 100-dim Latent Representation
    - 6-Layer True Bidirectional Mamba2 Decoder with CFM Flow Matching
    """
    def __init__(
        self,
        encoder=None,
        duration_module=None,
        decoder_module=None,
        n_vocab=179,
        d_text=256,
        d_enc=256,
        d_codec=100,
        d_dec=384,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.d_enc = d_enc
        self.d_codec = d_codec

        self.txt_emb = nn.Embedding(n_vocab, d_text)

        # 6-Layer Transformer Encoder
        self.encoder = encoder if encoder is not None else XTEncoder(
            dim=d_enc,
            depth=6,
            heads=6,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Duration Predictor with MAS alignment on 100-dim target
        self.duration = duration_module if duration_module is not None else duration(
            dims=d_enc,
            hidden_dim=256,
            target_dim=d_codec,
        )

        # Linear projection from encoder dimension (256) to 100-dim acoustic latent prior space
        self.to_latent = nn.Linear(d_enc, d_codec, bias=False)

        # 6-Layer Bidirectional Mamba2 Decoder
        self.decoder = decoder_module if decoder_module is not None else decoder(
            d_in=d_codec,
            d_cond=d_codec,
            d_out=d_codec,
            d_model=d_dec,
            n_layers=6,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.gradient_checkpointing = gradient_checkpointing

    @property
    def gradient_checkpointing(self):
        return getattr(self, "_gradient_checkpointing", False)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._gradient_checkpointing = bool(value)
        if hasattr(self.encoder, "gradient_checkpointing"):
            self.encoder.gradient_checkpointing = bool(value)
        if hasattr(self.decoder, "gradient_checkpointing"):
            self.decoder.gradient_checkpointing = bool(value)

    def forward(
        self,
        x,
        target_latent=None,
        mask=None,
        audio_mask=None,
        n_timesteps=10,
        temperature=1.0,
        length_scale=1.2,
    ):
        out = None
        alignment = None
        latent = None
        dur_pred = None
        mas_score = None

        target = target_latent

        # Text embedding
        x_in = self.txt_emb(x)

        # Pass through 6-Layer Transformer Encoder
        enc_out = self.encoder(x_in, mask=mask)

        # Base acoustic prior
        mu_text_base = self.to_latent(enc_out)
        if mask is not None:
            m_t = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            mu_text_base = mu_text_base * m_t

        # Duration & Alignment
        if target is not None:
            # Training Mode: Monotonic Alignment Search
            alignment, dur_pred, mas_score = self.duration(
                enc_out.detach(),
                mu_text=mu_text_base,
                latent=target,
                mask=mask,
                audio_mask=audio_mask,
            )
            durations = alignment.sum(dim=-1).long()  # [B, T_text]
        else:
            # Inference Mode: Predicted durations with length scaling (default 1.2)
            dur_int, dur_pred, _ = self.duration(
                enc_out.detach(),
                mask=mask,
                length_scale=length_scale,
            )
            durations = dur_int
            alignment = None

        # Clamp durations to ensure valid frame expansions
        durations = torch.clamp(durations, min=0)
        # Avoid empty sequence by ensuring at least 1 token has duration >= 1 per batch item
        for b in range(durations.size(0)):
            if durations[b].sum() == 0:
                durations[b, 0] = 1

        # Repeat-interleave 100-dim acoustic prior representations
        latent_expanded_list = [
            torch.repeat_interleave(mu_text_base[b], durations[b], dim=0)
            for b in range(mu_text_base.size(0))
        ]
        latent = pad_sequence(latent_expanded_list, batch_first=True)

        audio_lengths = durations.sum(dim=-1)
        max_len = latent.size(1)
        derived_audio_mask = torch.arange(max_len, device=x.device)[None, :] < audio_lengths[:, None]
        if audio_mask is None:
            audio_mask = derived_audio_mask

        if audio_mask is not None:
            m_c = audio_mask.unsqueeze(-1).float() if audio_mask.dim() == 2 else audio_mask.float()
            if m_c.size(1) > latent.size(1):
                m_c = m_c[:, :latent.size(1), :]
            elif m_c.size(1) < latent.size(1):
                latent = latent[:, :m_c.size(1), :]
            latent = latent * m_c

        # Decoder: Prior-Guided CFM Flow Matching
        if self.decoder is not None:
            out = self.decoder(
                mu=latent,
                mask=audio_mask,
                target=target,
                n_timesteps=n_timesteps,
                temperature=temperature,
            )

        if target is not None:
            return out, latent, alignment, dur_pred, mas_score
        else:
            return out, latent


# Backward compatibility alias
Model = MambaFlowTTSModel
