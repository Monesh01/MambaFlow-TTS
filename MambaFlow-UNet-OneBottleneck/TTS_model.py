import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from trans_encoder import BiMambaConvNeXtEncoder
from duration_predictor import duration
from mambaflow_decoder import MambaFlowUNetDecoder, decoder as Decoder


class MambaFlowTTSModel(nn.Module):
    """
    MambaFlow-TTS Architecture with Bidirectional Mamba2 + ConvNeXt Text Encoder
    and MambaFlow-UNet (Multi-Scale Temporal Pyramid with Bi-Mamba2 SSMs):
    - Text -> 6-Layer Bi-Mamba2 + ConvNeXt Encoder (192-dim, kernels=[9, 17, 31])
    - to_latent Projection: 192 -> 100-dim Base Acoustic Mel Prior (mu_text_base)
    - Duration Predictor over Detached 100-dim Base Acoustic Prior (kernels=[9, 17, 31])
    - MAS Alignment on 100-dim Normalized Mel Latent
    - Repeat-Interleaved 100-dim Base Mel Latent -> mu_expanded (Acoustic Conditioning)
    - Flow Matching Decoder (MambaFlow-UNet):
        * Multi-Scale Temporal Pyramid (1x -> 1/2x -> 1/4x -> 1/2x -> 1x)
        * Full Mel Velocity Target (u_t = target - (1 - sigma_min)*x_0)
        * Zero phase distortion (anti-dual audio multiple-of-4 symmetric padding)
    """
    def __init__(
        self,
        encoder=None,
        duration_module=None,
        decoder_module=None,
        n_vocab=179,
        d_text=None,
        d_enc=192,
        d_codec=100,
        d_dec=192,
        emb_dropout=0.0,
        use_residual_target=False,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.d_enc = d_enc
        self.d_codec = d_codec
        self.d_text = d_enc if d_text is None else d_text
        self.use_residual_target = use_residual_target

        self.txt_emb = nn.Embedding(n_vocab, self.d_text)
        self.emb_dropout = nn.Dropout(emb_dropout)

        # 6-Layer Bidirectional Mamba2 + ConvNeXt Text Encoder (192-dim)
        self.encoder = encoder if encoder is not None else BiMambaConvNeXtEncoder(
            dim=d_enc,
            depth=6,
            kernel_sizes=(9, 17, 31, 9, 17, 31),
            dropout=0.0,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Linear projection from text encoder (192) to 100-dim Base Acoustic Prior (supervised via L1 loss)
        self.to_latent = nn.Linear(d_enc, d_codec, bias=False)

        # Duration Predictor with MAS alignment on 100-dim target (input: detached 100-dim mu_text_base)
        self.duration = duration_module if duration_module is not None else duration(
            dims=d_codec,
            hidden_dim=256,
            target_dim=d_codec,
            kernel_sizes=(9, 17, 31),
            dropout_p=0.0,
        )

        # MambaFlow UNet Mamba 2 Decoder (One Bottleneck)
        self.decoder = decoder_module if decoder_module is not None else Decoder(
            d_in=d_codec,
            d_cond=d_codec,
            d_out=d_codec,
            d_model_1=d_dec,
            d_model_2=256,
            d_model_mid=256,
            d_t=192,
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
        n_timesteps=30,
        temperature=1.0,
        length_scale=1.0,
        solver="euler",
        t_end=1.0,
    ):
        alignment = None
        dur_pred = None
        mas_score = None

        target = target_latent

        # Text embedding with dropout
        x_in = self.emb_dropout(self.txt_emb(x))

        # 1. 6-Layer Bidirectional Mamba2 + ConvNeXt Text Encoder (192-dim)
        enc_out = self.encoder(x_in, mask=mask)

        # 2. Base Acoustic Prior (100-dim) for MAS alignment search
        mu_text_base = self.to_latent(enc_out)
        if mask is not None:
            m_t = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            mu_text_base = mu_text_base * m_t

        # 3. Duration & Alignment over Detached 100-dim Base Acoustic Prior
        if target is not None:
            # Training Mode: Monotonic Alignment Search
            alignment, dur_pred, mas_score = self.duration(
                mu_text_base,
                latent=target,
                mask=mask,
                audio_mask=audio_mask,
                mas_prior=mu_text_base,
            )
            durations = alignment.sum(dim=-1).long()  # [B, T_text]
        else:
            # Inference Mode: Predicted durations with length scaling
            dur_int, dur_pred, _ = self.duration(
                mu_text_base,
                latent=None,
                mask=mask,
                length_scale=length_scale,
            )
            durations = dur_int
            alignment = None

        # Clamp durations to ensure valid frame expansions
        durations = torch.clamp(durations, min=0)
        for b in range(durations.size(0)):
            if durations[b].sum() == 0:
                durations[b, 0] = 1

        # 4. Duration Expansion of 100-dim Base Acoustic Prior (mu_text_base)
        latent_expanded_list = [
            torch.repeat_interleave(mu_text_base[b], durations[b], dim=0)
            for b in range(mu_text_base.size(0))
        ]
        latent_expanded = pad_sequence(latent_expanded_list, batch_first=True)  # [B, T_audio, 100]

        audio_lengths = durations.sum(dim=-1)
        max_len = latent_expanded.size(1)
        derived_audio_mask = torch.arange(max_len, device=x.device)[None, :] < audio_lengths[:, None]
        if audio_mask is None:
            audio_mask = derived_audio_mask

        # Direct 100-dim Acoustic Prior (Frame Refinement removed)
        mu_expanded = latent_expanded
        if audio_mask is not None:
            m_c = audio_mask.unsqueeze(-1).float() if audio_mask.dim() == 2 else audio_mask.float()
            if m_c.size(1) > mu_expanded.size(1):
                m_c = m_c[:, :mu_expanded.size(1), :]
            elif m_c.size(1) < mu_expanded.size(1):
                mu_expanded = mu_expanded[:, :m_c.size(1), :]
            mu_expanded = mu_expanded * m_c

        # 5. Decoder: MambaFlow-UNet / Residual Flow Matching
        if self.decoder is not None:
            if target is not None:
                # Align lengths between target and mu_expanded
                min_T = min(target.size(1), mu_expanded.size(1))
                target = target[:, :min_T, :]
                mu_expanded = mu_expanded[:, :min_T, :]
                if audio_mask is not None:
                    audio_mask = audio_mask[:, :min_T]

                if self.use_residual_target:
                    # Residual target R = target - mu_expanded.detach()
                    flow_target = target - mu_expanded.detach()
                else:
                    # Direct Full Mel target (eliminates additive residual staircase leakage)
                    flow_target = target

                v_pred, u_target = self.decoder(
                    mu=mu_expanded,
                    mask=audio_mask,
                    target=flow_target,
                )
                return (v_pred, u_target), mu_expanded, alignment, dur_pred, mas_score
            else:
                # Inference Mode: Decoder integrates from noise
                dec_out = self.decoder(
                    mu=mu_expanded,
                    mask=audio_mask,
                    target=None,
                    n_timesteps=n_timesteps,
                    temperature=temperature,
                    solver=solver,
                    t_end=t_end,
                )
                if self.use_residual_target:
                    mel_pred = mu_expanded + dec_out
                else:
                    # Final Mel is generated directly by the continuous ODE integration
                    mel_pred = dec_out
                return mel_pred, mu_expanded

        if target is not None:
            return None, mu_expanded, alignment, dur_pred, mas_score
        else:
            return mu_expanded, mu_expanded


# Backward compatibility alias
Model = MambaFlowTTSModel
