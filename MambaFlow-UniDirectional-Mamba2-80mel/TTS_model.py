import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from trans_encoder import XTEncoder
from duration_predictor import duration
from mambaflow_tetra_decoder import MambaFlowTetraDecoder


class MambaFlowTTSModel(nn.Module):
    """
    MambaFlow-TTS Architecture with 6-Stack Transformer Text Encoder (Pure Self-Attention),
    and Pure Mamba-2 + ConvNeXt CFM Decoder (256-dim, 80-mel bins):
    - Text -> 6-Stack Pure Transformer Encoder (192-dim, 6 heads, RoPE, RMSNorm, Flash Attention, no Mamba)
    - to_latent Projection: 192 -> 80-dim Base Acoustic Mel Prior (mu_text_base)
    - Duration Predictor over Detached 192-dim Encoder Features (MFA ground-truth durations)
    - Repeat-Interleaved 80-dim Base Mel Latent -> mu_expanded
    - Pure Mamba-2 + ConvNeXt Flow Decoder (256-dim, 80 mel bins):
      * Input highway: concat([x_t, mu]) [80 + 80 = 160 -> 256]
      * Timestep FiLM conditioning (no mu FiLM conditioning)
      * 4 Alternating Pure Layers: ConvNeXt (k=7) + Mamba-2 Forward SSM (No global self-attention)
    """
    def __init__(
        self,
        encoder=None,
        duration_module=None,
        decoder_module=None,
        n_vocab=179,
        d_text=None,
        d_enc=192,
        d_codec=80,
        d_dec=384,
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

        # 6-Stack Transformer Text Encoder with Pure Attention (192-dim, 6 heads)
        self.encoder = encoder if encoder is not None else XTEncoder(
            dim=d_enc,
            depth=6,
            heads=6,
            gradient_checkpointing=gradient_checkpointing,
            attn_dropout=0.1,
            ff_dropout=0.1,
            layer_dropout=0.1,
        )

        # Linear projection from text encoder (192) to 80-dim Base Acoustic Prior (supervised via L1 loss)
        self.to_latent = nn.Linear(d_enc, d_codec, bias=False)

        # 3-Layer High-Capacity Duration Predictor (k=5, 256-dim, Receptive Field = 13 phonemes)
        self.duration = duration_module if duration_module is not None else duration(
            dims=d_enc,
            hidden_dim=256,
            target_dim=d_codec,
            kernel_size=5,
            dropout_p=0.1,
        )

        # Modified Tetra Decoder (Fwd Mamba2 + ConvNeXt)
        self.decoder = decoder_module if decoder_module is not None else MambaFlowTetraDecoder(
            d_in=d_codec,
            d_cond=d_codec,
            d_out=d_codec,
            d_model=d_dec,
            d_t=256,
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
        durations=None,
        mask=None,
        audio_mask=None,
        n_timesteps=30,
        temperature=1.0,
        length_scale=1.0,
        solver="euler",
        t_end=1.0,
        skip_decoder=False,
        cfg_dropout_prob=0.15,
        cfg_scale=1.0,
    ):
        alignment = None
        dur_pred = None
        mas_score = None

        target = target_latent

        # Text embedding with dropout
        x_in = self.emb_dropout(self.txt_emb(x))

        # 1. 6-Stack Pure Transformer Text Encoder (XTEncoder, 192-dim, RoPE, RMSNorm, Flash Attention)
        enc_out = self.encoder(x_in, mask=mask)

        # 2. Base Acoustic Prior (100-dim) for MAS alignment search
        mu_text_base = self.to_latent(enc_out)
        if mask is not None:
            m_t = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            mu_text_base = mu_text_base * m_t

        # Input feature representation: enc_out (192-dim)
        dur_feat = enc_out

        # 3. Duration Prediction Head (Pure Log-Duration Regression):
        dur_int, dur_pred, _ = self.duration(
            dur_feat.detach(),
            mask=mask,
            length_scale=length_scale,
        )

        # In training, use ground-truth MFA durations. In inference, use predicted durations.
        if durations is None:
            durations = dur_int

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

        # Direct 100-dim Acoustic Prior
        mu_expanded = latent_expanded
        if audio_mask is not None:
            m_c = audio_mask.unsqueeze(-1).float() if audio_mask.dim() == 2 else audio_mask.float()
            if m_c.size(1) > mu_expanded.size(1):
                m_c = m_c[:, :mu_expanded.size(1), :]
            elif m_c.size(1) < mu_expanded.size(1):
                mu_expanded = mu_expanded[:, :m_c.size(1), :]
            mu_expanded = mu_expanded * m_c

        # 5. Acoustic prior latent representation
        mu_conditioned = mu_expanded

        # Apply Classifier-Free Guidance (CFG) conditioning dropout during training
        if self.training and cfg_dropout_prob > 0.0:
            drop_mask = torch.rand(mu_conditioned.size(0), device=mu_conditioned.device) < cfg_dropout_prob
            mu_conditioned = mu_conditioned.masked_fill(drop_mask.view(-1, 1, 1), 0.0)

        # 6. Decoder: MambaFlow-UNet / Residual Flow Matching
        if skip_decoder:
            return (None, None), mu_expanded, alignment, dur_pred, mas_score

        if self.decoder is not None:
            if target is not None:
                # Align lengths between target and mu_conditioned
                min_T = min(target.size(1), mu_conditioned.size(1))
                target = target[:, :min_T, :]
                mu_conditioned = mu_conditioned[:, :min_T, :]
                mu_expanded = mu_expanded[:, :min_T, :]
                if audio_mask is not None:
                    audio_mask = audio_mask[:, :min_T]

                if self.use_residual_target:
                    # Residual target R = target - mu_conditioned.detach()
                    flow_target = target - mu_conditioned.detach()
                else:
                    # Direct Full Mel target
                    flow_target = target

                v_pred, u_target = self.decoder(
                    mu=mu_conditioned,
                    mask=audio_mask,
                    target=flow_target,
                )
                return (v_pred, u_target), mu_expanded, alignment, dur_pred, mas_score
            else:
                # Inference Mode: Decoder integrates from noise
                dec_out = self.decoder(
                    mu=mu_conditioned,
                    mask=audio_mask,
                    target=None,
                    n_timesteps=n_timesteps,
                    temperature=temperature,
                    solver=solver,
                    t_end=t_end,
                    #cfg_scale=cfg_scale,
                )
                if self.use_residual_target:
                    mel_pred = mu_conditioned + dec_out
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
