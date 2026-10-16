import torch
import torch.nn as nn
import torch.nn.functional as F
from trans_encoder import XTEncoder
from duration_predictor import duration
from mamba_decoder import MambaDecoder


class MambaFlowTTSModel(nn.Module):
    """
    MambaFlow-TTS Architecture with:
    - 8-Stack Pure Transformer Text Encoder (192-dim, 6 heads, RoPE, RMSNorm, Flash Attention)
    - 3-Layer High-Capacity Duration Predictor (k=5, 192-dim, hidden 256-dim, Receptive Field = 13 phonemes)
    - 6-Layer Optimal Transport CFM Decoder (256-dim, 128-dim DACAE continuous audio latents):
        * Direct 128D -> 256D audio highway projection + frame positional embeddings
        * 192D unexpanded phoneme text representation + duration-centered relative positional embeddings
        * Continuous flow timestep conditioning via AdaLN-Zero (6 parameters per layer for BiMamba-2 and MLP)
        * 3 sub-blocks per layer:
            1. Cross-Attention FIRST (Queries=audio noisy latent, Keys/Values=unexpanded 192D text representation, Pre-Norm RMSNorm)
            2. BiMamba-2 SECOND (Forward + Backward scans with symmetric additive fusion)
            3. Pointwise Dense MLP THIRD (Channel mixing 256 -> 1024 -> 256)
        * Zero-initialized final velocity projection (256 -> 128)
    """
    def __init__(
        self,
        encoder=None,
        duration_module=None,
        decoder_module=None,
        n_vocab=179,
        d_text=None,
        d_enc=192,
        d_codec=128,
        d_dec=448,
        num_dec_blocks=6,
        emb_dropout=0.0,
        cfg_dropout=0.15,
        use_residual_target=False,
        gradient_checkpointing=False,
        logit_normal_mean=0.0,
        logit_normal_std=1.0,
        **kwargs,
    ):
        super().__init__()
        self.d_enc = d_enc
        self.d_codec = d_codec
        self.d_dec = d_dec
        self.num_dec_blocks = num_dec_blocks
        self.d_text = d_enc if d_text is None else d_text
        self.cfg_dropout = cfg_dropout
        self.use_residual_target = use_residual_target

        self.txt_emb = nn.Embedding(n_vocab, self.d_text)
        self.emb_dropout = nn.Dropout(emb_dropout)

        # 8-Stack Transformer Text Encoder with Pure Attention (192-dim, 8 layers, 6 heads, head_dim=32)
        self.encoder = encoder if encoder is not None else XTEncoder(
            dim=d_enc,
            depth=8,
            heads=6 if d_enc == 192 else max(1, d_enc // 32),
            gradient_checkpointing=gradient_checkpointing,
            attn_dropout=0.1,
            ff_dropout=0.1,
            layer_dropout=0.1,
        )

        # 3-Layer High-Capacity Duration Predictor (k=5, 192-dim, hidden 256-dim, Receptive Field = 13 phonemes)
        self.duration = duration_module if duration_module is not None else duration(
            dims=d_enc,
            hidden_dim=256,
            target_dim=d_codec,
            kernel_size=5,
            dropout_p=0.1,
        )

        # 6-Layer CFM Decoder with Cross-Attention + BiMamba-2 + Pointwise Dense MLP (448-dim)
        self.decoder = decoder_module if decoder_module is not None else MambaDecoder(
            d_in=d_codec,
            d_cond=d_enc,
            d_out=d_codec,
            d_model=d_dec,
            d_enc=d_enc,
            mlp_hidden_dim=4 * d_dec,
            num_blocks=num_dec_blocks,
            gradient_checkpointing=gradient_checkpointing,
            logit_normal_mean=logit_normal_mean,
            logit_normal_std=logit_normal_std,
            **kwargs,
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
        n_timesteps=50,
        temperature=1.0,
        length_scale=1.0,
        solver="euler",
        t_end=1.0,
        skip_decoder=False,
        cfg_scale=1.0,
        **kwargs,
    ):
        alignment = None
        dur_pred = None
        mas_score = None

        # 1. Text embedding with dropout
        x_in = self.emb_dropout(self.txt_emb(x))

        # 2. 8-Stack Pure Transformer Text Encoder (XTEncoder, 192-dim, RoPE, RMSNorm, Flash Attention)
        enc_out = self.encoder(x_in, mask=mask)
        if mask is not None:
            m_t = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            enc_out = enc_out * m_t

        # 3. Duration Prediction Head (Pure Log-Duration Regression):
        dur_int, dur_pred, _ = self.duration(
            enc_out.detach(),
            mask=mask,
            length_scale=length_scale,
        )

        # In training, use ground-truth MFA durations. In inference, use predicted durations.
        if durations is None:
            durations = dur_int

        if durations.dim() == 3:
            durations = durations.squeeze(-1)

        # Clamp durations to ensure valid frame counts
        durations = torch.clamp(durations, min=0)
        for b in range(durations.size(0)):
            if durations[b].sum() == 0:
                durations[b, 0] = 1

        # 4. Determine audio length and mask
        audio_lengths = durations.sum(dim=-1)
        if target_latent is not None:
            T_audio = target_latent.size(1)
        else:
            T_audio = int(audio_lengths.max().item())

        if audio_mask is None:
            audio_mask = torch.arange(T_audio, device=x.device)[None, :] < audio_lengths[:, None]

        if skip_decoder:
            return (None, None), enc_out, alignment, dur_pred, mas_score

        if self.decoder is not None:
            if target_latent is not None:
                # Align audio sequence length to target if there is any mismatch
                min_T = min(target_latent.size(1), audio_mask.size(1))
                target = target_latent[:, :min_T, :]
                step_audio_mask = audio_mask[:, :min_T]

                drop_mask = None
                if self.training and self.cfg_dropout > 0.0:
                    drop_mask = torch.rand(x.size(0), device=x.device) < self.cfg_dropout

                v_pred, u_target = self.decoder(
                    text_repr=enc_out,
                    durations=durations,
                    text_mask=mask,
                    audio_mask=step_audio_mask,
                    target=target,
                    cond_drop_mask=drop_mask,
                )
                return (v_pred, u_target), enc_out, alignment, dur_pred, mas_score
            else:
                # Inference Mode: Continuous ODE integration from Gaussian noise
                dec_out = self.decoder(
                    text_repr=enc_out,
                    durations=durations,
                    text_mask=mask,
                    audio_mask=audio_mask,
                    target=None,
                    n_timesteps=n_timesteps,
                    temperature=temperature,
                    solver=solver,
                    t_end=t_end,
                    cfg_scale=cfg_scale,
                )
                return dec_out, enc_out

        return None, enc_out

        return None, enc_out


# Backward compatibility alias
Model = MambaFlowTTSModel
