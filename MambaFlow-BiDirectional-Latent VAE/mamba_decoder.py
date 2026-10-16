import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2


def reverse_valid_sequence(x, mask=None):
    """
    Reverses sequence x along time dimension (dim=1) exclusively for valid non-padded tokens.
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


class ContinuousSinusoidalPosEmb(nn.Module):
    """
    Continuous Sinusoidal Positional Embedding for scalar positions [B, SeqLen].
    pe(pos, 2i)   = sin(pos / (10000 ** (2i / dim)))
    pe(pos, 2i+1) = cos(pos / (10000 ** (2i / dim)))
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, pos):
        if pos.ndim == 1:
            pos = pos.unsqueeze(-1)
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half_dim, dtype=torch.float32, device=pos.device)
            / max(1, half_dim - 1)
        )
        args = pos.float().unsqueeze(-1) * freqs.view(*([1] * pos.ndim), half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.to(dtype=pos.dtype if pos.is_floating_point() else torch.float32)


SinusoidalPosEmb = ContinuousSinusoidalPosEmb


class TimeEmbeddingMLP(nn.Module):
    """
    Timestep Embedding MLP:
    SinusoidalPosEmb(d_model) on t*1000.0 -> Linear(d_model, d_model) -> SiLU -> Linear(d_model, d_model)
    Produces d_model-dimensional timestep conditioning vector.
    """
    def __init__(self, d_model=448):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, t):
        t = t.view(-1, 1).float() * 1000.0
        emb = self.sin_emb(t).squeeze(1)
        return self.mlp(emb)


class PointwiseDenseMLPBlock(nn.Module):
    """
    Pointwise Dense MLP Block: Linear(d_model -> 4*d_model) -> SiLU -> Linear(4*d_model -> d_model)
    """
    def __init__(self, d_model=448, mlp_hidden_dim=1792):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mlp_hidden_dim, d_model, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


DenseMLPBlock = PointwiseDenseMLPBlock


class DecoderLayer(nn.Module):
    """
    Single MambaFlow Decoder Layer:
    Order:
      1. Cross-Attention FIRST (Injects 192D Text into x via Pre-Norm RMSNorm, no timestep shifts)
      2. BiMamba-2 SECOND (Scans Noise + Fetched Text Together)
      3. Pointwise Dense MLP THIRD (Channel Mixing)
    Modulated via AdaLN-Zero from timestep embedding t_emb (6 parameters: BiMamba-2 and Dense MLP).
    All layer inputs normalized with RMSNorm.
    """
    def __init__(
        self,
        d_model=448,
        d_enc=192,
        num_heads=None,
        mlp_mult=4,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
    ):
        super().__init__()
        self.d_model = d_model
        if num_heads is None:
            num_heads = 7 if d_model == 448 else (8 if d_model % 8 == 0 else 7)

        # AdaLN-Zero MLP: generates 6 modulation parameters (2 sub-blocks [BiMamba-2, MLP] x [gamma, beta, alpha])
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # Sub-block 1: Cross-Attention
        self.norm1 = nn.RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            kdim=d_enc,
            vdim=d_enc,
            batch_first=True,
        )

        # Sub-block 2: BiMamba-2
        self.norm2 = nn.RMSNorm(d_model)
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

        # Sub-block 3: Pointwise Dense MLP
        self.norm3 = nn.RMSNorm(d_model)
        self.dense_mlp = PointwiseDenseMLPBlock(
            d_model=d_model,
            mlp_hidden_dim=mlp_mult * d_model,
        )

    def forward(self, x, t_emb, text_kv, text_mask=None, audio_mask=None, cond_drop_mask=None):
        m_audio = audio_mask.unsqueeze(-1).float() if (audio_mask is not None and audio_mask.dim() == 2) else (audio_mask.float() if audio_mask is not None else None)

        # 1. Predict 6 AdaLN-Zero parameters (BiMamba-2 and Pointwise Dense MLP)
        params = self.adaLN_modulation(t_emb).unsqueeze(1)  # [B, 1, 6 * d_model]
        (
            gamma_mamba, beta_mamba, alpha_mamba,
            gamma_mlp, beta_mlp, alpha_mlp,
        ) = params.chunk(6, dim=-1)

        # --- SUB-BLOCK 1: Cross-Attention FIRST (Pre-Norm RMSNorm only, no timestep shifts) ---
        norm1_x = self.norm1(x)

        if text_mask is not None:
            kp_mask = ~text_mask.bool()
            # Guard against all-masked rows to prevent softmax NaNs
            all_masked = kp_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                kp_mask = kp_mask & ~all_masked
        else:
            kp_mask = None

        h_ca, _ = self.cross_attn(
            query=norm1_x,
            key=text_kv,
            value=text_kv,
            key_padding_mask=kp_mask,
            need_weights=False,
        )
        if cond_drop_mask is not None:
            h_ca = h_ca.masked_fill(cond_drop_mask.view(-1, 1, 1), 0.0)
        if m_audio is not None:
            h_ca = h_ca * m_audio
        x = x + h_ca

        # --- SUB-BLOCK 2: BiMamba-2 SECOND ---
        norm2_x = self.norm2(x)
        shifted_2 = norm2_x * (1.0 + gamma_mamba) + beta_mamba

        fwd = self.mamba_fwd(shifted_2)
        if m_audio is not None:
            fwd = fwd * m_audio

        rev_in = reverse_valid_sequence(shifted_2, mask=audio_mask)
        bwd_rev = self.mamba_bwd(rev_in)
        if m_audio is not None:
            bwd_rev = bwd_rev * m_audio
        bwd = reverse_valid_sequence(bwd_rev, mask=audio_mask)
        if m_audio is not None:
            bwd = bwd * m_audio

        h_mamba = (fwd + bwd) * 0.7071067811865475  # 1/sqrt(2) keeps variance at 1.0
        if m_audio is not None:
            h_mamba = h_mamba * m_audio
        x = x + alpha_mamba * h_mamba

        # --- SUB-BLOCK 3: Pointwise Dense MLP THIRD ---
        norm3_x = self.norm3(x)
        shifted_3 = norm3_x * (1.0 + gamma_mlp) + beta_mlp
        h_mlp = self.dense_mlp(shifted_3)
        if m_audio is not None:
            h_mlp = h_mlp * m_audio
        x = x + alpha_mlp * h_mlp

        return x


class MambaDecoder(nn.Module):
    """
    Optimal Transport CFM Velocity Decoder with 6-Layer Architecture:
    Each Layer: Cross-Attention (192D Text K/V) -> BiMamba-2 -> Pointwise Dense MLP
    Conditioned via AdaLN-Zero from timestep embedding.
    """
    def __init__(
        self,
        d_in=128,
        d_cond=192,
        d_out=128,
        d_model=448,
        d_enc=192,
        mlp_hidden_dim=1792,
        num_blocks=6,
        sigma_min=1e-4,
        gradient_checkpointing=False,
        logit_normal_mean=0.0,
        logit_normal_std=1.0,
        **kwargs,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_cond = d_cond
        self.d_out = d_out
        self.d_model = d_model
        self.d_enc = d_enc
        self.sigma_min = sigma_min
        self.gradient_checkpointing = gradient_checkpointing
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

        # 1. Main Flow Highway: x_t [d_in -> d_model] + Frame Positional Embedding
        self.x_proj = nn.Linear(d_in, d_model, bias=False)
        nn.init.xavier_uniform_(self.x_proj.weight, gain=1.0)
        self.pos_emb_audio = SinusoidalPosEmb(dim=d_model)

        # 2. Text Duration-Centered Relative Positional Embedding
        self.pos_emb_text = SinusoidalPosEmb(dim=d_enc)

        # 3. Timestep Embedding
        self.t_embed = TimeEmbeddingMLP(d_model=d_model)

        # 4. Decoder Layers (CA -> BiMamba-2 -> Dense MLP)
        mlp_mult = mlp_hidden_dim // d_model if mlp_hidden_dim is not None else 4
        num_heads = kwargs.get("num_heads", 7 if d_model == 448 else (8 if d_model % 8 == 0 else 7))
        self.blocks = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                d_enc=d_enc,
                num_heads=num_heads,
                mlp_mult=mlp_mult,
                d_state=64,
                d_conv=4,
                expand=2,
                headdim=64,
            )
            for _ in range(num_blocks)
        ])

        # 5. Final Output Velocity Projection (After Layer 6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.final_norm = nn.RMSNorm(d_model)
        self.final_out_proj = nn.Linear(d_model, d_out, bias=False)
        nn.init.zeros_(self.final_out_proj.weight)

    def predict_velocity(
        self,
        x,
        t,
        text_repr=None,
        durations=None,
        text_mask=None,
        audio_mask=None,
        cond_drop_mask=None,
        **kwargs,
    ):
        if text_repr is None and "cond" in kwargs:
            text_repr = kwargs["cond"]
        if text_mask is None and "mask" in kwargs:
            text_mask = kwargs["mask"]
        if durations is None and "durations" in kwargs:
            durations = kwargs["durations"]
        if cond_drop_mask is None and "drop_mask" in kwargs:
            cond_drop_mask = kwargs["drop_mask"]

        if durations is not None and durations.dim() == 3:
            durations = durations.squeeze(-1)

        B, T_audio, _ = x.shape

        m_audio = audio_mask.unsqueeze(-1).float() if (audio_mask is not None and audio_mask.dim() == 2) else (audio_mask.float() if audio_mask is not None else None)
        m_text = text_mask.unsqueeze(-1).float() if (text_mask is not None and text_mask.dim() == 2) else (text_mask.float() if text_mask is not None else None)

        # 1. Project x_t [B, T, 128] -> [B, T, d_model] and add frame pos emb
        pos_audio = torch.arange(T_audio, device=x.device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        h = self.x_proj(x) + self.pos_emb_audio(pos_audio)
        if m_audio is not None:
            h = h * m_audio

        # 2. Add duration-centered relative pos emb to unexpanded 192D text encoder output
        if durations is not None:
            cum_d = torch.cumsum(durations.float(), dim=-1)
            starts = cum_d - durations.float()
            pos_text = starts + 0.5 * durations.float()
        else:
            L_text = text_repr.size(1) if text_repr is not None else 1
            pos_text = torch.arange(L_text, device=x.device, dtype=torch.float32).unsqueeze(0).expand(B, -1)

        if text_repr is not None:
            text_kv = text_repr + self.pos_emb_text(pos_text)
            if m_text is not None:
                text_kv = text_kv * m_text
        else:
            text_kv = torch.zeros(B, pos_text.size(1), self.d_enc, device=x.device, dtype=x.dtype)

        # Classifier-Free Guidance: zero out text_kv completely for dropped samples
        if cond_drop_mask is not None:
            text_kv = text_kv.masked_fill(cond_drop_mask.view(-1, 1, 1), 0.0)
        elif text_repr is not None:
            is_zero = (text_repr.abs().sum(dim=(1, 2)) == 0)
            if is_zero.any():
                text_kv = text_kv.masked_fill(is_zero.view(-1, 1, 1), 0.0)

        # 3. Embed flow timestep t [B] -> [B, d_model]
        t_emb = self.t_embed(t)

        # 4. Decoder Layers (6 Layers: CA -> BiMamba-2 -> MLP)
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                h = checkpoint(block, h, t_emb, text_kv, text_mask, audio_mask, cond_drop_mask, use_reentrant=False)
            else:
                h = block(h, t_emb, text_kv, text_mask=text_mask, audio_mask=audio_mask, cond_drop_mask=cond_drop_mask)

        # 5. After Layer 6: Final AdaLN + Zero-Init Linear Projection (d_model -> d_out)
        gamma_final, beta_final = self.final_adaLN(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h_final = self.final_norm(h) * (1.0 + gamma_final) + beta_final
        v_pred = self.final_out_proj(h_final)
        if m_audio is not None:
            v_pred = v_pred * m_audio

        return v_pred

    @torch.no_grad()
    def solve_euler(
        self,
        x_0,
        text_repr,
        durations,
        text_mask=None,
        audio_mask=None,
        n_timesteps=50,
        solver="euler",
        t_end=1.0,
        cfg_scale=1.0,
        **kwargs,
    ):
        B, T, _ = x_0.shape
        device = x_0.device
        dtype = x_0.dtype

        x = x_0
        m = audio_mask.unsqueeze(-1).float() if (audio_mask is not None and audio_mask.dim() == 2) else (audio_mask.float() if audio_mask is not None else None)
        if m is not None:
            x = x * m

        dt = float(t_end) / float(n_timesteps)
        use_cfg = (cfg_scale is not None and cfg_scale != 1.0 and cfg_scale > 0.0)

        for step in range(n_timesteps):
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)

            if use_cfg:
                x_batched = torch.cat([x, x], dim=0)
                t_batched = torch.cat([t_tensor, t_tensor], dim=0)
                text_batched = torch.cat([text_repr, text_repr], dim=0) if text_repr is not None else None
                dur_batched = torch.cat([durations, durations], dim=0) if durations is not None else None
                text_mask_batched = torch.cat([text_mask, text_mask], dim=0) if text_mask is not None else None
                audio_mask_batched = torch.cat([audio_mask, audio_mask], dim=0) if audio_mask is not None else None
                drop_mask = torch.cat([
                    torch.zeros(B, dtype=torch.bool, device=device),
                    torch.ones(B, dtype=torch.bool, device=device),
                ], dim=0)

                v_batched = self.predict_velocity(
                    x=x_batched,
                    t=t_batched,
                    text_repr=text_batched,
                    durations=dur_batched,
                    text_mask=text_mask_batched,
                    audio_mask=audio_mask_batched,
                    cond_drop_mask=drop_mask,
                )
                v_cond, v_uncond = v_batched.chunk(2, dim=0)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.predict_velocity(
                    x=x,
                    t=t_tensor,
                    text_repr=text_repr,
                    durations=durations,
                    text_mask=text_mask,
                    audio_mask=audio_mask,
                )

            if solver == "midpoint":
                if step == n_timesteps - 1:
                    x = x + dt * v
                else:
                    x_mid = x + (dt / 2.0) * v
                    if m is not None:
                        x_mid = x_mid * m
                    t_mid = torch.full((B,), t_val + dt / 2.0, device=device, dtype=dtype)
                    if use_cfg:
                        x_mid_batched = torch.cat([x_mid, x_mid], dim=0)
                        t_mid_batched = torch.cat([t_mid, t_mid], dim=0)
                        v_mid_batched = self.predict_velocity(
                            x=x_mid_batched,
                            t=t_mid_batched,
                            text_repr=text_batched,
                            durations=dur_batched,
                            text_mask=text_mask_batched,
                            audio_mask=audio_mask_batched,
                            cond_drop_mask=drop_mask,
                        )
                        v_mid_cond, v_mid_uncond = v_mid_batched.chunk(2, dim=0)
                        v_mid = v_mid_uncond + cfg_scale * (v_mid_cond - v_mid_uncond)
                    else:
                        v_mid = self.predict_velocity(
                            x=x_mid,
                            t=t_mid,
                            text_repr=text_repr,
                            durations=durations,
                            text_mask=text_mask,
                            audio_mask=audio_mask,
                        )
                    x = x + dt * v_mid
            else:
                x = x + dt * v

            if m is not None:
                x = x * m

        return x

    def forward(
        self,
        text_repr=None,
        durations=None,
        text_mask=None,
        audio_mask=None,
        target=None,
        n_timesteps=50,
        temperature=1.0,
        solver="euler",
        t_end=1.0,
        cfg_scale=1.0,
        cond_drop_mask=None,
        **kwargs,
    ):
        if text_repr is None and "cond" in kwargs:
            text_repr = kwargs["cond"]
        if text_mask is None and "mask" in kwargs:
            text_mask = kwargs["mask"]
        if durations is None and "durations" in kwargs:
            durations = kwargs["durations"]
        if cond_drop_mask is None and "drop_mask" in kwargs:
            cond_drop_mask = kwargs["drop_mask"]

        if durations is not None and durations.dim() == 3:
            durations = durations.squeeze(-1)

        if target is not None:
            # Training Mode: Direct Optimal Transport Flow Matching with Logit-Normal Timestep Sampling
            B = target.size(0)
            u = torch.randn(B, device=target.device, dtype=target.dtype) * self.logit_normal_std + self.logit_normal_mean
            t = torch.sigmoid(u)
            t = torch.clamp(t, min=1e-5, max=1.0 - 1e-5)
            x_0 = torch.randn_like(target)

            t_expand = t.view(B, 1, 1)
            x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x_0 + t_expand * target
            u_target = target - (1.0 - self.sigma_min) * x_0

            v_pred = self.predict_velocity(
                x=x_t,
                t=t,
                text_repr=text_repr,
                durations=durations,
                text_mask=text_mask,
                audio_mask=audio_mask,
                cond_drop_mask=cond_drop_mask,
            )
            return v_pred, u_target
        else:
            # Inference Mode:
            B = text_repr.size(0)
            if durations is not None:
                audio_lengths = durations.sum(dim=-1)
                T_total = int(audio_lengths.max().item())
            else:
                T_total = text_repr.size(1) * 4
                audio_lengths = torch.full((B,), T_total, device=text_repr.device)

            if audio_mask is None:
                audio_mask = torch.arange(T_total, device=text_repr.device)[None, :] < audio_lengths[:, None]

            temp_scale = max(float(temperature), 1e-4) if temperature is not None else 1.0
            x_0 = torch.randn(B, T_total, self.d_in, device=text_repr.device, dtype=text_repr.dtype) * temp_scale
            if audio_mask is not None:
                x_0 = x_0 * audio_mask.unsqueeze(-1).float()

            return self.solve_euler(
                x_0=x_0,
                text_repr=text_repr,
                durations=durations,
                text_mask=text_mask,
                audio_mask=audio_mask,
                n_timesteps=n_timesteps,
                solver=solver,
                t_end=t_end,
                cfg_scale=cfg_scale,
            )


# Backward and export aliases
decoder = MambaDecoder
MambaFlowDecoder = MambaDecoder
MambaFlowTetraDecoder = MambaDecoder
MambaFlowParallelDecoder = MambaDecoder
