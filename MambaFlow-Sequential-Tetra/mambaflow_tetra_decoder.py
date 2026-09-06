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


class BiMamba2GatedFusionBlock(nn.Module):
    """
    Bidirectional Mamba-2 block with Gated Forward-Backward Fusion and AdaLN Timestep Modulation.
    - If use_residual=True: returns x + alpha * h_out (AdaLN-Zero residual)
    - If use_residual=False: returns h_out directly without residual shortcut bypass
    """
    def __init__(
        self,
        d_model=384,
        d_t=256,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        use_residual=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_residual = use_residual
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
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # AdaLN modulation driven by timestep t
        if self.use_residual:
            # AdaLN-Zero predicts scale (gamma), shift (beta), and residual gate (alpha)
            self.adaln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_t, 3 * d_model, bias=True),
            )
            nn.init.zeros_(self.adaln[1].weight)
            nn.init.zeros_(self.adaln[1].bias)
        else:
            # Standard AdaLN: predicts scale (gamma) and shift (beta)
            self.adaln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_t, 2 * d_model, bias=True),
            )
            nn.init.zeros_(self.adaln[1].weight)
            nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x, t_emb=None, mask=None):
        res = x

        # AdaLN modulation
        if t_emb is not None:
            if self.use_residual:
                gamma, beta, alpha = self.adaln(t_emb).unsqueeze(1).chunk(3, dim=-1)
                h = (1.0 + gamma) * self.norm(x) + beta
            else:
                gamma, beta = self.adaln(t_emb).unsqueeze(1).chunk(2, dim=-1)
                h = (1.0 + gamma) * self.norm(x) + beta
                alpha = 1.0
        else:
            h = self.norm(x)
            alpha = 1.0

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
        out = self.out_proj(fused)
        if m is not None:
            out = out * m

        # Return with or without residual shortcut
        if self.use_residual:
            return res + alpha * out
        else:
            return out


class ConvNeXtAdaLNBlock(nn.Module):
    """
    ConvNeXt-1D Local Refinement Block with AdaLN-Zero Timestep Conditioning:
    - Depthwise Conv1D with large kernel (k=17) for local temporal & harmonic continuity
    - AdaLN-Zero modulation: predicts scale (gamma), shift (beta), and residual gate (alpha) from t_emb
    - Inverted Bottleneck: Linear (dim -> 4*dim) -> SiLU -> Linear (4*dim -> dim)
    - Residual connection ensures smooth refinement of input features
    """
    def __init__(self, dim=384, d_t=256, kernel_size=17, mlp_ratio=4):
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
        self.pw2 = nn.Linear(mlp_ratio * dim, dim, bias=False)

        # AdaLN-Zero: predicts scale (gamma), shift (beta), and residual gate (alpha)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t, 3 * dim, bias=True),
        )
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x, t_emb=None, mask=None):
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
        h = self.pw2(h)

        if m_c is not None:
            h = h * m_c.float()

        return res + alpha * h


class MambaFlowTetraDecoder(nn.Module):
    """
    Tetra Bidirectional Mamba-2 Sequential Decoder with ConvNeXt Local Smoothing:
    - Pure full-resolution sequential processing (eliminates multiscale aliasing & dual audio).
    - Layer 1 (BiMamba-2): Conditioned on t + mu, with residual connection.
    - Layer 2 (BiMamba-2): Conditioned on t only, NO residual connection (pure transform).
    - Layer 3 (BiMamba-2): Conditioned on t + mu, NO residual connection (pure transform).
    - Layer 4 (ConvNeXt): Conditioned on t only, with residual connection (local smoothing).
    - Output: RMSNorm -> Linear(d_model, d_out) zero-initialized.
    """
    def __init__(
        self,
        d_in=100,
        d_cond=100,
        d_out=100,
        d_model=384,
        d_t=256,
        kernel_size=17,
        sigma_min=1e-4,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_cond = d_cond
        self.d_out = d_out
        self.d_model = d_model
        self.d_t = d_t
        self.sigma_min = sigma_min
        self.gradient_checkpointing = gradient_checkpointing

        # Timestep embedding network
        self.t_embed = TimestepEmbedding(d_t=d_t)

        # Layer 1 Input: concatenates noisy state x_t and prior mu
        self.in_proj = nn.Linear(d_in + d_cond, d_model, bias=False)

        # Layer 1: BiMamba-2 (Conditioned on t + mu via input, with residual)
        self.layer1 = BiMamba2GatedFusionBlock(
            d_model=d_model,
            d_t=d_t,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            use_residual=True,
        )

        # Layer 2: BiMamba-2 (Conditioned on t only, NO residual connection)
        self.layer2 = BiMamba2GatedFusionBlock(
            d_model=d_model,
            d_t=d_t,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            use_residual=False,
        )

        # Layer 3 mu projection: injects acoustic prior mu into Layer 3
        self.mu_proj3 = nn.Linear(d_cond, d_model, bias=False)

        # Layer 3: BiMamba-2 (Conditioned on t + mu, NO residual connection)
        self.layer3 = BiMamba2GatedFusionBlock(
            d_model=d_model,
            d_t=d_t,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            use_residual=False,
        )

        # Layer 4: ConvNeXt Local Smoothing Block (Conditioned on t only, with residual)
        self.layer4_conv = ConvNeXtAdaLNBlock(
            dim=d_model,
            d_t=d_t,
            kernel_size=kernel_size,
            mlp_ratio=4,
        )

        # Output Projection Head
        self.final_norm = nn.RMSNorm(d_model)
        self.final_out_proj = nn.Linear(d_model, d_out, bias=False)
        nn.init.zeros_(self.final_out_proj.weight)

    def _run_block(self, block, h, t_emb, mask):
        if self.training and self.gradient_checkpointing:
            return checkpoint(block, h, t_emb, mask, use_reentrant=False)
        return block(h, t_emb=t_emb, mask=mask)

    def predict_velocity(self, x, t, mu, mask=None):
        """
        Predicts CFM velocity v_theta at time t given noisy state x and prior mu.
        """
        B, T, _ = x.size()

        # Timestep embedding [B, d_t]
        t_emb = self.t_embed(t)

        # Input projection (concatenating x and mu) -> Layer 1 gets t + mu
        cat_in = torch.cat([x, mu], dim=-1)
        h0 = self.in_proj(cat_in)
        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            h0 = h0 * m
        else:
            m = None

        # Layer 1: BiMamba-2 (Conditioned on t + mu, WITH residual)
        h1 = self._run_block(self.layer1, h0, t_emb, mask)

        # Layer 2: BiMamba-2 (Conditioned on t only, NO residual)
        h2 = self._run_block(self.layer2, h1, t_emb, mask)

        # Layer 3: Inject mu conditioning, BiMamba-2 (Conditioned on t + mu, NO residual)
        mu_inj = self.mu_proj3(mu)
        if m is not None:
            mu_inj = mu_inj * m
        h2_mu = h2 + mu_inj
        h3 = self._run_block(self.layer3, h2_mu, t_emb, mask)

        # Layer 4: ConvNeXt (Conditioned on t only, NO mu, WITH residual)
        h4 = self._run_block(self.layer4_conv, h3, t_emb, mask)

        # Final output projection
        h_norm = self.final_norm(h4)
        v = self.final_out_proj(h_norm)

        if m is not None:
            v = v * m

        return v

    @torch.no_grad()
    def solve_euler(self, mu, mask=None, n_timesteps=30, temperature=1.0, solver="euler", t_end=1.0):
        """
        Continuous ODE integration solving dx/dt = v_theta(x, t, mu).
        """
        B, T, _ = mu.size()
        device = mu.device
        dtype = mu.dtype

        # Initial state from Gaussian noise scaled by temperature
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

            # Predict velocity at current state
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

        return x

    def forward(self, mu, mask=None, target=None, n_timesteps=30, temperature=1.0, solver="euler", t_end=1.0):
        B, T, _ = mu.size()

        if target is not None:
            # Training Mode: Optimal Transport Flow Matching
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
            # Inference Mode: Continuous ODE integration
            return self.solve_euler(
                mu=mu,
                mask=mask,
                n_timesteps=n_timesteps,
                temperature=temperature,
                solver=solver,
                t_end=t_end,
            )


# Backward and export aliases
decoder = MambaFlowTetraDecoder
