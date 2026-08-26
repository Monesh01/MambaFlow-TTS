"""
Monotonic Alignment Search (MAS) module for MambaFlow-TTS.
Provides GPU-accelerated alignment with vectorized CPU/NumPy fallback.
"""

import numpy as np
import torch

try:
    from super_monotonic_align import maximum_path as super_mas
except ImportError:
    super_mas = None


def maximum_path_numpy(value, mask):
    """
    Pure NumPy dynamic programming implementation of Viterbi Monotonic Alignment Search.
    value: [B, T_text, T_audio] log-likelihood or negative distance matrix
    mask:  [B, T_text, T_audio] boolean/binary valid mask
    Returns:
    path:  [B, T_text, T_audio] binary alignment matrix
    """
    B, T_x, T_y = value.shape
    path = np.zeros((B, T_x, T_y), dtype=np.int32)

    for b in range(B):
        # Determine valid text and audio lengths for batch item b
        m = mask[b]
        t_x = int(m.sum(axis=0)[0]) if m.ndim == 2 else T_x
        t_y = int(m.sum(axis=1)[0]) if m.ndim == 2 else T_y
        if t_x == 0 or t_y == 0:
            continue

        v = value[b, :t_x, :t_y]
        Q = np.full((t_x, t_y), -np.inf, dtype=np.float32)
        Q[0, 0] = v[0, 0]

        for y in range(1, t_y):
            for x in range(min(y + 1, t_x)):
                if x == 0:
                    prev_val = Q[0, y - 1]
                else:
                    prev_val = max(Q[x, y - 1], Q[x - 1, y - 1])
                Q[x, y] = prev_val + v[x, y]

        # Backtrack optimal Viterbi path from (t_x - 1, t_y - 1) to (0, 0)
        curr_x = t_x - 1
        for y in range(t_y - 1, -1, -1):
            path[b, curr_x, y] = 1
            if y > 0 and curr_x > 0:
                if Q[curr_x - 1, y - 1] >= Q[curr_x, y - 1]:
                    curr_x -= 1

    return path


def maximum_path(value, mask):
    """
    Computes optimal monotonic alignment path.
    value: [B, T_text, T_audio]
    mask:  [B, T_text, T_audio]
    """
    if value.is_cuda and super_mas is not None:
        return super_mas(value, mask)

    # Fast CPU / NumPy path
    device = value.device
    dtype = value.dtype
    v_np = value.detach().cpu().numpy().astype(np.float32)
    m_np = mask.detach().cpu().numpy().astype(bool)

    path_np = maximum_path_numpy(v_np, m_np)
    return torch.from_numpy(path_np).to(device=device, dtype=dtype)
