import re
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from preprocessing.text import text_to_sequence


def clean_path(path_str, mount_prefix="/mnt/windows"):
    if not isinstance(path_str, str) or path_str.startswith("/home/"):
        return path_str
    if path_str.startswith(mount_prefix):
        return path_str
    path_str = path_str.replace("\\", "/")
    if re.match(r"^[a-zA-Z]:", path_str):
        path_str = re.sub(r"^[a-zA-Z]:[/]*", "", path_str)
        return os.path.normpath(os.path.join(mount_prefix, path_str.lstrip("/")))
    return os.path.normpath(path_str)


def load_normalization_stats(stats_path="/home/monesh/ljspeech/LJSpeech-1.1/stats_hifigan.py", dim=80):
    """
    Loads per-dimension normalization statistics (mean, std).
    Supports .npy files as well as Python stats files (e.g. stats_hifigan.py, stats_vocos.py).
    """
    # Check if direct Python stats file
    if os.path.exists(stats_path) and stats_path.endswith(".py"):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("stats_mod", stats_path)
            stats_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stats_mod)
            if hasattr(stats_mod, "MEL_MEAN") and hasattr(stats_mod, "MEL_STD"):
                mean = np.array(stats_mod.MEL_MEAN, dtype=np.float32)
                std = np.array(stats_mod.MEL_STD, dtype=np.float32)
                return mean, std
        except Exception:
            pass

    if os.path.exists(stats_path) and stats_path.endswith(".npy"):
        try:
            stats = np.load(stats_path, allow_pickle=True).item()
            mean = np.array(stats["mean"], dtype=np.float32)
            std = np.array(stats["std"], dtype=np.float32)
            return mean, std
        except Exception:
            stats = np.load(stats_path, allow_pickle=True)
            if len(stats) == 2:
                return np.array(stats[0], dtype=np.float32), np.array(stats[1], dtype=np.float32)

    # Check for stats_hifigan.py, stats_vocos.py or stats.py in the same directory
    stats_dir = os.path.dirname(stats_path) if os.path.dirname(stats_path) else "."
    for py_candidate in ["stats_hifigan.py", "stats_vocos.py", "stats.py"]:
        py_path = os.path.join(stats_dir, py_candidate)
        if os.path.exists(py_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("stats_mod", py_path)
                stats_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(stats_mod)
                if hasattr(stats_mod, "MEL_MEAN") and hasattr(stats_mod, "MEL_STD"):
                    mean = np.array(stats_mod.MEL_MEAN, dtype=np.float32)
                    std = np.array(stats_mod.MEL_STD, dtype=np.float32)
                    return mean, std
            except Exception:
                pass

    return np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32)


def denormalize_mel(latent, stats_path="/home/monesh/ljspeech/LJSpeech-1.1/stats_hifigan.py"):
    """
    Denormalizes mel latent back to raw scale: arr * std + mean.
    """
    mean, std = load_normalization_stats(stats_path, dim=latent.shape[-1])
    if isinstance(latent, torch.Tensor):
        mean_t = torch.tensor(mean, device=latent.device, dtype=latent.dtype)
        std_t = torch.tensor(std, device=latent.device, dtype=latent.dtype)
        return latent * std_t + mean_t
    return latent * std + mean


def collate_fn(batch):
    mels = [torch.tensor(x["mel"], dtype=torch.float32) for x in batch]
    mels = pad_sequence(mels, batch_first=True)

    lengths = torch.tensor([x["mel"].shape[0] for x in batch], dtype=torch.long)

    text_seqs = [x["ipa_seq"] for x in batch]
    ipa = pad_sequence([torch.tensor(x, dtype=torch.long) for x in text_seqs], batch_first=True, padding_value=0)
    ipa_lengths = torch.tensor([len(x) for x in text_seqs], dtype=torch.long)

    ret = {
        "mel": mels,
        "lengths": lengths,
        "ipa_lengths": ipa_lengths,
        "ipa": ipa,
    }

    if "durations" in batch[0] and batch[0]["durations"] is not None:
        dur_seqs = [torch.tensor(x["durations"], dtype=torch.long) for x in batch]
        ret["durations"] = pad_sequence(dur_seqs, batch_first=True, padding_value=0)

    return ret


class MyDataset(Dataset):
    """
    Clean Dataset Module for LJSpeech:
    - Loads 80-dim mel spectrogram from CSV file paths
    - Applies per-dimension z-score normalization: (x - mean) / (std + 1e-8)
    - Returns normalized 80-dim target for latent supervision and CFM flow matching
    - Supports loading precomputed phoneme durations directly from CSV
    """
    def __init__(self, csv_path, mount_prefix="/mnt/windows", stats_path="/home/monesh/ljspeech/LJSpeech-1.1/stats_hifigan.py", prefetch_data=False):
        self.df = pd.read_csv(csv_path)
        for col in ['mel_path', 'mel_spectrogram_path', 'mel']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).apply(lambda p: clean_path(p, mount_prefix))
        self.phoneme_cache = {}
        self.prefetch_data = prefetch_data
        self.stats_mean, self.stats_std = load_normalization_stats(stats_path, dim=80)

    def __len__(self):
        return len(self.df)

    def _get_phonemes(self, text):
        if text not in self.phoneme_cache:
            seq = text_to_sequence(text, ["english_cleaners2"])[0]
            self.phoneme_cache[text] = seq
        return self.phoneme_cache[text]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        mel_path = None
        for col in ['mel_path', 'mel_spectrogram_path', 'mel']:
            if col in row and pd.notna(row[col]):
                mel_path = clean_path(str(row[col]))
                break
        if mel_path is None:
            raise KeyError(f"Could not find mel path in row with columns: {list(row.index)}")

        _mel_raw = np.squeeze(np.load(mel_path, mmap_mode="r" if not self.prefetch_data else None))
        
        if _mel_raw.ndim == 2 and _mel_raw.shape[0] in [80, 100] and _mel_raw.shape[1] != _mel_raw.shape[0]:
            _mel_raw = _mel_raw.T

        _mel_norm = (_mel_raw.astype(np.float32) - self.stats_mean) / (self.stats_std + 1e-8)

        _text = ""
        for col in ['normalized_text', 'normalized', 'text', 'transcript', 'txt']:
            if col in row and pd.notna(row[col]):
                _text = str(row[col])
                break

        _ipa_seq = self._get_phonemes(_text)

        durations = None
        for col in ['durations', 'duration', 'durations_mfa', 'durs']:
            if col in row and pd.notna(row[col]):
                val = row[col]
                if isinstance(val, str):
                    import json
                    try:
                        durations = json.loads(val)
                    except Exception:
                        import ast
                        durations = ast.literal_eval(val)
                elif isinstance(val, (list, np.ndarray)):
                    durations = list(val)
                break

        if durations is not None:
            # Align length with phoneme sequence
            if len(durations) != len(_ipa_seq):
                if len(durations) < len(_ipa_seq):
                    durations = list(durations) + [0] * (len(_ipa_seq) - len(durations))
                else:
                    durations = list(durations[:len(_ipa_seq)])

            # Match total duration to mel frame length
            mel_len = _mel_norm.shape[0]
            dur_sum = sum(durations)
            diff = mel_len - dur_sum
            if diff != 0:
                durations = list(durations)
                durations[-1] = max(1, durations[-1] + diff)

        item = {
            "mel": _mel_norm,
            "text": _text,
            "ipa_seq": _ipa_seq,
        }
        if durations is not None:
            item["durations"] = durations

        return item


def align_seq_len(predict, target, mask):
    min_len = min(predict.size(1), target.size(1), mask.size(1))
    return predict[:, :min_len], target[:, :min_len], mask[:, :min_len]


def masked_mse_loss(predict, target, mask):
    """Masked Mean Squared Error loss across valid sequence tokens."""
    predict, target, mask = align_seq_len(predict, target, mask)
    if predict.ndim == 3 and target.ndim == 2:
        target = target.unsqueeze(-1)

    loss = F.mse_loss(predict, target, reduction='none')
    mask_float = mask.unsqueeze(-1).float() if mask.ndim < loss.ndim else mask.float()
    loss = loss * mask_float
    num_feats = predict.shape[-1] if predict.ndim > 2 else 1
    valid_elements = mask_float.sum() * num_feats
    return loss.sum() / (valid_elements + 1e-8)


def masked_covariance_loss(predict, target, mask):
    """
    Batch Covariance Matching Loss (72 x 72 covariance alignment).
    Computes feature covariance matrices across all valid batch/time positions
    and returns MSE between predicted and target covariance matrices.
    """
    predict, target, mask = align_seq_len(predict, target, mask)
    if predict.ndim == 3 and target.ndim == 2:
        target = target.unsqueeze(-1)

    mask_float = mask.unsqueeze(-1).float() if mask.ndim < predict.ndim else mask.float()
    valid_count = mask_float.sum().clamp(min=1.0)

    # Compute global mean per feature dimension over valid masked positions
    mean_pred = (predict * mask_float).sum(dim=(0, 1), keepdim=True) / valid_count
    mean_tgt = (target * mask_float).sum(dim=(0, 1), keepdim=True) / valid_count

    # Zero-center features for valid tokens
    pred_centered = (predict - mean_pred) * mask_float
    tgt_centered = (target - mean_tgt) * mask_float

    D = predict.shape[-1]
    pred_flat = pred_centered.reshape(-1, D)
    tgt_flat = tgt_centered.reshape(-1, D)

    denom = (valid_count - 1.0).clamp(min=1.0)
    cov_pred = torch.matmul(pred_flat.T, pred_flat) / denom
    cov_tgt = torch.matmul(tgt_flat.T, tgt_flat) / denom

    return F.mse_loss(cov_pred, cov_tgt)


def masked_latent_loss(
    predict,
    target,
    mask,
    w_mse=1.0,
    w_cov=0.1,
):
    """
    Masked Latent Loss: w_mse * Masked MSE + w_cov * Batch Covariance Loss (72 dims)
    """
    loss_mse = masked_mse_loss(predict, target, mask)
    loss_cov = masked_covariance_loss(predict, target, mask)

    total_loss = w_mse * loss_mse + w_cov * loss_cov
    return total_loss, {
        "mse": loss_mse,
        "cov": loss_cov,
    }


def masked_smooth_l1_loss(predict, target, mask, beta=1.0):
    """
    Masked Smooth L1 (Huber) Loss across valid sequence tokens.
    Prevents outlier duration gradients from destabilizing training.
    """
    predict, target, mask = align_seq_len(predict, target, mask)
    if predict.ndim == 3 and target.ndim == 2:
        target = target.unsqueeze(-1)

    loss = F.smooth_l1_loss(predict, target, beta=beta, reduction='none')
    mask_float = mask.unsqueeze(-1).float() if mask.ndim < loss.ndim else mask.float()
    loss = loss * mask_float
    num_feats = predict.shape[-1] if predict.ndim > 2 else 1
    valid_elements = mask_float.sum() * num_feats
    return loss.sum() / (valid_elements + 1e-8)


