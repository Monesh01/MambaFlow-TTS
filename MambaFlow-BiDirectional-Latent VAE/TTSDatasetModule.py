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


def collate_fn(batch):
    mels = [torch.tensor(x["mel"], dtype=torch.float32) for x in batch]
    mels = pad_sequence(mels, batch_first=True)

    lengths = torch.tensor([x["mel"].shape[0] for x in batch], dtype=torch.long)

    text_seqs = [x["ipa_seq"] for x in batch]
    ipa = pad_sequence([torch.tensor(x, dtype=torch.long) for x in text_seqs], batch_first=True, padding_value=0)
    ipa_lengths = torch.tensor([len(x) for x in text_seqs], dtype=torch.long)

    ret = {
        "mel": mels,
        "latent": mels,
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
    Dataset Module for LJSpeech:
    - Loads 64-dim KVAE latent directly from CSV file paths without normalization
    - Returns raw float32 latent representation for CFM flow matching
    - Supports loading precomputed phoneme durations directly from CSV
    """
    def __init__(self, csv_path, mount_prefix="/mnt/windows", target_dim=128, prefetch_data=False):
        self.df = pd.read_csv(csv_path)
        self.target_dim = target_dim
        for col in ['latent', 'latent_path', 'kvae_latent', 'mel_path', 'mel_spectrogram_path', 'mel']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).apply(lambda p: clean_path(p, mount_prefix))
        self.phoneme_cache = {}
        self.prefetch_data = prefetch_data

        self.latent_mean = None
        self.latent_std = None
        
        # Robust candidate path search for latent_mean.npy and latent_std.npy
        cand_dirs = [
            os.path.dirname(os.path.abspath(__file__)),
            os.path.dirname(os.path.abspath(csv_path)),
            os.getcwd(),
        ]
        mean_path, std_path = None, None
        for d in cand_dirs:
            mp = os.path.join(d, "latent_mean.npy")
            sp = os.path.join(d, "latent_std.npy")
            if os.path.exists(mp) and os.path.exists(sp):
                mean_path, std_path = mp, sp
                break

        if mean_path is not None and std_path is not None:
            self.latent_mean = np.load(mean_path).astype(np.float32)
            self.latent_std = np.load(std_path).astype(np.float32)
            print(f"[+] Loaded 128-dim latent_mean.npy and latent_std.npy for Z-Score standardization from: {mean_path}")
        else:
            try:
                from stats import LATENT_MEAN_ARR, LATENT_STD_ARR
                self.latent_mean = LATENT_MEAN_ARR
                self.latent_std = LATENT_STD_ARR
                print(f"[+] Loaded 128-dim LATENT_MEAN and LATENT_STD from stats.py for Z-Score standardization.")
            except ImportError:
                print("[-] Warning: No latent mean/std statistics found. Using unnormalized latents.")

    def __len__(self):
        return len(self.df)

    def _get_phonemes(self, text):
        if text not in self.phoneme_cache:
            seq = text_to_sequence(text, ["english_cleaners2"])[0]
            self.phoneme_cache[text] = seq
        return self.phoneme_cache[text]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        target_path = None
        for col in ['latent', 'latent_path', 'kvae_latent', 'mel_path', 'mel_spectrogram_path', 'mel']:
            if col in row and pd.notna(row[col]):
                target_path = clean_path(str(row[col]))
                break
        if target_path is None:
            raise KeyError(f"Could not find latent or mel path in row with columns: {list(row.index)}")

        _raw = np.squeeze(np.load(target_path, mmap_mode="r" if not self.prefetch_data else None))
        
        if _raw.ndim == 2 and _raw.shape[0] in [64, 72, 80, 100, 128] and _raw.shape[1] != _raw.shape[0]:
            _raw = _raw.T

        _latent = _raw.astype(np.float32)

        # 128-dim Z-Score Standardization: (x - mean) / (std + 1e-8) -> mean=0, std=1.0
        if self.latent_mean is not None and self.latent_std is not None:
            _latent = (_latent - self.latent_mean) / (self.latent_std + 1e-8)

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

            # Match total duration to target frame length
            target_len = _latent.shape[0]
            dur_sum = sum(durations)
            diff = target_len - dur_sum
            if diff != 0:
                durations = list(durations)
                durations[-1] = max(1, durations[-1] + diff)

        item = {
            "mel": _latent,
            "latent": _latent,
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


