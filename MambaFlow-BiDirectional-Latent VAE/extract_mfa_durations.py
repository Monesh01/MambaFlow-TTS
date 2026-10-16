#!/usr/bin/env python3
"""
extract_mfa_durations.py

Full-fledged Phone-Level Duration Generation using Montreal Forced Aligner (MFA).
- Prepares audio and transcripts for forced alignment
- Runs MFA alignment using the pretrained english_us_mfa dictionary and english_mfa acoustic model
- Parses output TextGrids to get exact start/end timestamps for words and phones
- Converts timestamps to discrete latent frame counts (sr=48000, hop_length=960 -> 50.0 fps)
- Maps frame durations 1-to-1 onto the phoneme sequence tokens produced by text_to_sequence(text, ['english_cleaners2'])
- Guarantees len(durations) == len(sequence) and sum(durations) == total_frames (matching KVAE latent dimensions)
- Parallel multiprocessing execution for ultra-fast dataset processing
- Saves the integer duration list directly into train.csv, val.csv, and meta.csv under the 'durations' column
"""

import os
import sys
import json
import re
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure preprocessing module is discoverable from TTSModel if not in current directory
for p in [
    os.path.abspath(os.path.dirname(__file__)),
    "/home/monesh/TTSModel",
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

from preprocessing.text import text_to_sequence

# Default parameters for 48kHz KVAE latent space
DEFAULT_DATASET_DIR = "/home/monesh/ljspeech/LJSpeech-1.1"
DEFAULT_TRAIN_CSV = os.path.join(DEFAULT_DATASET_DIR, "train.csv")
DEFAULT_VAL_CSV = os.path.join(DEFAULT_DATASET_DIR, "val.csv")
DEFAULT_META_CSV = os.path.join(DEFAULT_DATASET_DIR, "meta.csv")
DEFAULT_WAVS_DIR = os.path.join(DEFAULT_DATASET_DIR, "wavs")
DEFAULT_LATENT_DIR = os.path.join(DEFAULT_DATASET_DIR, "latent")
DEFAULT_MEL_DIR = os.path.join(DEFAULT_DATASET_DIR, "mel_hifigan")
DEFAULT_TEXTGRID_DIR = os.path.join(DEFAULT_DATASET_DIR, "textgrids")
MFA_BIN = "/home/monesh/miniconda3/envs/aligner/bin/mfa"

# Target sample rate and hop length for KVAE latent representations (50 Hz frame rate)
SAMPLE_RATE = 48000
HOP_LENGTH = 960


def run_cmd(cmd, check=True):
    print(f"[CMD] {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        sys.exit(res.returncode)
    return res


def ensure_mfa_models():
    """Ensures acoustic model and pronunciation dictionary are downloaded."""
    mfa_path = MFA_BIN if os.path.exists(MFA_BIN) else "mfa"
    print("[+] Checking and downloading MFA models...")
    run_cmd(f"{mfa_path} model download acoustic english_mfa", check=False)
    run_cmd(f"{mfa_path} model download dictionary english_us_mfa", check=False)


def prepare_transcripts(df, wavs_dir):
    """Writes a .txt file for each utterance in the wavs directory."""
    print(f"[+] Preparing transcript files in {wavs_dir}...")
    count = 0
    for _, row in df.iterrows():
        audio_id = str(row["audio_id"])
        txt_path = os.path.join(wavs_dir, f"{audio_id}.txt")
        text = str(row.get("normalized_text", row.get("text", ""))).strip()
        if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            count += 1
    print(f"[+] Wrote {count} transcript (.txt) files.")


def parse_textgrid(tg_path):
    """
    Parses a Praat TextGrid file into word and phone intervals.
    Returns:
        words: list of (xmin, xmax, label)
        phones: list of (xmin, xmax, label)
    """
    with open(tg_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    words = []
    phones = []

    tier_blocks = re.split(r'item\s*\[\d+\]:', content)
    for block in tier_blocks[1:]:
        name_match = re.search(r'name\s*=\s*"(.*?)"', block)
        if not name_match:
            continue
        tier_name = name_match.group(1).lower()

        interval_matches = re.findall(
            r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"(.*?)"',
            block,
            re.DOTALL
        )
        for xmin, xmax, text in interval_matches:
            interval = (float(xmin), float(xmax), text.strip())
            if "word" in tier_name:
                words.append(interval)
            elif "phone" in tier_name:
                phones.append(interval)

    return words, phones


def map_intervals_to_clean_tokens(clean_text, words, phones, total_frames, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """
    Robust Dynamic Programming (DTW / Needleman-Wunsch) alignment between MFA phone
    intervals and the character sequence of clean_text.
    
    Correctly handles:
    - Multi-word contractions and collocations merged without spaces by espeak
      (e.g. 'there was' -> 'ðɛɹwˌʌz', 'that the' -> 'ðætðə', 'had been' -> 'hɐdbɪn')
    - Diphthongs (e.g. 'aɪ' in clean_text matching 'aj' in MFA)
    - Stress marks, diacritics, and punctuation
    - Leading, trailing, and inter-word/clause silence intervals
    - Guarantees: len(durations) == len(clean_text) and sum(durations) == total_frames
    """
    n_tokens = len(clean_text)
    durations = [0] * n_tokens

    if not phones or total_frames <= 0 or n_tokens == 0:
        base = total_frames // max(1, n_tokens)
        rem = total_frames - base * n_tokens
        durations = [base] * n_tokens
        if durations:
            durations[-1] += rem
        return durations

    # 1. Parse phone intervals into discrete frame durations
    fps = sample_rate / hop_length
    phone_frames = []
    for xmin, xmax, lbl in phones:
        f_start = int(round(xmin * fps))
        f_end = int(round(xmax * fps))
        dur = max(0, f_end - f_start)
        lbl_str = lbl.strip()
        is_sil = lbl_str in ("", "sil", "sp")
        phone_frames.append({
            "xmin": xmin,
            "xmax": xmax,
            "f_start": f_start,
            "f_end": f_end,
            "dur": dur,
            "lbl": lbl_str,
            "is_sil": is_sil,
        })

    # 2. Extract speech characters from clean_text (ignore diacritics, spaces, punctuation)
    char_list = []
    for i, ch in enumerate(clean_text):
        if ch not in ("ˈ", "ˌ", "ː", " ", ".", ",", "!", "?", ";", ":", '"', "'", "-"):
            char_list.append((i, ch))

    if not char_list:
        base = total_frames // max(1, n_tokens)
        durations = [base] * n_tokens
        durations[-1] += total_frames - sum(durations)
        return durations

    speech_phone_indices = [k for k, p in enumerate(phone_frames) if not p["is_sil"]]
    if not speech_phone_indices:
        base = total_frames // max(1, n_tokens)
        durations = [base] * n_tokens
        durations[-1] += total_frames - sum(durations)
        return durations

    speech_phones = [phone_frames[k] for k in speech_phone_indices]

    # 3. Phonetic similarity score for sequence alignment
    def phone_sim(ch, p_lbl):
        p_clean = p_lbl.lower()
        if p_clean in ("spn", "<unk>", "oov", ""):
            return 0.2  # Equal low cost for all speech characters in an unknown word
        if ch in p_clean or p_clean in ch:
            return 0.0
        if ch == "ð" and "d" in p_clean:
            return 0.0
        if ch == "θ" and "t" in p_clean:
            return 0.0
        if ch in ("ə", "ɐ", "ʌ") and any(v in p_clean for v in ("ə", "ɐ", "ʌ", "a")):
            return 0.0
        if ch in ("l", "ɫ") and ("l" in p_clean or "ɫ" in p_clean):
            return 0.0
        if ch in ("ɹ", "r") and ("ɹ" in p_clean or "r" in p_clean or "ɚ" in p_clean):
            return 0.0
        if ch in ("æ", "ɛ") and any(v in p_clean for v in ("æ", "ɛ", "e")):
            return 0.0
        return 1.0

    M = len(char_list)
    N = len(speech_phones)
    dp = [[float("inf")] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            cost = phone_sim(char_list[i-1][1], speech_phones[j-1]["lbl"])
            dp[i][j] = cost + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])

    i, j = M, N
    alignment = []
    while i > 0 and j > 0:
        alignment.append((char_list[i-1][0], j - 1))
        prev_scores = [
            (dp[i-1][j-1], i-1, j-1),
            (dp[i-1][j], i-1, j),
            (dp[i][j-1], i, j-1),
        ]
        prev_scores.sort(key=lambda x: x[0])
        _, next_i, next_j = prev_scores[0]
        i, j = next_i, next_j
    alignment.reverse()

    # Distribute speech phone durations
    phone_to_chars = {}
    for c_idx, sp_idx in alignment:
        phone_to_chars.setdefault(sp_idx, []).append(c_idx)

    for sp_idx, p_info in enumerate(speech_phones):
        chars = phone_to_chars.get(sp_idx, [])
        p_dur = p_info["dur"]
        if not chars:
            continue
        base = p_dur // len(chars)
        rem = p_dur % len(chars)
        for k, c_idx in enumerate(chars):
            durations[c_idx] += base + (1 if k < rem else 0)

    # 4. Distribute silence intervals:
    # 4a. Leading silence (before first speech phone)
    first_sp_orig_idx = speech_phone_indices[0]
    lead_dur = sum(p["dur"] for p in phone_frames[:first_sp_orig_idx])
    durations[0] += lead_dur

    # 4b. Trailing silence (after last speech phone)
    last_sp_orig_idx = speech_phone_indices[-1]
    trail_dur = sum(p["dur"] for p in phone_frames[last_sp_orig_idx+1:])
    durations[-1] += trail_dur

    # 4c. Intermediate pauses (between speech phones)
    for k in range(len(speech_phone_indices) - 1):
        idx1 = speech_phone_indices[k]
        idx2 = speech_phone_indices[k + 1]
        if idx2 > idx1 + 1:
            pause_dur = sum(phone_frames[m]["dur"] for m in range(idx1 + 1, idx2))
            if pause_dur > 0:
                c1 = phone_to_chars.get(k, [0])[-1]
                c2 = phone_to_chars.get(k + 1, [n_tokens - 1])[0]
                target_c = None
                for c_mid in range(c1 + 1, c2):
                    if clean_text[c_mid] in (".", ",", "!", "?", ";", ":"):
                        target_c = c_mid
                        break
                if target_c is None:
                    for c_mid in range(c1 + 1, c2):
                        if clean_text[c_mid] == " ":
                            target_c = c_mid
                            break
                if target_c is None:
                    target_c = c1
                durations[target_c] += pause_dur

    # 5. Exact total frame matching with total_frames
    diff = total_frames - sum(durations)
    if diff > 0:
        durations[-1] += diff
    elif diff < 0:
        rem = -diff
        for k in range(len(durations) - 1, -1, -1):
            if rem <= 0:
                break
            can_sub = durations[k]
            sub = min(can_sub, rem)
            durations[k] -= sub
            rem -= sub

    return durations


def _process_utterance_worker(item):
    """
    Worker function executed in parallel across CPU cores.
    item = (idx, audio_id, text, lat_cand, mel_cand, audio_cand, tg_path, sample_rate, hop_length)
    """
    idx, audio_id, text, lat_cand, mel_cand, audio_cand, tg_path, sample_rate, hop_length = item
    seq, clean_text = text_to_sequence(text, ["english_cleaners2"])

    # 1. Determine total frames
    total_frames = None
    if lat_cand and os.path.exists(lat_cand):
        try:
            arr = np.load(lat_cand, mmap_mode="r")
            total_frames = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
        except Exception:
            pass

    if total_frames is None and mel_cand and os.path.exists(mel_cand):
        try:
            m = np.load(mel_cand, mmap_mode="r")
            total_frames = m.shape[1] if m.shape[0] in (80, 100) else m.shape[0]
        except Exception:
            pass

    if total_frames is None and audio_cand and os.path.exists(audio_cand):
        try:
            import soundfile as sf
            info = sf.info(audio_cand)
            total_frames = int(round(info.duration * sample_rate / hop_length))
        except Exception:
            pass

    if total_frames is None:
        total_frames = max(1, int(round(len(seq) * 3.5)))

    # 2. Extract durations from TextGrid or Fallback
    has_tg = os.path.exists(tg_path)
    if has_tg:
        words, phones = parse_textgrid(tg_path)
        durs = map_intervals_to_clean_tokens(
            clean_text, words, phones, total_frames, sample_rate=sample_rate, hop_length=hop_length
        )
    else:
        base = total_frames // max(1, len(seq))
        rem = total_frames - base * len(seq)
        durs = [base] * len(seq)
        if durs:
            durs[-1] += rem

    # 3. Ensure length matches sequence exactly
    if len(durs) != len(seq):
        if len(durs) < len(seq):
            durs = durs + [0] * (len(seq) - len(durs))
        else:
            durs = durs[:len(seq)]

        d_sum = sum(durs)
        diff = total_frames - d_sum
        if diff > 0:
            durs[-1] += diff
        elif diff < 0:
            rem = -diff
            for k in range(len(durs) - 1, -1, -1):
                if rem <= 0:
                    break
                can_sub = durs[k]
                sub = min(can_sub, rem)
                durs[k] -= sub
                rem -= sub

    return idx, durs, has_tg


def process_dataset_durations(
    csv_path,
    tg_dir=DEFAULT_TEXTGRID_DIR,
    latent_dir=DEFAULT_LATENT_DIR,
    mel_dir=DEFAULT_MEL_DIR,
    wavs_dir=DEFAULT_WAVS_DIR,
    sample_rate=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
    num_workers=8,
):
    """
    Reads CSV, matches each row with its TextGrid / Latent, extracts 48k Hz phone-level durations,
    and updates the CSV with an exact 'durations' column.
    """
    df = pd.read_csv(csv_path)
    print(f"[+] Processing {len(df)} entries from {csv_path} at {sample_rate}Hz (hop_length={hop_length})...")

    work_items = []
    for idx, row in df.iterrows():
        audio_id = str(row["audio_id"])
        text = str(row.get("normalized_text", row.get("text", "")))

        # Find latent candidate
        lat_cand = None
        for lat_col in ["latent", "latent_path"]:
            if lat_col in row and pd.notna(row[lat_col]):
                lat_p = str(row[lat_col])
                if not os.path.isabs(lat_p) and latent_dir:
                    lat_p = os.path.join(latent_dir, lat_p)
                lat_cand = lat_p
                break
        if not lat_cand and latent_dir:
            lat_cand = os.path.join(latent_dir, f"{audio_id}.npy")

        # Find mel candidate
        mel_cand = None
        for mel_col in ["mel_path", "mel_spectrogram_path", "mel"]:
            if mel_col in row and pd.notna(row[mel_col]):
                mel_cand = str(row[mel_col])
                break
        if not mel_cand and mel_dir:
            mel_cand = os.path.join(mel_dir, f"{audio_id}.npy")

        # Find audio candidate
        audio_cand = None
        if "audio_path" in row and pd.notna(row["audio_path"]):
            ap = str(row["audio_path"])
            if not os.path.isabs(ap) and wavs_dir:
                ap = os.path.join(wavs_dir, os.path.basename(ap))
            audio_cand = ap
        elif wavs_dir:
            audio_cand = os.path.join(wavs_dir, f"{audio_id}.wav")

        tg_path = os.path.join(tg_dir, f"{audio_id}.TextGrid")
        work_items.append((
            idx,
            audio_id,
            text,
            lat_cand,
            mel_cand,
            audio_cand,
            tg_path,
            sample_rate,
            hop_length,
        ))

    results = [None] * len(df)
    matched_tg = 0
    missing_tg = 0

    desc = f"Durations for {os.path.basename(csv_path)}"
    if num_workers > 1 and len(work_items) > 100:
        chunksize = max(1, min(50, len(work_items) // (num_workers * 4)))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for idx, durs, has_tg in tqdm(
                executor.map(_process_utterance_worker, work_items, chunksize=chunksize),
                total=len(work_items),
                desc=desc,
            ):
                results[idx] = json.dumps(durs)
                if has_tg:
                    matched_tg += 1
                else:
                    missing_tg += 1
    else:
        for item in tqdm(work_items, desc=desc):
            idx, durs, has_tg = _process_utterance_worker(item)
            results[idx] = json.dumps(durs)
            if has_tg:
                matched_tg += 1
            else:
                missing_tg += 1

    df["durations"] = results

    # Reorder columns logically
    order = ["audio_id", "text", "normalized_text", "audio_path", "mel_path", "latent", "durations"]
    cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    df = df[cols]

    df.to_csv(csv_path, index=False)
    print(f"[+] Successfully saved updated {csv_path} with 48k Hz 'durations' column!")
    print(f"[+] Matched TextGrids: {matched_tg} | Fallbacks: {missing_tg}")
    return df


def main():
    parser = argparse.ArgumentParser(description="MFA Phone-level Duration Extraction for LJSpeech (48kHz Latent)")
    parser.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val_csv", type=str, default=DEFAULT_VAL_CSV)
    parser.add_argument("--meta_csv", type=str, default=DEFAULT_META_CSV)
    parser.add_argument("--wavs_dir", type=str, default=DEFAULT_WAVS_DIR)
    parser.add_argument("--latent_dir", type=str, default=DEFAULT_LATENT_DIR)
    parser.add_argument("--mel_dir", type=str, default=DEFAULT_MEL_DIR)
    parser.add_argument("--textgrid_dir", type=str, default=DEFAULT_TEXTGRID_DIR)
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--hop_length", type=int, default=HOP_LENGTH)
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel CPU workers for duration extraction")
    parser.add_argument("--num_jobs", type=int, default=10, help="Parallel CPU workers for MFA alignment")
    parser.add_argument("--skip_align", action="store_true", default=True, help="Skip MFA align step if TextGrids already exist")
    args = parser.parse_args()

    os.makedirs(args.textgrid_dir, exist_ok=True)
    mfa_path = MFA_BIN if os.path.exists(MFA_BIN) else "mfa"

    # Step 1: Ensure models & forced alignment (if needed)
    if not args.skip_align:
        ensure_mfa_models()

        if os.path.exists(args.train_csv):
            df_train = pd.read_csv(args.train_csv)
            prepare_transcripts(df_train, args.wavs_dir)
        if os.path.exists(args.val_csv):
            df_val = pd.read_csv(args.val_csv)
            prepare_transcripts(df_val, args.wavs_dir)

        print("=" * 80)
        print(f"[+] Launching MFA Alignment on {args.wavs_dir} -> {args.textgrid_dir}")
        print(f"[+] Using {args.num_jobs} parallel CPU worker processes...")
        print("=" * 80)
        align_cmd = (
            f"{mfa_path} align {args.wavs_dir} english_us_mfa english_mfa "
            f"{args.textgrid_dir} --clean --overwrite -j {args.num_jobs} "
            f"--use_mp --fine_tune"
        )
        run_cmd(align_cmd, check=True)

    # Step 2: Process TextGrids and update all CSV files with 48k Hz durations
    print("=" * 80)
    print(f"[+] Converting TextGrids into Phone Durations for 48k Hz (hop_length={args.hop_length})...")
    print("=" * 80)

    train_df = None
    val_df = None

    if os.path.exists(args.train_csv):
        train_df = process_dataset_durations(
            csv_path=args.train_csv,
            tg_dir=args.textgrid_dir,
            latent_dir=args.latent_dir,
            mel_dir=args.mel_dir,
            wavs_dir=args.wavs_dir,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            num_workers=args.num_workers,
        )

    if os.path.exists(args.val_csv):
        val_df = process_dataset_durations(
            csv_path=args.val_csv,
            tg_dir=args.textgrid_dir,
            latent_dir=args.latent_dir,
            mel_dir=args.mel_dir,
            wavs_dir=args.wavs_dir,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            num_workers=args.num_workers,
        )

    # Step 3: Keep meta.csv in sync if train and val are both available
    if train_df is not None and val_df is not None and args.meta_csv:
        meta_df = pd.concat([train_df, val_df], ignore_index=True)
        meta_df.to_csv(args.meta_csv, index=False)
        print(f"[+] Saved updated {args.meta_csv} ({len(meta_df)} rows)!")

    print("\n[+] MFA Phone Duration Generation & CSV Repopulation Complete!")


if __name__ == "__main__":
    main()
