"""
Prepare SNAC-encoded datasets from CREMA-D-style CSVs.

This script:
  - Reads one or more CREMA-D-style CSVs (including Steven_cremad_format.csv).
  - Resolves audio file paths and resamples everything to 24 kHz.
  - Runs SNAC to encode each audio file into 7-codes-per-frame acoustic tokens.
  - Drops duplicate frames to match the Unsloth / Orpheus reference notebook.
  - Builds a Hugging Face DatasetDict with train/validation splits.
  - Saves the final dataset to disk for use in model training.

Used as a data-prep step for Orpheus multi-speaker emotional TTS, not in the final runtime pipeline.
"""

import os, json, locale, argparse, csv
from pathlib import Path
import torch
import soundfile as sf
import torchaudio.transforms as T
from datasets import Dataset, DatasetDict
from snac import SNAC

# ========================= Config =========================

# Default CREMA-D audio location (your local path). Can be overridden with --audio_root.
AUDIO_ROOT_DEFAULT = "/mnt/4SSD/Uni/Year 3/Tri 3/CD_DATA/AudioWAV"
TARGET_SR = 24000   # SNAC model expects 24 kHz audio

# Force UTF-8 encoding for CSV reading/writing
locale.getpreferredencoding = lambda: "UTF-8"


# ========================= CSV helpers =========================

def read_csv_cremad(path):
    """
    Read a CREMA-D style CSV (e.g. cremad_train.csv, cremad_val.csv, Steven_cremad_format.csv)
    and map columns into a standard schema for SNAC encoding.

    Expected header:
        ["audio", "text", "speaker_id", "speaker_token", "emotion", "intensity", "utt_id"]

    Returns:
        List[dict]: each dict contains:
          - text
          - emotion
          - intensity
          - speaker (id or name)
          - audio (original audio path or filename reference)
    """
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapped = {
                # Text already includes <emotion> tags and any speaker/emotion markup
                "text": (row.get("text") or "").strip(),
                "emotion": row.get("emotion"),
                "intensity": row.get("intensity"),
                "speaker": row.get("speaker_id") or row.get("speaker"),
                "audio": row.get("audio") or row.get("filename"),
            }
            rows.append(mapped)
    return rows

def resolve_audio_path(rec, audio_root):
    """
    Resolve the actual audio file path for a given record from the CSV.

    Tries, in order:
      1) 'orig_path' or 'path' if present and exists.
      2) 'audio' / 'file' / 'filename' directly (absolute or relative).
      3) Same filename under 'audio_root' (explicit or default).
      4) Same filename under ./data/AudioWAV as a final fallback.

    Returns:
        str or None: the resolved path, or None if nothing can be found.
    """
    # 1) Explicit path fields
    for key in ("orig_path", "path"):
        p = rec.get(key)
        if p and os.path.exists(p):
            return p

    # 2) Direct filename reference
    fname = rec.get("audio") or rec.get("file") or rec.get("filename")
    if not fname:
        return None

    # If fname itself is a valid path, use it
    if os.path.exists(fname):
        return fname
    
    # Try relative to current working directory
    cand_rel = os.path.join(".", fname)
    if os.path.exists(cand_rel):
        return cand_rel

    # 3) Try under audio_root (CREMA-D location or override)
    if audio_root:
        p1 = os.path.join(audio_root, fname)
        p2 = os.path.join(audio_root, os.path.basename(fname))
        p3 = os.path.join(audio_root, os.path.splitext(os.path.basename(fname))[0] + ".wav")
        for p in (p1, p2, p3):
            if os.path.exists(p):
                return p

    # 4) Fallback to ./data/AudioWAV
    guess_dir = os.path.join("data", "AudioWAV")
    p4 = os.path.join(guess_dir, fname)
    p5 = os.path.join(guess_dir, os.path.basename(fname))
    p6 = os.path.join(guess_dir, os.path.splitext(os.path.basename(fname))[0] + ".wav")
    for p in (p4, p5, p6):
        if os.path.exists(p):
            return p

    return None


# ========================= Audio + SNAC helpers =========================

def load_and_resample(path, resamplers):
    """
    Load an audio file using soundfile and resample it to TARGET_SR (24 kHz).

    Args:
        path (str): path to the audio file.
        resamplers (dict): cache of torchaudio.Resample objects keyed by original sample rate.

    Returns:
        torch.Tensor: tensor of shape (1, 1, T) suitable for SNAC.encode, or None on failure.
    """
    wav, sr = sf.read(path, always_2d=False)
    if wav is None:
        return None
    
    import numpy as np

    # Convert stereo -> mono if needed
    if wav.ndim == 2:
        wav = wav.mean(axis=-1)
    wav_t = torch.from_numpy(wav.astype("float32")).unsqueeze(0)  # (1, T)

    # Reuse resamplers for each sample rate to avoid re-creating them
    if sr not in resamplers:
        resamplers[sr] = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
    wav_24k = resamplers[sr](wav_t)  # (1, T)
    return wav_24k.unsqueeze(0)      # (1,1,T) for SNAC

def interleave_snac_codes(c):
    """
    Interleave SNAC code streams into a single 7-codes-per-frame sequence.

    This reproduces the Unsloth/Orpheus notebook's packing layout:

      c[0]: (1, F)
      c[1]: (1, 2F)
      c[2]: (1, 4F)

    For each frame i in [0, F):
      - construct 7 tokens with fixed offsets so that Orpheus can unpack them back.
    """
    base = 128266
    off1 = 4096
    out = []
    F = c[0].shape[1]
    for i in range(F):
        out.append(              c[0][0][i].item()      + base)
        out.append(              c[1][0][2*i].item()    + base + off1)
        out.append(              c[2][0][4*i].item()    + base + 2*off1)
        out.append(              c[2][0][4*i+1].item()  + base + 3*off1)
        out.append(              c[1][0][2*i+1].item()  + base + 4*off1)
        out.append(              c[2][0][4*i+2].item()  + base + 5*off1)
        out.append(              c[2][0][4*i+3].item()  + base + 6*off1)
    return out

def drop_duplicate_frames(codes):
    """
    Drop repeated frames like the reference notebook.

    The code sequence is grouped into 7-code "frames".
    We inspect the first code of each frame; if it matches the last one in the
    current output, we treat it as a duplicate and skip that frame.
    """
    if not codes:
        return codes
    
    # Start with the first frame (7 codes)
    out = codes[:7]
    for i in range(7, len(codes), 7):
        # Compare first code of this frame to the first code of the last frame in 'out'
        if codes[i] != out[-7]:
            out.extend(codes[i:i+7])
    return out

def build_hf_dataset(rows, device, snac, audio_root):
    """
    Given a list of row dicts (from one or more CREMA-style CSVs),
    build a Hugging Face Dataset with SNAC codes and metadata.

    For each row:
      - resolve the audio path
      - load + resample to 24 kHz
      - encode with SNAC to get codes
      - interleave + drop duplicates
      - store text, codes_list, emotion, intensity, speaker, and paths

    Returns:
      datasets.Dataset
    """
    resamplers = {}
    records, skipped = [], 0

    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            skipped += 1
            continue

        apath = resolve_audio_path(r, audio_root)
        if not apath:
            skipped += 1
            continue

        try:
            wav_24k = load_and_resample(apath, resamplers)  # (1,1,T)
            if wav_24k is None:
                skipped += 1
                continue

            with torch.inference_mode():
                c = snac.encode(wav_24k.to(device))  # List of 3 tensors

            codes = interleave_snac_codes(c)
            codes = drop_duplicate_frames(codes)
            if not codes:
                skipped += 1
                continue

            records.append({
                "text": text,
                "codes_list": codes,
                "emotion": r.get("emotion"),
                "intensity": r.get("intensity"),
                "speaker": r.get("speaker"),
                "audio_file": r.get("audio"),
                "path_used": apath,
            })

        except Exception as e:
            print(f"[warn] failed {apath}: {e}")
            skipped += 1
            continue

    print(f"Built {len(records)} examples (skipped {skipped}).")
    return Dataset.from_list(records)


# ========================= Main =========================

def main():
    """
    Entry point for SNAC dataset preparation.

    Typical usage:
      - Read CREMA-D train/val CSVs.
      - Optionally add one or more extra CSVs (e.g. Steven_cremad_format.csv).
      - Optionally oversample the extra speaker data to balance the dataset.
      - Encode all audio with SNAC and save a DatasetDict to disk.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/cremad_train.csv",)
    ap.add_argument("--val_csv",   default="data/cremad_val.csv",)
    ap.add_argument("--extra_train_csv", action="append", default=[],)
    ap.add_argument("--extra_val_csv", action="append", default=[],)
    ap.add_argument("--out_dir",   default="data/cremad_snac_24k",)
    ap.add_argument("--audio_root",  default=None,)
    args = ap.parse_args()

    # Decide audio root (CREMA-D; Steven uses explicit paths in 'audio')
    audio_root = args.audio_root or os.environ.get("AUDIO_ROOT", AUDIO_ROOT_DEFAULT)
    print("Using AUDIO_ROOT:", audio_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)

    # Load main CREMA-D CSVs
    train_rows = read_csv_cremad(args.train_csv)
    val_rows   = read_csv_cremad(args.val_csv)

    # Load extra CSVs (e.g. Steven_cremad_format.csv)
    for extra in args.extra_train_csv:
        print(f"Adding extra train CSV: {extra}")
        train_rows.extend(read_csv_cremad(extra))

    # Oversampling for extra train CSVs to increase Stevens weight
    if args.extra_train_csv:
        print("Oversampling Steven's dataset...")
        for extra in args.extra_train_csv:
            steven_rows = read_csv_cremad(extra)
            # Increase factor (3, 5, 10, etc.) if more weight is needed (it is...)
            oversample_factor = 3
            train_rows.extend(steven_rows * oversample_factor)
            print(f"  Added {len(steven_rows) * oversample_factor} oversampled rows.")

    # Extra CSVs for validation set
    for extra in args.extra_val_csv:
        print(f"Adding extra val CSV: {extra}")
        val_rows.extend(read_csv_cremad(extra))

    print(f"Loaded CSVs: train={len(train_rows)}  val={len(val_rows)}")

    # Build HF Datasets with SNAC codes
    train_ds = build_hf_dataset(train_rows, device, snac, audio_root)
    val_ds   = build_hf_dataset(val_rows,   device, snac, audio_root)

    dsd = DatasetDict({"train": train_ds, "validation": val_ds})
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(args.out_dir)

    print("Saved to", args.out_dir)
    print("train samples:", len(train_ds), "val samples:", len(val_ds))

    # Quick emotion distribution summary for sanity checking
    emo_train = {}
    for ex in train_ds:
        emo_train[ex.get("emotion")] = emo_train.get(ex.get("emotion"), 0) + 1
    print("train emotion counts:", emo_train)

if __name__ == "__main__":
    main()

"""
Run with:
python scripts_new/new_prep_for_snac.py \
  --train_csv data/cremad_train.csv \
  --val_csv data/cremad_val.csv \
  --extra_train_csv data/Steven_cremad_format.csv \
  --audio_root "/mnt/4SSD/Uni/Year 3/Tri 3/CD_DATA/AudioWAV" \
  --out_dir data/cremad_plus_steven_snac_24k
"""