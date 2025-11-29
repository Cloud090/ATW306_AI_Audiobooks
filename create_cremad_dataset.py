"""
Create a Hugging Face dataset from local CREMA-D audio files for multi-speaker emotional TTS training.

This script:
  - Recursively scans a CREMA-D directory for .wav files.
  - Parses filenames to extract actor, sentence, emotion, and intensity.
  - Maps codes to human-readable labels.
  - Builds a HF Dataset with audio + text + metadata.
  - Pushes the dataset to the Hugging Face Hub.

Used as a data-prep step for initial training, not in the final runtime pipeline.
"""
import os, re, argparse, pandas as pd
from pathlib import Path
from datasets import Dataset, Audio
from huggingface_hub import login


# ========================= Mappings / config =========================

# Map CREMA-D sentence acronyms to full text
# Gathered from CREMA-D documentation
SENTENCE_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

# Map CREMA-D three letter emotion codes to labels
EMO_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# Map CREMA-D intensity codes to labels
INT_MAP = {
    "LO": "low",
    "MD": "medium",
    "HI": "high",
    "XX": "unspecified",
}

# Filename pattern
# E.g. 1001_IEO_HAP_HI.wav
FN_RE = re.compile(r"^(?P<actor>\d{4})_(?P<sent>[A-Z]{3})_(?P<emo>[A-Z]{3})_(?P<int>[A-Z]{2})\.wav$")


# ========================= Scanning helpers =========================

def scan_wavs(root):
    """
    Recursively scan a CREMA-D directory for .wav files and extract metadata.

    This parses the filename to get:
      - actor ID
      - sentence code
      - emotion label
      - intensity label

    Then builds rows suitable for turning into a HF Dataset, with:
      - audio path
      - text (with speaker prefix, for Orpheus multispeaker)
      - speaker ID
      - sentence, emotion, intensity metadata
    """
    rows = []
    for p in Path(root).rglob("*.wav"):
        m = FN_RE.match(p.name)
        if not m: 
            # Skip anything that doesn't match the CREMA-D filename pattern
            continue

        actor = m["actor"]
        sent  = m["sent"]
        emo   = EMO_MAP.get(m["emo"], m["emo"]) # fallback to raw code if unknown
        inten = INT_MAP.get(m["int"], m["int"]) # fallback to raw intensity code
        text  = SENTENCE_MAP.get(sent, sent)    # fallback to code if not in map
        speaker = f"spk{actor}" # Speaker ID used by the TTS model (e.g. "spk1058")

        # Prefixed text is the convention used for Orpheus multispeaker training:
        # "spk1058: It's eleven o'clock"
        prefixed = f"{speaker}: {text}"

        rows.append({
            "audio": str(p.resolve()),
            "text": prefixed,
            "speaker": speaker,
            "sentence": sent,
            "emotion": emo,
            "intensity": inten,
        })
    return rows


# ========================= Main =========================

def main():
    """
    Build a HF Dataset from CREMA-D wav files and push it to the Hub.

    Expected usage:
      - Point --cremad_dir at the root folder containing CREMA-D .wav files.
      - Provide a Hugging Face repo_id and token.
      - The script will:
        * scan the directory
        * convert it to a HF Dataset
        * cast the audio column properly
        * push the dataset (sharded) to your HF account

    This was just used as a data-prep step for training, not for the final runtime pipeline.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cremad_dir", required=True")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--hf_token", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    login(args.hf_token)    # Authenticate to Hugging Face Hub

    # Scan local CREMA-D folder into a list of dict rows
    rows = scan_wavs(args.cremad_dir)
    if not rows:
        raise SystemExit("No wavs found")

    # Convert to a HF Dataset. audio paths stay as-is, we attach audio metadata below
    ds = Dataset.from_pandas(pd.DataFrame(rows))

    # Keep the original sampling rate for now, resampling can be done later
    ds = ds.cast_column("audio", Audio())

    # Create (or reuse) the dataset repo on HF Hub
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=args.hf_token)

    # Push the dataset to the Hub.
    # Needed to shard it to avoid upload stalls.
    ds.push_to_hub(
        args.repo_id,
        private=args.private,
        max_shard_size="64MB",
    )

    print(f"Pushed dataset to {args.repo_id}")

if __name__ == "__main__":
    main()

"""
Run with:
python scripts_new/create_cremad_dataset.py \                                                     venv 3.13.7  16:09 
  --cremad_dir data/AudioWAV \
  --repo_id Smallan/crema-d-multi \
  --hf_token TOKEN \
  --private\


"""