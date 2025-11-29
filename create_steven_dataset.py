"""
Convert the custom 'Steven' synthetic speaker dataset into a CREMA-D-style CSV.

This script:
  - Reads a metadata CSV describing Steven's lines (emotion, text).
  - Normalises emotion labels.
  - Locates matching audio files on disk.
  - Writes a CSV in the same general style as the CREMA-D CSV used for training.

Used as a one-off data-prep step to integrate the Steven synthetic speaker into
the same training pipeline as CREMA-D.
"""

import argparse
import csv
from pathlib import Path


# ========================= Mappings / config =========================

# Emotions we expect (the "base" label before _001 etc.)
VALID_EMOTIONS = {"neutral", "anger", "disgust", "fear", "happy", "sad"}

# Map slightly different text labels, I messed up some of the naming while creating the data.
EMO_MAP = {
    "angry": "anger",
    "anger": "anger",
    "sad": "sad",
    "neutral": "neutral",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
}


# ========================= Parsing helpers =========================

def parse_label(label: str):
    """
    Parse emotion labels like 'angry_001', 'neutral_010', 'happy_123' into
    (emotion, index) tuples.

    If there is no underscore present, the entire label is treated as the emotion
    and the index defaults to "000".

    The emotion is normalised through EMO_MAP so 'angry' -> 'anger'.
    """
    parts = label.split("_", 1)
    if len(parts) == 2:
        emo_raw, idx = parts[0].lower(), parts[1]
    else:
        emo_raw, idx = label.lower(), "000"

    emo_base = EMO_MAP.get(emo_raw, emo_raw)  # fall back to raw if unknown
    return emo_base, idx


def find_audio_file(audio_dir: Path, emo: str, idx: str):
    """
    Locate a matching audio file for a given emotion + index in audio_dir.

    For example, if emo='anger' and idx='001', this will try:
      - anger_001.wav / .mp3 / .flac / .m4a
      - angry_001.wav / etc. (for anger only, since naming might differ)

    Returns:
      POSIX-style path string to the audio file.

    Raises:
      FileNotFoundError if no matching file can be found.
    """
    stems_to_try = []

    # Normalised stem (anger_001 etc.)
    stems_to_try.append(f"{emo}_{idx}")

    # If this is anger, also try angry_XXX
    if emo == "anger":
        stems_to_try.append(f"angry_{idx}")

    # Try multiple possible audio file extensions since data varies
    for stem in stems_to_try:
        for ext in [".wav", ".mp3", ".flac", ".m4a"]:
            candidate = audio_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate.as_posix()

    # If none of the combinations exist, bail out with a detailed error
    raise FileNotFoundError(
        f"No audio file found for emotion '{emo}', index '{idx}' "
        f"(tried stems: {', '.join(stems_to_try)}) in {audio_dir}"
    )



def open_metadata_safely(path: Path):
    """
    Open the metadata CSV with a best-effort approach to encoding.

    Tries UTF-8 first, and if that fails, falls back to latin-1.

    Returns:
      (file_handle, csv_reader, fieldnames)
    """
    try:
        f = open(path, "r", encoding="utf-8")
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        return f, reader, fieldnames
    except UnicodeDecodeError:
        print("[warn] UTF-8 decode failed — retrying with latin-1")
        f = open(path, "r", encoding="latin-1")
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        return f, reader, fieldnames


# ========================= Main =========================

def main():
    """
    Entry point for building a CREMA-D-style CSV for the 'Steven' synthetic speaker.

    Expected metadata CSV format:
      - Columns: 'emotion', 'text'
      - emotion values like 'neutral_001', 'angry_010', etc.
      - audio_dir contains matching files (anger_001.wav, neutral_010.mp3, ...)

    The script:
      - Normalises emotion labels
      - Resolves the correct audio file
      - Injects an emotion tag into the text, e.g. "<anger> I am going to the store."
      - Applies a fixed speaker_id (2001) and constructs a speaker token "<spk=2001>"
      - Produces a CSV suitable for merging with the CREMA-D master CSV before training
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audio_dir",
        type=str,
        default="Steven",
    )
    ap.add_argument(
        "--metadata_csv",
        type=str,
        default="Steven/Steven_data.csv",
    )
    ap.add_argument(
        "--speaker_id",
        type=str,
        default="2001",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="Steven_cremad_format.csv",
    )
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    metadata_path = Path(args.metadata_csv)

    rows_out = []

    # Read metadata CSV with safe encoding handling
    f, reader, fieldnames = open_metadata_safely(metadata_path)

    if "emotion" not in fieldnames or "text" not in fieldnames:
        raise SystemExit("Steven_data.csv must have columns: emotion,text")

    for line_num, row in enumerate(reader, start=1):
        label = row["emotion"].strip()
        raw_text = row["text"].strip()

        # Skip any rows where emotion or text is missing
        if not label or not raw_text:
            print(f"[warn] Skipping empty row {line_num}")
            continue

        emo_base, idx = parse_label(label)

        if emo_base not in VALID_EMOTIONS:
            print(f"[warn] Unknown emotion '{emo_base}' on row {line_num}, keeping as-is.")

        intensity = "xx"  # no hi/md/lo info for synthetic speaker

        try:
            audio_rel = find_audio_file(audio_dir, emo_base, idx)
        except FileNotFoundError as e:
            print(f"[warn] {e} (row {line_num})")
            continue

        # Speaker information. 
        speaker_id = args.speaker_id
        speaker_token = f"<spk={speaker_id}>"

        # Emotion tag goes in front of the text, e.g. "<anger> text here"
        text_with_tag = f"<{emo_base}> {raw_text}"

        # utt_id is similar to CREMA-D's: actor_sentence_emotion_intensity, but adapted for Steven (STV marker) – e.g. 2001_STV_ANG_001
        emo_code = emo_base[:3].upper()  # NEU, ANG, DIS, FEA, HAP, SAD-ish
        utt_id = f"{speaker_id}_STV_{emo_code}_{idx}"

        rows_out.append(
            {
                "audio": audio_rel,
                "text": text_with_tag,
                "speaker_id": speaker_id,
                "speaker_token": speaker_token,
                "emotion": emo_base,
                "intensity": intensity,
                "utt_id": utt_id,
            }
        )

    f.close()

    if not rows_out:
        raise SystemExit("No rows created. Check your paths / metadata.")

    fieldnames_out = [
        "audio",
        "text",
        "speaker_id",
        "speaker_token",
        "emotion",
        "intensity",
        "utt_id",
    ]

    # Write out the final CSV which can be merged with the CREMA-D master CSV prior to training.
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames_out)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"[ok] Wrote {len(rows_out)} rows to {args.out_csv}")
    print("You can now merge this with cremad_master.csv for training.")


if __name__ == "__main__":
    main()

"""
Run with:
python scripts_new/create_steven_dataset.py \
  --audio_dir Steven \
  --metadata_csv Steven/Steven_data.csv \
  --speaker_id 2001 \
  --out_csv Steven_cremad_format.csv
"""