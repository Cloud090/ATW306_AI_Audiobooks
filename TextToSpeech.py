"""
Tagged Text-To-Speech component for the Emotional Audiobook pipeline.
This module:
  - Loads the merged Orpheus multi-speaker emotional TTS model.
  - Loads SNAC to decode acoustic tokens into 24 kHz waveforms.
  - Exposes an API for:
      - speak_text(text), which returns a single audio clip as a numpy array.
      - speak_json(items), which generates multiple clips and combines them.
  - Handles model, tokenizer, and SNAC loading once, then reuses them.
  - Normalises and combines segments into a single WAV per run.
"""

import os
import json
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from snac import SNAC
from scipy.signal import butter, filtfilt
import noisereduce as nr
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========================= Orpheus token config =========================

# Core Orpheus special token IDs (fixed for this model).
TOKENISER_LEN = 128256
START_HUMAN = TOKENISER_LEN + 3
END_HUMAN = TOKENISER_LEN + 4
END_OF_TEXT = 128009
EOS_SPEECH = 128258
START_SPEECH = TOKENISER_LEN + 1


# SNAC output sample rate.
SR = 24000 # 24 kHz


# Global variables to hold loaded model, tokenizer, and SNAC decoder.
_MODEL = None
_TOKENIZER = None
_SNAC = None


# ========================= SNAC helpers =========================

# Clamp SNAC code indices to valid ranges for each quantizer layer.
# Ensures all codes are within [0, num_codes-1] for each layer so decoding is stable and never hits out-of-range indices.
def clamp_layers_to_codebook(layers, snac):
    quantizers = snac.quantizer.quantizers
    clamped = []

    for i, layer in enumerate(layers):
        num_codes = quantizers[i].codebook.num_embeddings

        # Ensure layer is on the same device as the codebook.
        layer = layer.to(quantizers[i].codebook.weight.device)

        # Clamp values.
        layer = torch.clamp(layer, 0, num_codes - 1)

        clamped.append(layer)

    return clamped


# ========================= Checkpoint helper =========================

# Prefer a local checkpoint if it looks valid, otherwise fall back to HF.
def _resolve_checkpoint(local_path, hf_repo):
    config = os.path.join(local_path, "config.json")
    if os.path.isfile(config):
        if any(f.endswith(".safetensors") for f in os.listdir(local_path)):
            print(f"[Tagged text to speech] Using local model: {local_path}")
            return local_path
    print(f"[Tagged text to speech] Using HF model: {hf_repo}")
    return hf_repo


# ========================= Audio helpers =========================

# A low-pass filter to roll off high-end noise.
def _lowpass(a, cutoff=9000):
    b, c = butter(4, cutoff / (0.5 * SR), btype="low")
    return filtfilt(b, c, a)

# Basic audio cleanup (noisereduce + lowpass).
def _clean_audio(a):
    a = np.nan_to_num(a).astype("float32")
    noise = a[: int(0.3 * SR)]

    a = nr.reduce_noise(
        y=a,
        y_noise=noise,
        sr=SR,
        prop_decrease=0.85,
        stationary=False
    )

    a = _lowpass(a)
    return a.astype("float32")

# Decode flat SNAC codes into audio using the training layout.
def _decode_snac(snac, codes):
    L1, L2, L3 = [], [], []

    # Process exact SNAC frames of 7 tokens.
    for i in range(len(codes) // 7):
        base = 7 * i
        L1.append(codes[base])
        L2.append(codes[base + 1] - 4096)
        L3.append(codes[base + 2] - 8192)
        L3.append(codes[base + 3] - 12288)
        L2.append(codes[base + 4] - 16384)
        L3.append(codes[base + 5] - 20480)
        L3.append(codes[base + 6] - 24576)

    layers = [
        torch.tensor(L1).unsqueeze(0),
        torch.tensor(L2).unsqueeze(0),
        torch.tensor(L3).unsqueeze(0),
    ]

    layers = clamp_layers_to_codebook(layers, snac)  # Clamp to valid indices before decoding
    audio = snac.decode(layers).detach().squeeze().cpu().numpy()
    return audio


# Normalise to a target RMS and clamp to [-1, 1].
def _normalize(a, rms=0.1):
    a = a.astype(np.float32)
    power = np.sqrt(np.mean(a ** 2))
    if power < 1e-8:
        return a
    return np.clip(a * (rms / power), -1.0, 1.0).astype(np.float32)

# Combine per-segment WAVs into a single file.
def _combine_wavs(folder):
    files = sorted(
        [f for f in os.listdir(folder) if f.startswith("out_") and f.endswith(".wav")]
    )
    if not files:
        return None

    audio_parts = []
    for f in files:
        a, _ = sf.read(os.path.join(folder, f))
        audio_parts.append(a.astype("float32"))

    merged = np.concatenate(audio_parts)
    merged = _normalize(merged, 0.15)
    out_path = os.path.join(folder, "combined.wav")
    sf.write(out_path, merged, SR)
    return out_path


# ========================= Model loading =========================

# Load Orpheus + SNAC once and cache globally.
def preload_model(
    local_model_path="models/orpheus_merged_cremad_plus_steven_fp16_mspk_v5",
    hf_repo_id="Smallan/final_fine_tune_fp16" # Update to latest model as needed
):
    global _MODEL, _TOKENIZER, _SNAC

    if _MODEL is not None:
        return  # already loaded

    ckpt = _resolve_checkpoint(local_model_path, hf_repo_id)

    print("[Tagged text to speech] Loading SNAC decoder...")
    _SNAC = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

    print("[Tagged text to speech] Loading tokenizer...")
    _TOKENIZER = AutoTokenizer.from_pretrained(
        ckpt, trust_remote_code=True, use_fast=False
    )

    print("[Tagged text to speech] Loading model...")
    _MODEL = AutoModelForCausalLM.from_pretrained(
        ckpt,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("[Tagged text to speech] Model preloaded.")


# Helper to auto-load model on first use.
def _ensure_loaded():
    if _MODEL is None:
        preload_model()


# ========================= TTS =========================

# Text-to-speech for a single text string, returns raw audio as numpy array.
def speak_text(text):
    _ensure_loaded()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenise full prompt (includes <spk>, emotion, intensity tags).
    ids = _TOKENIZER(text, return_tensors="pt").input_ids.to(device)

    # Prepare model input with start/end tokens.
    start = torch.tensor([[START_HUMAN]], device=device)
    end = torch.tensor([[END_OF_TEXT, END_HUMAN]], device=device)
    inp = torch.cat([start, ids, end], dim=1)
    attn = torch.ones_like(inp)

    # Main generation settings tuned for audiobook-style output.
    gen = _MODEL.generate(
        input_ids=inp,
        attention_mask=attn,
        # 80 tokens = ~1 second of audio
        max_new_tokens=5000,    # Allow longer output, up to ~1 min of audio, but limited to stop runaway.
        # min_new_tokens=200,   # Ensure at least some audio is generated. Left out for flexibility.
        do_sample=True,
        temperature=0.6,        # Lower temperature for more coherent speech, higher for more variety
        top_p=0.9,              # Nucleus sampling to focus on top tokens
        repetition_penalty=1.1, # Mild penalty to reduce repetition
        eos_token_id=EOS_SPEECH, # Stop generation at speech EOS
        use_cache=True          # Use past key values for faster generation
    )[0]

    # Find speech start and drop any leading non-speech tokens.
    idx = (gen == START_SPEECH).nonzero(as_tuple=True)[0]
    if len(idx):
        gen = gen[idx[-1] + 1:]

    # Drop EOS markers.
    gen = gen[gen != EOS_SPEECH]

    # SNAC expects groups of 7 codes.
    usable = (len(gen) // 7) * 7
    tokens = [int(x) - 128266 for x in gen[:usable]]

    raw = _decode_snac(_SNAC, tokens)
    return _clean_audio(raw)


# Batch TTS for a list of {"text": "..."} dicts, used by the pipeline.
def speak_json(json_items, output_folder="outputs", progress=None):
    _ensure_loaded()

    # Validate input
    if not isinstance(json_items, list):
        raise ValueError("speak_json expects a list of {text:...} dicts.")

    os.makedirs(output_folder, exist_ok=True)

    texts = [x.get("text", "").strip() for x in json_items if x.get("text")]
    total = len(texts)

    if progress:
        progress(f"Generating spoken audio ({total} segments)")

    print(f"Generating spoken audio ({total} segments)")

    index = 0

    for t in texts:
        index += 1

        if progress:
            progress(f"Generating spoken audio ({total} segments)")

        print(f"Generating spoken audio ({total} segments)")

        out_path = os.path.join(output_folder, f"out_{index:03d}.wav")

        # Generate audio for this segment and save.
        audio = speak_text(t)

        # Save
        sf.write(out_path, audio, SR)

        if progress:
            progress(f"Saved segment {index}/{total}")

    # Merge all per-segment files into a combined .wav for convenience.
    combined = _combine_wavs(output_folder)

    if progress:
        progress("Combining all segments into final audio...")

    if progress:
        progress(f"Orpheus completed. Output folder: {output_folder}")

    return output_folder
