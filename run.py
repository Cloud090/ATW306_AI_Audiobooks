"""
Run inference with the merged Orpheus multi-speaker emotional TTS model.

This script:
  - Loads a merged Orpheus checkpoint (local or from HF).
  - Loads SNAC for decoding acoustic tokens back to 24 kHz waveforms.
  - Takes text prompts from prompts.json.
  - Generates emotional speech based on embedded tags (<spk=...> <emotion> <intensity>).
  - Cleans up the audio (noise reduction + lowpass).
  - Saves each line as an individual WAV.
  - Optionally combines all WAVs into a single audiobook-style output.

This file is the main testing and evaluation tool used during development.
This is not part of the final pipeline, but essential to demonstrate model behaviour.
"""
import time
import os
import re
import json
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from snac import SNAC
import noisereduce as nr
from scipy.signal import butter, filtfilt
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========================= Checkpoint helper =========================

def resolve_checkpoint(local_path: str, hf_repo: str):
    """
    Decide whether to use a local model folder or a HF repo.

    A local path is considered valid if it contains:
      - config.json
      - at least one .safetensors file
    """
    # Must contain at least a config.json + one .safetensors file
    config_path = os.path.join(local_path, "config.json")

    if os.path.exists(local_path) and os.path.isfile(config_path):
        # check if safetensors exist
        has_weights = any(
            f.endswith(".safetensors") 
            for f in os.listdir(local_path)
        )
        if has_weights:
            print(f"[INFO] Using LOCAL model: {local_path}")
            return local_path

    print(f"[INFO] Local model not found, using HF repo: {hf_repo}")
    return hf_repo


# ========================= Config =========================

# Model and tokeniser in the same folder. If model not local, will pull from HF.
LOCAL_MODEL_PATH = "models/orpheus_merged_cremad_plus_steven_fp16_mspk_v5"
HF_REPO_ID = "TEST Smallan/orpheus_merged_cremad_full_fp16_mspk"

CKPT = resolve_checkpoint(LOCAL_MODEL_PATH, HF_REPO_ID)

# JSON file with prompts.
PROMPTS_JSON = "data/prompts.json"

# Where to save the generated wavs.
OUTPUT_ROOT = "outputs"

# Where to save combined runs.
OUTPUT_COMBINED = os.path.join(OUTPUT_ROOT, "combined")

# Filename prefix inside each run folder.
BASE_PREFIX = "out_"

# Device selection.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Orpheus special token IDs (fixed for this model).
TOKENISER_LEN = 128256
START_HUMAN = TOKENISER_LEN + 3
END_HUMAN = TOKENISER_LEN + 4
END_OF_TEXT = 128009
EOS_SPEECH = 128258
START_SPEECH = TOKENISER_LEN + 1


# ========================= Helpers =========================

# Clamp SNAC code indices safely, staying fully on the correct device.
def clamp_layers_to_codebook(layers, snac):
    """
    Clamp SNAC code indices to valid ranges for each quantizer layer.

    Ensures all codes are within [0, num_codes-1] for each layer so
    decoding is stable and never hits out-of-range indices.
    """
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

# Load prompts from JSON.
def load_json_prompts(path: str):
    """
    Load prompts from JSON.

    Supports:
      1) ["str1", "str2", ...]
      2) [{"text": "..."} , ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            return [s.strip() for s in data if s.strip()]
        if all(isinstance(x, dict) and "text" in x for x in data):
            return [str(x["text"]).strip() for x in data if str(x["text"]).strip()]
        raise ValueError("prompts.json list must be strings or dicts with 'text'.")

    raise ValueError("Unsupported prompts.json format.")


# ========================= Model load =========================

# SNAC: text tokens -> 24 kHz audio.
snac_load_start = time.perf_counter()
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
# snac = snac.to(device)
snac_load_time = time.perf_counter() - snac_load_start

# Tokenizer.
tok_load_start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(
    CKPT, # Expects tokeniser files in same folder as model, move anything token related from the lora to the merged folder when adding new models
    trust_remote_code=True,
    use_fast=False,
)
tok_load_time = time.perf_counter() - tok_load_start

# Model.
model_load_start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    CKPT, 
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model_load_time = time.perf_counter() - model_load_start


# ========================= Generation =========================

# Prompt format is expected to include tags, e.g.:
# "<spk=2001> <happy> <0.7> This is an example line of dialogue."
# Emotion + intensity are interpreted by the fine-tuned Orpheus model.
def generate_audio(prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    start = torch.tensor([[START_HUMAN]], device=device)
    end = torch.tensor([[END_OF_TEXT, END_HUMAN]], device=device)

    inp = torch.cat([start, ids, end], dim=1)
    attn = torch.ones_like(inp)

    # Generation params
    gen = model.generate(
        input_ids=inp,  # Input token IDs including start and end tokens.
        attention_mask=attn, # Attention mask for the input tokens.
        max_new_tokens=5000, # Upper limit to prevent runaway generation. Capable of ~60 seconds but runs into issues with these longer generations.
        min_new_tokens=700, # Force at least ~9 seconds of audio to avoid early stopping. ~80 tokens per second.
        do_sample=True, # Enable sampling for more varied outputs.
        temperature=0.6, # Lower temperature for more coherent and consistent speech. Higher for more variety but risk of gibberish.
        top_p=0.9, # Nucleus sampling to focus on top tokens.
        repetition_penalty=1.08, # Penalty to reduce repetition.
        eos_token_id=EOS_SPEECH, # End of speech token.
        num_return_sequences=1, # Generate a single sequence.
        use_cache=True, # Use past key values for faster generation.
    )
    row = gen[0]

    # Find where the actual speech starts.
    idx = (row == START_SPEECH).nonzero(as_tuple=True)[0]
    if len(idx):
        row = row[idx[-1] + 1:]

    # Drop EOS tokens.
    row = row[row != EOS_SPEECH]

    # SNAC expects groups of 7 codes.
    trim = (row.size(0) // 7) * 7
    codes = [int(t) - 128266 for t in row[:trim]]
    print(f"[debug] generated {len(codes)} acoustic tokens for this line")


    # Unpack into the three SNAC code streams.
    def split_codes(code_list):
        L1, L2, L3 = [], [], []
        for i in range((len(code_list) + 1) // 7):
            L1.append(code_list[7 * i])
            L2.append(code_list[7 * i + 1] - 4096)
            L3.append(code_list[7 * i + 2] - 8192)
            L3.append(code_list[7 * i + 3] - 12288)
            L2.append(code_list[7 * i + 4] - 16384)
            L3.append(code_list[7 * i + 5] - 20480)
            L3.append(code_list[7 * i + 6] - 24576)
        return [
            torch.tensor(L1).unsqueeze(0),
            torch.tensor(L2).unsqueeze(0),
            torch.tensor(L3).unsqueeze(0),
        ]

    layers = clamp_layers_to_codebook(split_codes(codes), snac)
    audio = snac.decode(layers).detach().squeeze().cpu().numpy()
    return audio

# Simple lowpass filter to roll off high-end noise.
def lowpass(data, cutoff=9000, sr=24000):
    b, a = butter(4, cutoff / (0.5 * sr), btype="low")
    return filtfilt(b, a, data)

# Basic cleanup and noisereduce settings
def clean_audio(audio, sr=24000):
    """
    Basic cleanup:
      - replace NaNs/Infs
      - noise reduction using the first 0.3s as noise profile
      - lowpass filter
    """
    audio = np.nan_to_num(audio).astype("float32")
    noise_clip = audio[: int(0.3 * sr)]
    
    # Noisereduce settings
    audio = nr.reduce_noise(
        y=audio,
        y_noise=noise_clip,
        sr=sr,
        prop_decrease=0.4, # For raw CREMA-D speaker, keep prop_decrease high at ~0.85, for cleaner speakers can go lower. Anything higher than ~0.9 starts to degrade quality noticeably
        stationary=False,
        n_std_thresh_stationary=1.0
    )

    audio = lowpass(audio, cutoff=9000, sr=sr)
    return audio.astype("float32")

# Print seconds as 'Xm Ys' or 'Ys'.
def fmt_seconds(t):
    t = int(t)
    m, s = divmod(t, 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ========================= Combiner helpers =========================
# Normalise a clip to a target RMS and clamp to [-1, 1].
def normalize_audio(audio, target_rms=0.1):
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio
    gain = target_rms / rms
    audio = audio * gain
    return np.clip(audio, -1.0, 1.0).astype(np.float32)

# Extract numeric part from filename (out_001.wav -> 1).
def extract_index(filename):
    digits = "".join(c for c in filename if c.isdigit())
    return int(digits) if digits else 0

# Combine all wavs
def combine_wavs(folder, per_file_rms=0.1, global_rms=0.15, silence_ms=200):
    """
    Combine all wavs in a run folder:
      - sort by index
      - normalise each file
      - add small silence between
      - final global normalisation
    """
    wavs = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
    if not wavs:
        print(f"[combine] No WAV files in {folder}")
        return None, None

    wavs_sorted = sorted(wavs, key=extract_index)
    print(f"[combine] Found {len(wavs_sorted)} WAV files in {folder}")

    combined = []
    sr = None
    silence_seconds = silence_ms / 1000.0

    for idx, wavname in enumerate(wavs_sorted):
        path = os.path.join(folder, wavname)
        audio, samplerate = sf.read(path)
        audio = audio.astype(np.float32)

        if sr is None:
            sr = samplerate
            silence_block = np.zeros(int(silence_seconds * sr), dtype=np.float32)
        else:
            if samplerate != sr:
                raise ValueError(f"Sample rate mismatch in {wavname}")

        audio = normalize_audio(audio, per_file_rms)
        combined.append(audio)

        if idx < len(wavs_sorted) - 1:
            combined.append(silence_block)

    full_audio = np.concatenate(combined, axis=0)
    full_audio = normalize_audio(full_audio, global_rms)
    return full_audio, sr


# ========================= Main =========================

if __name__ == "__main__":
    total_start = time.perf_counter()

    # Load prompts.
    prompt_load_start = time.perf_counter()
    prompts = load_json_prompts(PROMPTS_JSON)
    prompt_load_time = time.perf_counter() - prompt_load_start

    if not prompts:
        print("No prompts found in prompts JSON.")
        raise SystemExit(1)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Create a new run folder: outputs/run01, run02, ...
    existing = [
        d for d in os.listdir(OUTPUT_ROOT)
        if d.startswith("run") and os.path.isdir(os.path.join(OUTPUT_ROOT, d))
    ]
    numbers = [int(d.replace("run", "")) for d in existing if d.replace("run", "").isdigit()]
    next_num = max(numbers, default=0) + 1
    run_dir = os.path.join(OUTPUT_ROOT, f"run{next_num:02d}")
    os.makedirs(run_dir, exist_ok=True)

    file_index = 0

    total_gen_time = 0.0
    total_clean_time = 0.0
    total_wav_time = 0.0

    for prompt in tqdm(prompts, desc="Generating", unit="line"):
        text = prompt.strip()
        if not text:
            continue

        file_index += 1
        out_path = os.path.join(run_dir, f"{BASE_PREFIX}{file_index:03d}.wav")

        # Generate raw audio.
        gen_start = time.perf_counter()
        audio = generate_audio(text)
        gen_time = time.perf_counter() - gen_start
        total_gen_time += gen_time

        # Cleanup.
        clean_start = time.perf_counter()
        audio = clean_audio(audio)
        clean_time = time.perf_counter() - clean_start
        total_clean_time += clean_time

        # Save wav.
        wav_start = time.perf_counter()
        sf.write(out_path, audio, 24000)
        wav_time = time.perf_counter() - wav_start
        total_wav_time += wav_time

    total_end = time.perf_counter()
    total_time = total_end - total_start

    measured = (
        snac_load_time +
        tok_load_time +
        model_load_time +
        prompt_load_time +
        total_gen_time +
        total_clean_time +
        total_wav_time
    )
    other_time = total_time - measured

    # Timing summary, used comparing unsloth vs raw hf but good to keep anyway.
    print("\n================ Timing Summary ================")
    print(f"SNAC load:               {fmt_seconds(snac_load_time)}")
    print(f"Tokeniser load:          {fmt_seconds(tok_load_time)}")
    print(f"Model load:              {fmt_seconds(model_load_time)}")
    print(f"Prompt load:             {fmt_seconds(prompt_load_time)}")
    print("-----------------------------------------------")
    print(f"Generation (total):      {fmt_seconds(total_gen_time)}")
    print(f"Cleanup (total):         {fmt_seconds(total_clean_time)}")
    print(f"WAV write (total):       {fmt_seconds(total_wav_time)}")
    print(f"Other:                   {fmt_seconds(other_time)}")
    print("-----------------------------------------------")
    print(f"TOTAL runtime:           {fmt_seconds(total_time)}")
    print("================================================\n")

    print(f"Done! Files written to {run_dir}/{BASE_PREFIX}###.wav")

    # ----- Combine the run into a single file -----
    os.makedirs(OUTPUT_COMBINED, exist_ok=True)

    # Extract run number from folder name (run08 -> "08").
    run_name = os.path.basename(run_dir)       # e.g. "run08"
    run_num_str = run_name.replace("run", "")  # e.g. "08"

    combined_audio, sr = combine_wavs(run_dir)
    if combined_audio is not None:
        combined_path = os.path.join(
            OUTPUT_COMBINED,
            f"combined_{run_num_str}.wav"
        )
        sf.write(combined_path, combined_audio, sr)
        print(f"[combine] Combined file written to {combined_path}")
    else:
        print("[combine] Skipped (no wavs to combine).")
