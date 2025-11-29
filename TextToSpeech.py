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

TOKENISER_LEN = 128256
START_HUMAN = TOKENISER_LEN + 3
END_HUMAN = TOKENISER_LEN + 4
END_OF_TEXT = 128009
EOS_SPEECH = 128258
START_SPEECH = TOKENISER_LEN + 1

SR = 24000  # sample rate

_MODEL = None
_TOKENIZER = None
_SNAC = None

def _resolve_checkpoint(local_path, hf_repo):
    config = os.path.join(local_path, "config.json")
    if os.path.isfile(config):
        if any(f.endswith(".safetensors") for f in os.listdir(local_path)):
            print(f"[Tagged text to speech] Using local model: {local_path}")
            return local_path
    print(f"[Tagged text to speech] Using HF model: {hf_repo}")
    return hf_repo


def _lowpass(a, cutoff=9000):
    b, c = butter(4, cutoff / (0.5 * SR), btype="low")
    return filtfilt(b, c, a)


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


def _decode_snac(snac, codes):
    L1, L2, L3 = [], [], []

    for i in range((len(codes) + 1) // 7):
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

    audio = snac.decode(layers).detach().squeeze().cpu().numpy()
    return audio


def _normalize(a, rms=0.1):
    a = a.astype(np.float32)
    power = np.sqrt(np.mean(a ** 2))
    if power < 1e-8:
        return a
    return np.clip(a * (rms / power), -1.0, 1.0).astype(np.float32)


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

def preload_model(
    local_model_path="models/orpheus_merged_cremad_full_fp16_mspk",
    hf_repo_id="Smallan/orpheus_merged_cremad_full_fp16_mspk"
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


def _ensure_loaded():
    """Auto-load model if not already loaded."""
    if _MODEL is None:
        preload_model()


def speak_text(text):
    _ensure_loaded()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ids = _TOKENIZER(text, return_tensors="pt").input_ids.to(device)

    start = torch.tensor([[START_HUMAN]], device=device)
    end = torch.tensor([[END_OF_TEXT, END_HUMAN]], device=device)
    inp = torch.cat([start, ids, end], dim=1)
    attn = torch.ones_like(inp)

    gen = _MODEL.generate(
        input_ids=inp,
        attention_mask=attn,
        max_new_tokens=5000,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=EOS_SPEECH
    )[0]

    # Find speech start
    idx = (gen == START_SPEECH).nonzero(as_tuple=True)[0]
    if len(idx):
        gen = gen[idx[-1] + 1:]

    gen = gen[gen != EOS_SPEECH]

    usable = (len(gen) // 7) * 7
    tokens = [int(x) - 128266 for x in gen[:usable]]

    raw = _decode_snac(_SNAC, tokens)
    return _clean_audio(raw)


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

        # Generate audio
        audio = speak_text(t)

        # Save
        sf.write(out_path, audio, SR)

        if progress:
            progress(f"Saved segment {index}/{total}")

    # Combine all WAVs
    combined = _combine_wavs(output_folder)

    if progress:
        progress("Combining all segments into final audio...")

    if progress:
        progress(f"Orpheus completed. Output folder: {output_folder}")

    return output_folder