# BGSFX_Test_v12_OffsetSafe.py
# -----------------------------
# Offset-based prefix trimming (no lost first word), threshold debug,
# tag-change dampening. No intensity head assumed for BGSFX.

import os, re, torch, numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ---------- SETTINGS ----------
BaseModel = "roberta-base"
BaseDir = os.path.dirname(os.path.abspath(__file__))
ModelDir = os.path.join(BaseDir, "bgsfx_tagger_v3", "best")
SfxListPath = os.path.join(BaseDir, "BackgroundSFX.txt")

MaxTokens = 450
Device = "cuda" if torch.cuda.is_available() else "cpu"
DropoutP = 0.2
ThresholdDefault = 0.40
TagChangeDelta = 0.15   # minimum jump in B-prob to allow switching tags

# ---------- HELPERS ----------
def ReadList(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def Normalize(s): return re.sub(r"\s+", " ", s).strip()

def ExtractSfxStyle(t):
    m = re.search(r"<<\s*SFXSTYLE\s*=\s*([^>]+)>>", t, re.I)
    return m.group(1).strip() if m else "Neutral"

def StripTags(t):
    # Remove SFXSTYLE (kept only for prefix), PERSONA/FGSFX, and any emotion tags
    t = re.sub(r"<<\s*SFXSTYLE\s*=\s*[^>]+>>", "", t, flags=re.I)
    t = re.sub(r"<<\s*(PERSONA|FGSFX)[^>]*>>", "", t, flags=re.I)
    t = re.sub(r"<<\\\s*(PERSONA|FGSFX)\s*>>", "", t, flags=re.I)
    t = re.sub(r"<<\s*[A-Z]{2,}\s*(?:Intensity\s*=\s*\d+)?\s*>>", "", t)
    t = re.sub(r"<<\\\s*[A-Z]{2,}\s*>>", "", t)
    return t

def Preprocess(raw):
    style = ExtractSfxStyle(raw)
    txt = Normalize(StripTags(raw))
    prefix = f"SFXStyle: {style}. Story: "
    return prefix + txt, style, prefix

# ---------- MODEL ----------
class BGSFXTagger(nn.Module):
    def __init__(self, base, n):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(DropoutP)
        self.bio = nn.Linear(h, 3)
        self.cls = nn.Linear(h, n)
    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        h = self.drop(h)
        return {"Bio": self.bio(h), "Cls": self.cls(h)}

# ---------- LOAD ----------
SfxList = ReadList(SfxListPath)
Tok = AutoTokenizer.from_pretrained(ModelDir)
Model = BGSFXTagger(BaseModel, len(SfxList))
State = torch.load(os.path.join(ModelDir, "model.bin"), map_location=Device)
Model.load_state_dict(State, strict=False)
Model.to(Device).eval()
print(f"[Loaded model from {ModelDir}] ({len(SfxList)} SFX types)")

# ---------- RECONSTRUCT ----------
def Reconstruct(enc_ids, attn, offs, outs, thr=ThresholdDefault, prefix_char_len=0):
    """
    offs: token offsets from tokenizer (list of (start,end))
    We skip any token whose end <= prefix_char_len (i.e., inside the prefix).
    """
    bio = outs["Bio"].softmax(-1)[0].cpu().numpy()            # [T,3]
    cls_probs = outs["Cls"].softmax(-1)[0].cpu().numpy()      # [T,K]
    toks = Tok.convert_ids_to_tokens(enc_ids[0])

    # Build parallel arrays for kept tokens (exclude specials)
    kept = []
    for i, t in enumerate(toks):
        if t in ("<s>", "</s>", "<pad>"):
            continue
        kept.append((i, t))

    text_out = []
    prev_sfx, prev_pb = None, 0.0
    bcount = 0

    for orig_i, t in kept:
        start, end = offs[orig_i]
        tok_txt = t.replace("Ġ", " ")

        # skip tokens that are part of the prefix string
        if end <= prefix_char_len:
            continue

        # BIO start prob and best SFX class
        pb = bio[orig_i, 1] if orig_i < bio.shape[0] else 0.0
        best_cls = int(np.argmax(cls_probs[orig_i])) if orig_i < cls_probs.shape[0] else -1
        sfx = SfxList[best_cls] if 0 <= best_cls < len(SfxList) else "UNKNOWN"

        # tag-change dampening
        should_change = False
        if pb >= thr:
            if prev_sfx is None:
                should_change = True
            elif sfx != prev_sfx and (pb - prev_pb) > TagChangeDelta:
                should_change = True

            if should_change:
                if prev_sfx:
                    text_out.append("<<\\BGSFX>>")
                text_out.append(f"<<BGSFX={sfx}>>")
                prev_sfx, prev_pb = sfx, pb
                bcount += 1

        text_out.append(tok_txt)

    if prev_sfx:
        text_out.append("<<\\BGSFX>>")

    txt = "".join(text_out)
    txt = re.sub(r"\s+([.,!?;:])", r"\1", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip(), bcount

# ---------- PROCESS ----------
def Process(text):
    full, style, prefix = Preprocess(text)
    print(f"[Processing Style={style}]")

    enc = Tok(full, return_tensors="pt", truncation=True, max_length=MaxTokens, return_offsets_mapping=True)
    # offsets as a list of (start,end) pairs for the full string
    offsets = enc["offset_mapping"][0].tolist()
    # char length of prefix segment
    prefix_char_len = len(prefix)

    enc = {k: v.to(Device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "offset_mapping")}
    thresholds_to_try = [ThresholdDefault, 0.35, 0.30, 0.25, 0.20, 0.15]
    txt, b, final_thr = "", 0, 0.0

    for thr in thresholds_to_try:
        print(f"[Attempt] threshold={thr:.2f}")
        with torch.no_grad():
            outs = Model(enc["input_ids"], enc["attention_mask"])
        txt, b = Reconstruct(enc["input_ids"], enc["attention_mask"], offsets, outs,
                             thr=thr, prefix_char_len=prefix_char_len)
        if b > 0:
            final_thr = thr
            print(f"[Success] Tags found at threshold={thr:.2f} (B={b})")
            break
        else:
            print(f"[Retry] No tags found at threshold={thr:.2f}")

    if b == 0:
        print("[WARN] No tags found at any threshold; forcing best guess.")
        with torch.no_grad():
            outs = Model(enc["input_ids"], enc["attention_mask"])
        txt, _ = Reconstruct(enc["input_ids"], enc["attention_mask"], offsets, outs,
                             thr=0.0, prefix_char_len=prefix_char_len)
        final_thr = 0.0

    # Safety: strip any lingering prefix phrase if somehow emitted
    txt = re.sub(r"^\s*SFXStyle\s*:\s*[^.]+\.\s*Story\s*:\s*", "", txt, flags=re.I)
    print(f"[Result] Style={style} | Tags={b} | Threshold={final_thr:.2f}")
    return txt

# ---------- MAIN ----------
if __name__ == "__main__":
    sample = """<<SFXSTYLE = Cinematic>>
The rain hammered against the steel rooftops of New Arcadia, each droplet lost in a chorus of thunder.
Beneath the neon haze, Detective Lorne stepped over puddles reflecting signs for clubs long since closed.
The air reeked of ozone and engine fumes, mingling with the faint hum of distant turbines.
From the alleyway came the muffled hum of a power generator, steady and cold,
its rhythmic buzz the heartbeat of a dying city.

He reached the door — rusted, dented, half-swallowed by graffiti — and pushed it open.
The hinges screamed, metal scraping against itself like a wounded thing.
Inside, the smell of burnt circuitry mingled with cigarette smoke.
The lights flickered to life reluctantly, revealing a workshop frozen mid-project:
mechanical arms half-assembled, wires tangled like vines.
A single terminal blinked at the far end of the room, its fan groaning against dust.
Lorne pulled his coat tighter and exhaled; the sound was lost beneath the distant thrum of rain."""
    print("\n[INPUT]\n", sample)
    out = Process(sample)
    print("\n[OUTPUT]\n", out)
