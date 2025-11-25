import os
import re
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import unicodedata

BaseModel = "roberta-base"
BaseDir = os.path.dirname(os.path.abspath(__file__))
ModelDir = os.path.join(BaseDir, "BGSFXTagger")
SfxListPath = os.path.join(BaseDir, "BackgroundSFX.txt")
BiasPath = os.path.join(BaseDir, "BackgroundSFX_bias.txt")

MaxTokens = 450
Device = "cuda" if torch.cuda.is_available() else "cpu"
DropoutP = 0.2

# Minimum (PB + PI) for a subtoken to be considered B/I-like.
PBI_MIN = 0.15


def ReadList(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def NormalizeText(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def StripNonSfxTags(t: str) -> str:
    """Remove any existing <<...>> tags to avoid double-tagging."""
    return re.sub(r"<<.*?>>", "", t)

def norm(s):
    return unicodedata.normalize("NFKC", s)

def LoadBiases(tags, bias_path):
    biases = {tag: 0.0 for tag in tags}

    print("\n[BGSFX] Exact tag names in SfxTags:")
    for t in tags:
        print("   ", repr(t))

    if not os.path.exists(bias_path):
        print(f"[BGSFX] Bias file not found: {bias_path} (using zero biases)")
        return biases

    print(f"\n[BGSFX] Loading biases from: {bias_path}")
    print("[BGSFX] Raw lines from bias file:")

    # Read file
    lines = []
    with open(bias_path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            print("   ", repr(line.rstrip("\n")))

    # Process
    for line in lines:
        if not line.strip():
            continue

        if "=" in line:
            name, val = line.split("=", 1)
        elif "\t" in line:
            name, val = line.split("\t", 1)
        elif "," in line:
            name, val = line.split(",", 1)
        else:
            print(f"[BGSFX WARNING] Could not parse bias line: {repr(line)}")
            continue


        name = name.strip()
        try:
            bias = float(val.strip())
        except ValueError:
            continue

        # Normalize both sides
        nname = norm(name)

        matched = False
        for tag in tags:
            if norm(tag) == nname:
                biases[tag] = bias
                matched = True
                break

        if not matched:
            print(f"[BGSFX WARNING] Could not match bias entry: {repr(name)}")

    print("\n[BGSFX] FINAL loaded biases:")
    for t in tags:
        print(f"  {t:35s} bias={biases[t]:.3f}")

    return biases


class BGTagger(nn.Module):
    def __init__(self, base, num_tags: int):
        super().__init__()
        self.Enc = AutoModel.from_pretrained(base)
        h = self.Enc.config.hidden_size
        self.Drop = nn.Dropout(DropoutP)
        self.Bio = nn.Linear(h, 3)   # B, I, O
        self.Cls = nn.Linear(h, num_tags)

    def forward(self, input_ids, attention_mask):
        h = self.Enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.Drop(h)
        return {"Bio": self.Bio(h), "Cls": self.Cls(h)}


SfxTags = ReadList(SfxListPath)
BiasTable = LoadBiases(SfxTags, BiasPath)

Tok = AutoTokenizer.from_pretrained(ModelDir)
Model = BGTagger(BaseModel, len(SfxTags))
State = torch.load(os.path.join(ModelDir, "model.bin"), map_location=Device)
Model.load_state_dict(State, strict=False)
Model.to(Device).eval()
print(f"[Loaded {len(SfxTags)} BGSFX tags from {SfxListPath}]")


def _tokens_to_words(tokens, bio_probs, cls_logits):
    words = []
    scores = []
    tag_ids = []

    cur_word = ""
    cur_pbi = []   # PB+PI for candidate subtokens
    cur_cls = []   # class indices for candidate subtokens

    for tok, bvec, clog in zip(tokens, bio_probs, cls_logits):
        if tok in ["<s>", "</s>", "<pad>"]:
            continue

        pb, pi = bvec[1], bvec[2]
        pbi = float(pb + pi)
        cid = int(np.argmax(clog))

        is_new_word = tok.startswith("Ġ")
        tok_text = tok[1:] if is_new_word else tok

        if is_new_word:
            # flush previous word
            if cur_word:
                if cur_pbi:
                    score = max(cur_pbi)
                    tag_id = int(np.bincount(cur_cls).argmax())
                else:
                    score = 0.0
                    tag_id = -1
                words.append(cur_word)
                scores.append(score)
                tag_ids.append(tag_id)

            # start new word
            cur_word = tok_text
            cur_pbi = []
            cur_cls = []
        else:
            cur_word += tok_text

        if pbi >= PBI_MIN:
            cur_pbi.append(pbi)
            cur_cls.append(cid)

    # flush last word
    if cur_word:
        if cur_pbi:
            score = max(cur_pbi)
            tag_id = int(np.bincount(cur_cls).argmax())
        else:
            score = 0.0
            tag_id = -1
        words.append(cur_word)
        scores.append(score)
        tag_ids.append(tag_id)

    return words, scores, tag_ids

def _style_to_rel_threshold(style: str) -> float:
    s = style.lower()
    if s == "subtle":
        return 0.7
    if s == "cinematic":
        return 0.3
    # Balanced / unknown
    return 0.5


def _build_spans(scores, rel_thr: float, max_gap: int = 1):
    if not scores:
        return [], None

    nonzero = [s for s in scores if s > 0.0]
    if not nonzero:
        # no candidate words at all (no B/I signal)
        return [], None

    max_score = max(nonzero)
    abs_floor = 0.20
    thr = max(max_score * rel_thr, abs_floor)

    spans = []
    in_span = False
    span_start = 0
    gap = 0

    for i, s in enumerate(scores):
        if s >= thr:
            if not in_span:
                in_span = True
                span_start = i
                gap = 0
            else:
                gap = 0
        else:
            if in_span:
                gap += 1
                if gap > max_gap:
                    span_end = i - gap
                    if span_end >= span_start:
                        spans.append((span_start, span_end))
                    in_span = False
    if in_span:
        span_end = len(scores) - 1
        spans.append((span_start, span_end))

    # If no spans survived but we had candidate words, choose the best candidate word.
    if not spans:
        j = int(np.argmax(scores))
        if scores[j] > 0.0:
            spans = [(j, j)]

    return spans, thr


def _choose_span_tag(span_start, span_end, tag_ids, scores, sfx_tags, bias_table):
    class_scores = {}

    for i in range(span_start, span_end + 1):
        cid = tag_ids[i]
        if cid < 0:
            continue
        if scores[i] <= 0.0:
            continue
        class_scores[cid] = class_scores.get(cid, 0.0) + 1.0

    if not class_scores:
        return "Unknown"

    # Apply bias
    for cid in list(class_scores.keys()):
        tag_name = sfx_tags[cid] if 0 <= cid < len(sfx_tags) else "Unknown"
        bias = bias_table.get(tag_name, 0.0)
        class_scores[cid] += bias

    # Pick best class
    best_cid = max(class_scores, key=class_scores.get)
    if 0 <= best_cid < len(sfx_tags):
        return sfx_tags[best_cid]
    return "Unknown"


def _render_with_spans(words, scores, tag_ids, spans, sfx_tags, bias_table):
    if not words:
        return ""

    if not spans:
        # No spans: just plain text
        txt = " ".join(words)
        txt = re.sub(r"\s+([.,!?;:])", r"\1", txt)
        txt = re.sub(r"\s{2,}", " ", txt).strip()
        return txt

    # Precompute tag per span
    span_tags = []
    for (st, en) in spans:
        tag = _choose_span_tag(st, en, tag_ids, scores, sfx_tags, bias_table)
        span_tags.append(tag)

    out = []
    span_idx = 0
    current_span = spans[span_idx] if spans else None

    for i, w in enumerate(words):
        if current_span and i == current_span[0]:
            tag = span_tags[span_idx]
            out.append(f"<<BGSFX Sound={tag}>> ")

        out.append(w + " ")

        if current_span and i == current_span[1]:
            out.append("<<\\BGSFX>> ")
            span_idx += 1
            current_span = spans[span_idx] if span_idx < len(spans) else None

    txt = "".join(out)
    txt = re.sub(r"\s+([.,!?;:])", r"\1", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def Reconstruct(enc_ids, attn_mask, out_logits, sfx_tags, style: str):
    bio = out_logits["Bio"].softmax(-1)[0].cpu().numpy()      # [T, 3]
    cls_logits = out_logits["Cls"][0].cpu().numpy()           # [T, num_tags]
    toks = Tok.convert_ids_to_tokens(enc_ids[0])

    words, scores, tag_ids = _tokens_to_words(toks, bio, cls_logits)

    if not any(s > 0.0 for s in scores):
        # no B/I evidence at all -> no tags
        plain = " ".join(words)
        plain = re.sub(r"\s+([.,!?;:])", r"\1", plain)
        plain = re.sub(r"\s{2,}", " ", plain).strip()
        return plain, 0, None

    rel_thr = _style_to_rel_threshold(style)
    spans, thr = _build_spans(scores, rel_thr)

    text_tagged = _render_with_spans(words, scores, tag_ids, spans, sfx_tags, BiasTable)
    span_count = len(spans)
    return text_tagged, span_count, thr

def ProcessSingleText(raw_text, SFXStyle: str = "Balanced"):
    txt = NormalizeText(StripNonSfxTags(raw_text))
    style = SFXStyle.lower().strip()

    if style == "none":
        return {
            "text_tagged": txt,
            "b_tags": 0,
            "threshold": None,
            "style": style,
        }

    # Encode text
    enc = Tok(txt, return_tensors="pt", truncation=True, max_length=MaxTokens).to(Device)
    with torch.no_grad():
        out = Model(enc["input_ids"], enc["attention_mask"])

    # Attempt 1 - normal threshold
    tagged, span_count, thr = Reconstruct(enc["input_ids"], enc["attention_mask"],
                                          out, SfxTags, style)

    if span_count > 0:
        return {
            "text_tagged": tagged,
            "b_tags": span_count,
            "threshold": thr,
            "style": style,
        }

    #  Attempt 1 - lower threshold
    global PBI_MIN
    old_pbi = PBI_MIN
    for new_pbi in [0.20, 0.15, 0.10, 0.05, 0.01]:
        PBI_MIN = new_pbi
        tagged, span_count, thr = Reconstruct(enc["input_ids"], enc["attention_mask"],
                                              out, SfxTags, style)
        if span_count > 0:
            PBI_MIN = old_pbi  # restore
            return {
                "text_tagged": tagged,
                "b_tags": span_count,
                "threshold": thr,
                "style": style,
                "pbi_used": new_pbi,
            }

    #  Attempt 2 - force span
    bio = out["Bio"].softmax(-1)[0].cpu().numpy()
    cls_logits = out["Cls"][0].cpu().numpy()
    toks = Tok.convert_ids_to_tokens(enc["input_ids"][0])
    words, scores, tag_ids = _tokens_to_words(toks, bio, cls_logits)

    if words:
        best = int(np.argmax(scores))
        tag_id = tag_ids[best] if tag_ids[best] >= 0 else 0
        tag = SfxTags[tag_id]

        fallback_txt = f"<<BGSFX Sound={tag}>> {words[best]} <<\\BGSFX>>"
        return {
            "text_tagged": fallback_txt,
            "b_tags": 1,
            "threshold": "FORCED-FALLBACK",
            "style": style,
        }

    #  Catch empty text
    return {
        "text_tagged": txt,
        "b_tags": 0,
        "threshold": None,
        "style": style,
        "fallback": "no-words",
    }


def ProcessTextArray(text_list, SFXStyle: str = "Balanced"):
    return [ProcessSingleText(t, SFXStyle=SFXStyle) for t in text_list]
