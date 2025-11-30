# This module runs the Background SFX tagging model, a fine-tined RoBERTa-based model designed to identify context-based positions in the provided text where
# a background sound could be added. 

# The model returns Begin/Inside/Outside probabilities for each token, which can be used to create spans, and can then be mapped into the most likely sound effect from
# a predefined list (loaded from "BackgroundSFX.txt"). These can then be converted into tags, in the format "<<BGSFX Sound=[Sound]>>" & "<<\BGSFX>>".

import os
import re
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import unicodedata

# Set up base model name, and locations of needed files.
BaseModel = "roberta-base"
BaseDir = os.path.dirname(os.path.abspath(__file__))
ModelDir = os.path.join(BaseDir, "BGSFXTagger")
SfxListPath = os.path.join(BaseDir, "BackgroundSFX.txt")
BiasPath = os.path.join(BaseDir, "BackgroundSFX_bias.txt")

MaxTokens = 450
Device = "cuda" if torch.cuda.is_available() else "cpu"
DropoutP = 0.2

# Minimum probability (PB + PI) for a token to be considered part of a sound effect span.
PBI_MIN = 0.15

# Function to read a text file, returning a list of lines.
def ReadList(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

# Function to clean text - collapses multiple spaces into single space, and trims white space from front and end of text.
def NormalizeText(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# Function to get rid of any preexisting tags.
def StripNonSfxTags(t: str) -> str:
    return re.sub(r"<<.*?>>", "", t)

# Function to get rid of unicode characters.
def norm(s):
    return unicodedata.normalize("NFKC", s)

# The model favours a "Distant Sirens" sound effect - this function loads a list of biases for each sound effect that can be applied, in order to try to counteract this.
def LoadBiases(tags, bias_path):
    biases = {tag: 0.0 for tag in tags}

    print("\n[BGSFX] Exact tag names in SfxTags:")
    for t in tags:
        print("   ", repr(t))

    if not os.path.exists(bias_path):
        print(f"[BGSFX] Bias file not found: {bias_path} (not using biases)")
        return biases

    print(f"\n[BGSFX] Loading biases from: {bias_path}")
    print("[BGSFX] Raw lines from bias file:")

    # Read file
    lines = []
    with open(bias_path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            print("   ", repr(line.rstrip("\n")))

    # Process each line
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

        # Match each bias to a sound effect from the background sound effect list.
        nname = norm(name)

        matched = False
        for tag in tags:
            if norm(tag) == nname:
                biases[tag] = bias
                matched = True
                break

        if not matched:
            print(f"[BGSFX WARNING] Could not match bias entry: {repr(name)}")

    # Outputs and returns the bias list.
    print("\n[BGSFX] FINAL loaded biases:")
    for t in tags:
        print(f"  {t:35s} bias={biases[t]:.3f}")

    return biases

# Sets up a class for the BGSFX model.
class BGTagger(nn.Module):
    def __init__(self, base, num_tags: int):
        super().__init__()

        # Load the pretrained model encoder
        self.Enc = AutoModel.from_pretrained(base)

        # Hidden dimension size of the encoder output (needed for classifying)
        h = self.Enc.config.hidden_size

        # Apply a dropout (used in training, so remains for inference)
        self.Drop = nn.Dropout(DropoutP)

        # Layer to predict beginning, inside, and outside tags, to create spans
        self.Bio = nn.Linear(h, 3)

        # Layer to predict sound effect classification
        self.Cls = nn.Linear(h, num_tags)

    # Function to run a forward pass through the model
    def forward(self, input_ids, attention_mask):
        # Run input through the encoder
        h = self.Enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Apply the dropout
        h = self.Drop(h)

        # Return lists for each token - "beginning, inside, and outside" probabilities, and classifier probabilities.
        return {"Bio": self.Bio(h), "Cls": self.Cls(h)}

# Load sound effect list, bias table.
SfxTags = ReadList(SfxListPath)
BiasTable = LoadBiases(SfxTags, BiasPath)

# Loads the tokeniser saved with the trained model.
Tok = AutoTokenizer.from_pretrained(ModelDir)

# Loads the trained model architecture, using the above class.
Model = BGTagger(BaseModel, len(SfxTags))

# Loads the trained weights
State = torch.load(os.path.join(ModelDir, "model.bin"), map_location=Device)

# Loads the weights into the model.
Model.load_state_dict(State, strict=False)

# Moves the model to the appropriate device (should be GPU)
Model.to(Device).eval()

print(f"[Loaded {len(SfxTags)} BGSFX tags from {SfxListPath}]")

# Converts tokens back into words, collecting the calculated "beginning, inside, and outside" probabilities, and classifier probabilities. This ensures that created spans will not start in
# the middle of a word.
def _tokens_to_words(tokens, bio_probs, cls_logits):
    words = []
    scores = []
    tag_ids = []

    cur_word = ""
    cur_pbi = []
    cur_cls = []

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

# This takes the "SFX style" provided, and returns a threshold. Subtle (less sound effects) returns a higher threshold of 0.7, Cinematic (more sound effects) returns a lower threshold
# of 0.3, and Balanced (average amount of sound effects) or unknown values return a threshold of 0.5. This threshold is used to determine how likely a sound effect is to be placed.
def _style_to_rel_threshold(style: str) -> float:
    s = style.lower()
    if s == "subtle":
        return 0.7
    if s == "cinematic":
        return 0.3
    # Balanced / unknown
    return 0.5

# This function takes the list of per-word scores returned from the model, along with a threshold, and identifies where the scores are high enough to justify adding a sound effect. It
# returns a list of start and end spans, along with the final threshold used (as threshold may be reduced if no tag locations are initially found).
def _build_spans(scores, rel_thr: float, max_gap: int = 1):
    # If no scores exist, returns empty list.
    if not scores:
        return [], None

    # Filter out scores of 0. If no nonzero scores exist, return an empty list.
    nonzero = [s for s in scores if s > 0.0]
    if not nonzero:
        # no candidate words at all (no B/I signal)
        return [], None

    # Sets the tagging threshold, defined as either the maximum score times the style threshold, or 0.2, whichever is higher.
    max_score = max(nonzero)
    abs_floor = 0.20
    thr = max(max_score * rel_thr, abs_floor)

    # Prepare variables to define a sound effect span.
    spans = []
    in_span = False
    span_start = 0
    gap = 0

    # Go through each word, deciding if it belongs in a span.
    for i, s in enumerate(scores):
        # Checks if the score for word i is above the threshold. 
        # - If yes, and it's not currently a span, starts one. 
        # - If yes, and it's currently in a span, continues
        # - If no, and it's currently in a span, checks the current "gap" (how many low-scores words we allow in a span before ending). If it's above the maximum, ends the span.
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
                        # If a span ends, append it to the list.
                        spans.append((span_start, span_end))
                    in_span = False

    # If we're at the last word and still in a span, ends it.
    if in_span:
        span_end = len(scores) - 1
        spans.append((span_start, span_end))

    # If no spans were found but we had candidate words, choose the best candidate word.
    if not spans:
        j = int(np.argmax(scores))
        if scores[j] > 0.0:
            spans = [(j, j)]

    return spans, thr

# This function looks at the classifier scores of the words inside a span, counts how many times each sound effect is predicted, applies a bias, then selects the most likely sound effect
# for the span.
def _choose_span_tag(span_start, span_end, tag_ids, scores, sfx_tags, bias_table):
    # Sets up dictionary for the classifier scores.
    class_scores = {}

    for i in range(span_start, span_end + 1):
        # Go through every word in the span, skipping ones with no prediction or a score of 0. Counts each qualifying score, and adds it to the dictionary.
        cid = tag_ids[i]
        if cid < 0:
            continue
        if scores[i] <= 0.0:
            continue
        class_scores[cid] = class_scores.get(cid, 0.0) + 1.0

    # Returns "Unknown" if no class qualifies.
    if not class_scores:
        return "Unknown"

    # Apply a bias adjustment for each detected entry.
    for cid in list(class_scores.keys()):
        tag_name = sfx_tags[cid] if 0 <= cid < len(sfx_tags) else "Unknown"
        bias = bias_table.get(tag_name, 0.0)
        class_scores[cid] += bias

    # Pick the class with the highest count.
    best_cid = max(class_scores, key=class_scores.get)

    # Returns the best class if valid.
    if 0 <= best_cid < len(sfx_tags):
        return sfx_tags[best_cid]

    # Returns "Unknown" if no valid classes.
    return "Unknown"

# This function takes the word list and the detected span list, and returns the text with the generated BGSFX tags.
def _render_with_spans(words, scores, tag_ids, spans, sfx_tags, bias_table):
    # If no words are in the list, returns an empty string.
    if not words:
        return ""

    # If no spans are in the list, returns just the text.
    if not spans:
        txt = " ".join(words)
        txt = re.sub(r"\s+([.,!?;:])", r"\1", txt)
        txt = re.sub(r"\s{2,}", " ", txt).strip()
        return txt

    # For each detected span, uses the "_choose_span_tag" function to determine which sound effect to associate.
    span_tags = []
    for (st, en) in spans:
        tag = _choose_span_tag(st, en, tag_ids, scores, sfx_tags, bias_table)
        span_tags.append(tag)

    # Variables for use in building the output text.
    out = []
    span_idx = 0
    current_span = spans[span_idx] if spans else None

    # Goes through each word in the list. 
    # 1. If the word is the start of a span, appends initial "<<BGSFX Sound=[sound effect]>>" tag to the "out" list.
    # 2. It then appends the current word to the "out" list, along with a space.
    # 3. Finally, if it's the end of a span, appends "<<\\BGSFX>>" tag. 
    for i, w in enumerate(words):
        if current_span and i == current_span[0]:
            tag = span_tags[span_idx]
            out.append(f"<<BGSFX Sound={tag}>> ")

        out.append(w + " ")

        if current_span and i == current_span[1]:
            out.append("<<\\BGSFX>> ")
            span_idx += 1
            current_span = spans[span_idx] if span_idx < len(spans) else None

    # Converts the "out" list back to a single string. Cleans up punctuation, and returns the final string.
    txt = "".join(out)
    txt = re.sub(r"\s+([.,!?;:])", r"\1", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

# This function reconstructs the final tagged text from the model output. 
def Reconstruct(enc_ids, attn_mask, out_logits, sfx_tags, style: str):
    # Converts the ""beginning, inside, and outside" results into a probabilities list
    bio = out_logits["Bio"].softmax(-1)[0].cpu().numpy()

    # Converts the classifier results into a probabilities list.
    cls_logits = out_logits["Cls"][0].cpu().numpy()

    # Convert token IDs back to text-based tokens.
    toks = Tok.convert_ids_to_tokens(enc_ids[0])

    # Gets word list, along with associated scores and classes lists
    words, scores, tag_ids = _tokens_to_words(toks, bio, cls_logits)

    # If no words have a span possibility, returns plain text with no tags.
    if not any(s > 0.0 for s in scores):
        plain = " ".join(words)
        plain = re.sub(r"\s+([.,!?;:])", r"\1", plain)
        plain = re.sub(r"\s{2,}", " ", plain).strip()
        return plain, 0, None

    # Gets the relative threshold, dependant on the SFX style provided.
    rel_thr = _style_to_rel_threshold(style)

    # Builds a span list from the scores, using the relative threshold.
    spans, thr = _build_spans(scores, rel_thr)

    # Builds the tagged text.
    text_tagged = _render_with_spans(words, scores, tag_ids, spans, sfx_tags, BiasTable)

    # Returns the tagged text, number of spans, and threshold used.
    return text_tagged, len(spans), thr

# This function runs the BGSFX tagging process for a single text string. It tokenises the text, runs the model, builds the resulting spans, then returns the tagged text, along
# with some metadata.
def ProcessSingleText(raw_text, SFXStyle: str = "Balanced"):
    # Normalises the text - collapses multiple spaces into a single space, trims whitespace from the start and end.
    txt = NormalizeText(StripNonSfxTags(raw_text))

    # Gets the passed SFX style, lower case, with whitespace removed.
    style = SFXStyle.lower().strip()

    # If the style is "none", returns the text without adding tags.
    if style == "none":
        return {
            "text_tagged": txt,
            "b_tags": 0,
            "threshold": None,
            "style": style,
        }

    # Encode text using the tokenizer
    enc = Tok(txt, return_tensors="pt", truncation=True, max_length=MaxTokens).to(Device)

    # Runs the model, saving the output
    with torch.no_grad():
        out = Model(enc["input_ids"], enc["attention_mask"])

    # Attempts to build the tagged text using default settings.
    tagged, span_count, thr = Reconstruct(enc["input_ids"], enc["attention_mask"], out, SfxTags, style)

    # If at least one span was generated, return the result.
    if span_count > 0:
        return {
            "text_tagged": tagged,
            "b_tags": span_count,
            "threshold": thr,
            "style": style,
        }

    #  If no spans were found, attempts it again with lower thresholds of 0.20, 0.15, 0.10, 0.05, and 0.01 (Note: PBI_MIN is used internally in several above functions - reducing it 
    # makes spans more likely.
    global PBI_MIN
    old_pbi = PBI_MIN
    for new_pbi in [0.20, 0.15, 0.10, 0.05, 0.01]:
        PBI_MIN = new_pbi
        tagged, span_count, thr = Reconstruct(enc["input_ids"], enc["attention_mask"], out, SfxTags, style)
        if span_count > 0:
            PBI_MIN = old_pbi
            return {
                "text_tagged": tagged,
                "b_tags": span_count,
                "threshold": thr,
                "style": style,
                "pbi_used": new_pbi,
            }

    # Restore original PBI_MIN after all attempts.
    PBI_MIN = old_pbi

    # If no spans were found, returns untagged text.
    return {
        "text_tagged": txt,
        "b_tags": 0,
        "threshold": None,
        "style": style,
        "fallback": "no-words",
    }

# Function that processes text list, by running each item in the list through the "ProcessSingleText" function.
def ProcessTextArray(text_list, SFXStyle: str = "Balanced"):
    return [ProcessSingleText(t, SFXStyle=SFXStyle) for t in text_list]
