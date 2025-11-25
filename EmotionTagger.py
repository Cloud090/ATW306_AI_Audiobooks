# EmotionTagger.py
# Emotion tagging module with adjustable tone and word-safe tag placement

import os
import re
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

BaseModel = "roberta-base"
BaseDir = os.path.dirname(os.path.abspath(__file__))
ModelDir = os.path.join(BaseDir, "EmotionTagger")
EmotionsTxt = os.path.join(BaseDir, "Emotions.txt")
MaxTokens = 450
Device = "cuda" if torch.cuda.is_available() else "cpu"
DropoutP = 0.2
ThresholdDefault = 0.40
TagChangeDelta = 0.01

def ReadList(Path):
    with open(Path, "r", encoding="utf-8") as F:
        return [L.strip() for L in F if L.strip()]

def NormalizeText(S):
    return re.sub(r"\s+", " ", S).strip()

def StripNonEmotionTags(T):
    T = re.sub(r"<<\s*(BGSFX|FGSFX)[^>]*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\\\s*(BGSFX|FGSFX)\s*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\s*SFXSTYLE\s*=\s*[^>]+>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\s*PERSONA\s*=\s*[^>]+>>", "", T, flags=re.IGNORECASE)
    return T

def PreprocessInput(Raw):
    Txt = NormalizeText(StripNonEmotionTags(Raw))
    return Txt

class EmotionTagger(nn.Module):
    def __init__(self, Base, NumEmos):
        super().__init__()
        self.Enc = AutoModel.from_pretrained(Base)
        H = self.Enc.config.hidden_size
        self.Drop = nn.Dropout(DropoutP)
        self.Bio = nn.Linear(H, 3)
        self.Cls = nn.Linear(H, NumEmos)
        self.Inten = nn.Linear(H, 1)
    def forward(self, InputIds, AttentionMask):
        H = self.Enc(input_ids=InputIds, attention_mask=AttentionMask).last_hidden_state
        H = self.Drop(H)
        return {
            "Bio": self.Bio(H),
            "Cls": self.Cls(H),
            "Inten": torch.sigmoid(self.Inten(H)).squeeze(-1)
        }

Emos = ReadList(EmotionsTxt)
print(f"[Loaded {len(Emos)} emotions from {EmotionsTxt}]")

Tok = AutoTokenizer.from_pretrained(ModelDir)
Model = EmotionTagger(BaseModel, len(Emos))
State = torch.load(os.path.join(ModelDir, "model.bin"), map_location=Device)
Model.load_state_dict(State, strict=False)
Model.to(Device).eval()
print(f"[Model loaded successfully from {ModelDir}\\model.bin]")

def Reconstruct(EncIds, AttnMask, OutLogits, Emos, Threshold=ThresholdDefault, IntensityOffset=0):
    BioProbs = OutLogits["Bio"].softmax(-1)[0].cpu().numpy()
    ClsIds = OutLogits["Cls"].argmax(-1)[0].cpu().numpy()
    Intens = OutLogits["Inten"][0].cpu().numpy()
    Toks = Tok.convert_ids_to_tokens(EncIds[0])

    # Merge tokens into words
    Words, WordIndices = [], []
    current_word, current_indices = "", []
    for i, t in enumerate(Toks):
        if t in ["<s>", "</s>", "<pad>"]:
            continue
        if t.startswith("Ġ"):  # start of a new word
            if current_word:
                Words.append(current_word)
                WordIndices.append(current_indices)
            current_word = t[1:]
            current_indices = [i]
        else:
            current_word += t
            current_indices.append(i)
    if current_word:
        Words.append(current_word)
        WordIndices.append(current_indices)

    # Compute averaged probabilities per word
    WordPB, WordPI, WordEmo, WordInt = [], [], [], []
    for idxs in WordIndices:
        PBs = [BioProbs[i, 1] for i in idxs if i < BioProbs.shape[0]]
        PIs = [BioProbs[i, 2] for i in idxs if i < BioProbs.shape[0]]
        EmoIds = [ClsIds[i] for i in idxs if i < len(ClsIds)]
        Ints = [Intens[i] for i in idxs if i < len(Intens)]

        avg_PB, avg_PI = np.mean(PBs), np.mean(PIs)
        avg_Int = np.mean(Ints)
        emo_id = max(set(EmoIds), key=EmoIds.count)  # mode
        WordPB.append(avg_PB)
        WordPI.append(avg_PI)
        WordInt.append(avg_Int)
        WordEmo.append(Emos[emo_id] if 0 <= emo_id < len(Emos) else "NEUTRAL")

    # Build text with tag logic per word
    TextOut, PrevEmo, PrevInten, PrevPB = [], None, None, 0.0
    BCount, ICount = 0, 0

    for w, PB, PI, Emo, IntVal in zip(Words, WordPB, WordPI, WordEmo, WordInt):
        Intensity = int(max(1, min(9, round(IntVal * 8 + 1 + IntensityOffset))))

        if PB >= Threshold:
            should_change = False
            if PrevEmo is None:
                should_change = True
            elif Emo != PrevEmo:
                if abs(Intensity - (PrevInten or 0)) >= 2 or (PB - PrevPB) > TagChangeDelta:
                    should_change = True

            if should_change:
                if PrevEmo:
                    TextOut.append(f"<<\\{PrevEmo}>>")
                if Emo != "NEUTRAL":
                    TextOut.append(f"<<{Emo} Intensity={Intensity}>>")
                    PrevEmo, PrevInten, PrevPB = Emo, Intensity, PB
                    BCount += 1

        elif PI >= Threshold:
            ICount += 1

        TextOut.append(w + " ")

    if PrevEmo:
        TextOut.append(f"<<\\{PrevEmo}>>")

    Txt = "".join(TextOut)
    Txt = re.sub(r"\s+([.,!?;:])", r"\1", Txt)
    Txt = re.sub(r"\s{2,}", " ", Txt).strip()
    Txt = re.sub(r"<<Neutral Intensity=\d+>>", "", Txt)
    Txt = re.sub(r"<<\\Neutral>>", "", Txt)
    return Txt, BCount, ICount


def ProcessSingleText(RawText, Tone="Neutral"):
    Txt = PreprocessInput(RawText)

    # Tone adjustments
    Tone = Tone.lower().strip()
    if Tone == "unemotional":
        print("[Mode: Unemotional] Returning unedited text.")
        return {"text_tagged": Txt, "b_tags": 0, "i_tags": 0, "threshold": None}
    elif Tone == "calm":
        threshold = ThresholdDefault + 0.10
        intensity_offset = -2
    elif Tone == "dramatic":
        threshold = ThresholdDefault - 0.10
        intensity_offset = +2
    else:  # Neutral
        threshold = ThresholdDefault
        intensity_offset = 0

    Enc = Tok(Txt, return_tensors="pt", truncation=True, max_length=MaxTokens).to(Device)

    with torch.no_grad():
        Outputs = Model(Enc["input_ids"], Enc["attention_mask"])

    Text, BCount, ICount = Reconstruct(
        Enc["input_ids"], Enc["attention_mask"], Outputs, Emos,
        Threshold=threshold, IntensityOffset=intensity_offset
    )

    return {
        "text_tagged": Text.strip(),
        "b_tags": BCount,
        "i_tags": ICount,
        "threshold": threshold,
        "tone": Tone
    }


def ProcessTextArray(TextList, Tone="Neutral"):
    Results = []
    for Txt in TextList:
        Results.append(ProcessSingleText(Txt, Tone=Tone))
    return Results
