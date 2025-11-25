import os
import re
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

BaseModel = "roberta-base"
BaseDir = os.path.dirname(os.path.abspath(__file__))
ModelDir = os.path.join(BaseDir, "FGSFXTagger")
SfxListPath = os.path.join(BaseDir, "ForegroundSFX.txt")
MaxTokens = 450
Device = "cuda" if torch.cuda.is_available() else "cpu"
DropoutP = 0.2
ThresholdDefault = 0.40
TagChangeDelta = 0.15

def ReadList(Path):
    with open(Path, "r", encoding="utf-8") as F:
        return [L.strip() for L in F if L.strip()]

def NormalizeText(S):
    return re.sub(r"\s+", " ", S).strip()

def StripNonSfxTags(T):
    T = re.sub(r"<<\s*(HAPPY|SAD|FEAR|ANGRY|DISGUST)[^>]*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\\\s*(HAPPY|SAD|FEAR|ANGRY|DISGUST)\s*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\s*BGSFX[^>]*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\\\s*(BGSFX)\s*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\s*SFXSTYLE\s*=\s*[^>]+>>", "", T, flags=re.IGNORECASE)
    return T

def PreprocessInput(Raw):
    Txt = NormalizeText(StripNonSfxTags(Raw))
    return Txt

class FGTagger(nn.Module):
    def __init__(self, Base, NumTags):
        super().__init__()
        self.Enc = AutoModel.from_pretrained(Base)
        H = self.Enc.config.hidden_size
        self.Drop = nn.Dropout(DropoutP)
        self.Bio = nn.Linear(H, 3)
        self.Cls = nn.Linear(H, NumTags)
    def forward(self, InputIds, AttentionMask):
        H = self.Enc(input_ids=InputIds, attention_mask=AttentionMask).last_hidden_state
        H = self.Drop(H)
        return {"Bio": self.Bio(H), "Cls": self.Cls(H)}

SfxTags = ReadList(SfxListPath)
print(f"[Loaded {len(SfxTags)} FGSFX tags from {SfxListPath}]")

Tok = AutoTokenizer.from_pretrained(ModelDir)
Model = FGTagger(BaseModel, len(SfxTags))
State = torch.load(os.path.join(ModelDir, "model.bin"), map_location=Device)
Model.load_state_dict(State, strict=False)
Model.to(Device).eval()
print(f"[Model loaded successfully from {ModelDir}\\model.bin]")

def Reconstruct(EncIds, AttnMask, OutLogits, SfxTags, Threshold=ThresholdDefault):
    BioProbs = OutLogits["Bio"].softmax(-1)[0].cpu().numpy()
    ClsIds = OutLogits["Cls"].argmax(-1)[0].cpu().numpy()

    Toks = Tok.convert_ids_to_tokens(EncIds[0])
    Words = [T for T in Toks if T not in ["<s>", "</s>", "<pad>"]]
    TextOut, PrevTag, PrevPB = [], None, 0.0
    BCount, ICount = 0, 0

    for I, T in enumerate(Words):
        TokTxt = T.replace("Ġ", " ")
        PB = BioProbs[I, 1] if I < BioProbs.shape[0] else 0.0
        PI = BioProbs[I, 2] if I < BioProbs.shape[0] else 0.0
        Tag = SfxTags[ClsIds[I]] if 0 <= ClsIds[I] < len(SfxTags) else "Unknown"

        if PB >= Threshold:
            should_change = False
            if PrevTag is None:
                should_change = True
            elif Tag != PrevTag and (PB - PrevPB) > TagChangeDelta:
                should_change = True

            if should_change:
                TextOut.append(f"<<FGSFX Sound={Tag}>>")
                PrevTag, PrevPB = Tag, PB
                BCount += 1

        elif PI >= Threshold:
            ICount += 1

        TextOut.append(TokTxt)

    if (BCount + ICount) == 0:
        Bi = np.maximum(BioProbs[:len(Words), 1], BioProbs[:len(Words), 2])
        J = int(np.argmax(Bi))
        Tag = SfxTags[ClsIds[J]] if 0 <= ClsIds[J] < len(SfxTags) else "Unknown"
        Words[J] = f"<<FGSFX Sound={Tag}>>{Words[J].replace('Ġ', ' ')}"
        TextOut = ["".join(Words)]
        BCount, ICount = 1, 0

    Txt = "".join(TextOut)
    Txt = re.sub(r"\s+([.,!?;:])", r"\1", Txt)
    Txt = re.sub(r"\s{2,}", " ", Txt).strip()
    Txt = re.split(r"={10,}\s*Input\s*:", Txt, flags=re.IGNORECASE)[0].strip()

    return Txt, BCount, ICount

def ProcessSingleText(RawText, SFXStyle="Balanced"):
    Txt = PreprocessInput(RawText)

    # Style adjustments
    SFXStyle = SFXStyle.lower().strip()
    if SFXStyle == "none":
        print("[SFXStyle: None] Returning unedited text.")
        return {"text_tagged": Txt, "b_tags": 0, "i_tags": 0, "threshold": None}
    elif SFXStyle == "subtle":
        threshold = ThresholdDefault + 0.10
    elif SFXStyle == "cinematic":
        threshold = ThresholdDefault - 0.10
    else:
        threshold = ThresholdDefault

    Enc = Tok(Txt, return_tensors="pt", truncation=True, max_length=MaxTokens).to(Device)

    with torch.no_grad():
        Outputs = Model(Enc["input_ids"], Enc["attention_mask"])

    Text, BCount, ICount = Reconstruct(Enc["input_ids"], Enc["attention_mask"], Outputs, SfxTags, Threshold=threshold)
    TxtOut = Text.replace("Ġ", " ").strip()

    return {
        "text_tagged": TxtOut,
        "b_tags": BCount,
        "i_tags": ICount,
        "threshold": threshold,
        "style": SFXStyle
    }

def ProcessTextArray(TextList, SFXStyle="Balanced"):
    Results = []
    for Txt in TextList:
        Results.append(ProcessSingleText(Txt, SFXStyle=SFXStyle))
    return Results
