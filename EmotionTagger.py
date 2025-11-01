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
TagChangeDelta = 0.15    # minimum jump to switch emotion tags (solves multiple changing tags)

def ReadList(Path):
    with open(Path, "r", encoding="utf-8") as F:
        return [L.strip() for L in F if L.strip()]


def NormalizeText(S):
    return re.sub(r"\s+", " ", S).strip()


PersonaTags = [
    re.compile(r"<<\s*PERSONA\s*=\s*([^>]+)>>", re.IGNORECASE),
    re.compile(r"<<\s*PERSONA:?\s*([^>]+)>>", re.IGNORECASE)
]
SfxStyleTag = re.compile(r"<<\s*SFXSTYLE\s*=\s*[^>]+>>", re.IGNORECASE)


def ExtractPersona(Text):
    for Rx in PersonaTags:
        M = Rx.search(Text)
        if M:
            persona = M.group(1).strip()
            print(f"[Persona Detected] {persona}")
            return persona
    print("[Persona Not Detected] Using 'Unknown'")
    return "Unknown"


def StripNonEmotionTags(T):
    T = PersonaTags[0].sub("", T)
    T = PersonaTags[1].sub("", T)
    T = SfxStyleTag.sub("", T)
    T = re.sub(r"<<\s*(BGSFX|FGSFX)[^>]*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\\\s*(BGSFX|FGSFX)\s*>>", "", T, flags=re.IGNORECASE)
    return T


def PreprocessInput(Raw):
    Persona = ExtractPersona(Raw)
    Txt = NormalizeText(StripNonEmotionTags(Raw))
    Prefix = f"Persona: {Persona}. Story: "
    Encoded = Prefix + Txt
    print(f"[Preprocess] Persona='{Persona}' | Prefix='{Prefix}'")
    return Encoded, Persona, Prefix

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

def Reconstruct(EncIds, AttnMask, OutLogits, Emos, Threshold=ThresholdDefault, SkipTokens=0):
    BioProbs = OutLogits["Bio"].softmax(-1)[0].cpu().numpy()
    ClsIds = OutLogits["Cls"].argmax(-1)[0].cpu().numpy()
    Intens = OutLogits["Inten"][0].cpu().numpy()

    Toks = Tok.convert_ids_to_tokens(EncIds[0])
    Words = [T for T in Toks if T not in ["<s>", "</s>", "<pad>"]]
    TextOut = []
    BCount, ICount = 0, 0

    PrevEmo, PrevInten, PrevPB = None, None, 0.0

    for I, T in enumerate(Words):
        TokTxt = T.replace("Ġ", " ")

        if I < SkipTokens:
            TextOut.append(TokTxt)
            continue

        PB = BioProbs[I, 1] if I < BioProbs.shape[0] else 0.0
        PI = BioProbs[I, 2] if I < BioProbs.shape[0] else 0.0
        Emo = Emos[ClsIds[I]] if 0 <= ClsIds[I] < len(Emos) else "NEUTRAL"
        Intensity = int(max(1, min(9, round(float(Intens[I]) * 8 + 1))))

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

        TextOut.append(TokTxt)

    if PrevEmo:
        TextOut.append(f"<<\\{PrevEmo}>>")

    if (BCount + ICount) == 0:
        Bi = np.maximum(BioProbs[:len(Words), 1], BioProbs[:len(Words), 2])
        J = int(np.argmax(Bi))
        Emo = Emos[ClsIds[J]] if 0 <= ClsIds[J] < len(Emos) else "NEUTRAL"
        Intensity = int(max(1, min(9, round(float(Intens[J]) * 8 + 1))))
        if Emo != "NEUTRAL":
            Words[J] = f"<<{Emo} Intensity={Intensity}>>{Words[J].replace('Ġ', ' ')}<<\\{Emo}>>"
        TextOut = ["".join(Words)]
        BCount, ICount = 1, 0

    Txt = "".join(TextOut)
    Txt = re.sub(r"\s+([.,!?;:])", r"\1", Txt)
    Txt = re.sub(r"\s{2,}", " ", Txt).strip()
    Txt = re.split(r"={10,}\s*Input\s*:", Txt, flags=re.IGNORECASE)[0].strip()

    Txt = (
        Txt.encode("utf-8", "ignore").decode("utf-8")
            .replace("â€™", "’").replace("â€˜", "‘")
            .replace("â€œ", "“").replace("â€", "”")
            .replace("â€”", "—").replace("â€“", "–")
            .replace("â€¦", "…").replace("â€", "\"")
            .replace("âĢľ", "“").replace("âĢĿ", "”")
            .replace("âĢĻ", "‘").replace("âĢĽ", "’")
    )

    Txt = re.sub(r"<<Neutral Intensity=\d+>>", "", Txt)
    Txt = re.sub(r"<<\\Neutral>>", "", Txt)

    return Txt, BCount, ICount

def ProcessSingleText(RawText, Threshold=ThresholdDefault):
    """Process a single text string and return emotion-tagged text."""
    FullInput, Persona, Prefix = PreprocessInput(RawText)
    print(f"[ProcessSingleText] Persona='{Persona}' | Initial Threshold={Threshold:.2f}")

    Enc = Tok(FullInput, return_tensors="pt", truncation=True, max_length=MaxTokens).to(Device)
    PrefixEnc = Tok(Prefix, return_tensors="pt").input_ids.shape[1] - 2
    SkipTokens = PrefixEnc

    thresholds_to_try = [Threshold, 0.35, 0.30, 0.25, 0.20]
    Text, BCount, ICount = "", 0, 0

    for Th in thresholds_to_try:
        print(f"[Attempt] Processing at threshold={Th:.2f}")
        with torch.no_grad():
            Outputs = Model(Enc["input_ids"], Enc["attention_mask"])

        Text, BCount, ICount = Reconstruct(
            Enc["input_ids"], Enc["attention_mask"], Outputs, Emos,
            Threshold=Th, SkipTokens=SkipTokens
        )

        if BCount + ICount > 0:
            print(f"[Success] Tags detected at threshold={Th:.2f} (B={BCount}, I={ICount})")
            Threshold = Th
            break
        else:
            print(f"[Retry] No tags found at threshold={Th:.2f}")

    Text = re.sub(r"^Persona\s*:\s*[^.]+\.\s*Story\s*:\s*", "", Text, flags=re.IGNORECASE)
    Text = re.sub(r"\s*Persona\s*:\s*", "", Text, flags=re.IGNORECASE)
    Text = re.sub(r"\s*Story\s*:\s*", "", Text, flags=re.IGNORECASE)
    Text = Text.replace("Ġ", " ").strip()

    if BCount + ICount == 0:
        print("[WARN] No emotion tags detected even after all threshold attempts.")

    return {
        "persona": Persona,
        "text_tagged": Text,
        "b_tags": BCount,
        "i_tags": ICount,
        "threshold": Threshold
    }


def ProcessTextArray(TextList, Threshold=ThresholdDefault):
    Results = []
    for Txt in TextList:
        Results.append(ProcessSingleText(Txt, Threshold))
    return Results
