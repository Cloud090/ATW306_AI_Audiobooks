import os, re, csv, random, torch, numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

TrainTSV = "TaggedTextDataset_train.tsv"
ValTSV = "TaggedTextDataset_val.tsv"
EmotionsTxt = "Emotions.txt"
BaseModel = "roberta-base"
OutputDir = "emotion_tagger_v21c"

MaxTokens = 450
BatchSize = 4
Epochs = 25
LrEncoder = 2e-5
LrHeads = 5e-4
Warmup = 0.1
WeightDecay = 0.05
FreezeEncoder = True
UnfreezeTopLayers = 4
DropoutP = 0.2
LabelSmooth = 0.05
GradNoiseStd = 1e-5
EarlyStopPatience = 4
ColdRestartEvery = 12
Device = "cuda" if torch.cuda.is_available() else "cpu"
Seed = 42

# Weighting - preference tags over no tags
BioClassWeights = (1.0, 3.0, 3.0)
MinTagRate = 0.03
LambdaNoTag = 2.0
CoverageAlpha = 0.25

# Seed
def SetSeed(S=42):
    random.seed(S)
    np.random.seed(S)
    torch.manual_seed(S)
    torch.cuda.manual_seed_all(S)
SetSeed(Seed)

def ReadList(Path):
    with open(Path, "r", encoding="utf-8") as F:
        return [L.strip() for L in F if L.strip()]

def ReadTsv(Path):
    Rows = []
    with open(Path, "r", encoding="utf-8", newline="") as F:
        Reader = csv.DictReader(F, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for R in Reader:
            Rows.append({
                "Input": (R.get("input") or "").lstrip("\ufeff"),
                "Output": (R.get("output") or "")
            })
    return Rows

def NormalizeText(S):
    return re.sub(r"\s+", " ", S).strip()

PersonaTags = [
    re.compile(r"<<\s*PERSONA\s*=\s*([^>]+)>>", re.IGNORECASE),
    re.compile(r"<<\s*PERSONA:?\s*([^>]+)>>", re.IGNORECASE)
]
SfxStyleTag = re.compile(r"<<\s*SFXSTYLE\s*=\s*[^>]+>>", re.IGNORECASE)

def ExtractPersona(Txt):
    for Rx in PersonaTags:
        M = Rx.search(Txt)
        if M:
            return M.group(1).strip()
    return "Unknown"

def StripNonEmotionTags(T):
    T = PersonaTags[0].sub("", T)
    T = PersonaTags[1].sub("", T)
    T = SfxStyleTag.sub("", T)
    T = re.sub(r"<<\s*(BGSFX|FGSFX)[^>]*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\\\s*(BGSFX|FGSFX)\s*>>", "", T, flags=re.IGNORECASE)
    return T

EmoOpen = re.compile(r"<<\s*([A-Z]+)\s+Intensity\s*=\s*([1-9])\s*>>")
EmoClose = re.compile(r"<<\\\s*([A-Z]+)\s*>>")

def ParseEmotionSpans(Tagged):
    Spans, Text, Stack = [], [], []
    I = 0
    while I < len(Tagged):
        M1 = EmoOpen.match(Tagged, I)
        if M1:
            Emo = M1.group(1).upper()
            Inten = int(M1.group(2))
            Stack.append((Emo, Inten, len(Text)))
            I = M1.end()
            continue
        M2 = EmoClose.match(Tagged, I)
        if M2 and Stack:
            Emo, Inten, St = Stack.pop()
            Spans.append((St, len(Text), {"Emo": Emo, "Int": Inten}))
            I = M2.end()
            continue
        Text.append(Tagged[I])
        I += 1
    return "".join(Text), Spans

# Get dataset
class EmotionDataset(Dataset):
    def __init__(self, Rows, Tok, Emos):
        self.Tok = Tok
        self.EmoToId = {E.upper(): I for I, E in enumerate(Emos)}
        self.Samples = []
        for R in Rows:
            Persona = ExtractPersona(R["Input"])
            Out = StripNonEmotionTags(R["Output"])
            Clean, Sp = ParseEmotionSpans(NormalizeText(Out))
            Prefix = f"Persona: {Persona}. Story: "
            Enc = Tok(Prefix + Clean, return_offsets_mapping=True,
                      truncation=True, max_length=MaxTokens)
            N = len(Enc["input_ids"])
            Bio = np.zeros(N, int)
            Cls = np.full(N, -100, int)
            Inten = np.zeros(N, float)
            Msk = np.zeros(N, float)
            Shift = len(Prefix)
            for St, En, M in Sp:
                A, B = St + Shift, En + Shift
                Idxs = [I for I, (X, Y) in enumerate(Enc["offset_mapping"]) if not (Y <= A or X >= B)]
                if not Idxs:
                    continue
                Bio[Idxs[0]] = 1
                for T in Idxs[1:]:
                    Bio[T] = 2
                Eid = self.EmoToId.get(M["Emo"])
                if Eid is None:
                    continue
                Cls[Idxs] = Eid
                Inten[Idxs] = (M["Int"] - 1) / 8.0
                Msk[Idxs] = 1.0
            self.Samples.append({
                "Enc": Enc,
                "EmoBio": Bio,
                "EmoCls": Cls,
                "EmoInt": Inten,
                "EmoMsk": Msk,
                "Persona": Persona,
                "ExpectedOutput": R["Output"],
                "OrigInput": Prefix + Clean
            })
        print(f"[Dataset] {len(self.Samples)} samples.")
    def __len__(self): return len(self.Samples)
    def __getitem__(self, I): return self.Samples[I]

class Collator:
    def __call__(self, B):
        MaxLen = max(len(X["Enc"]["input_ids"]) for X in B)
        Pad = lambda V, P: V + [P] * (MaxLen - len(V))
        Ids, Mask, Bio, Cls, Inten, Msk = [], [], [], [], [], []
        for S in B:
            Ids.append(Pad(S["Enc"]["input_ids"], 1))
            Mask.append(Pad(S["Enc"]["attention_mask"], 0))
            Bio.append(Pad(S["EmoBio"].tolist(), 0))
            Cls.append(Pad(S["EmoCls"].tolist(), -100))
            Inten.append(Pad(S["EmoInt"].tolist(), 0.0))
            Msk.append(Pad(S["EmoMsk"].tolist(), 0.0))
        return {
            "InputIds": torch.tensor(Ids),
            "AttentionMask": torch.tensor(Mask),
            "EmoBio": torch.tensor(Bio),
            "EmoCls": torch.tensor(Cls),
            "EmoInt": torch.tensor(Inten),
            "EmoMsk": torch.tensor(Msk)
        }

# Model definition
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

# Loss function
def ComputeLoss(O, B):
    Dev = B["AttentionMask"].device
    BioW = torch.tensor(BioClassWeights, device=Dev)
    CeBio = nn.CrossEntropyLoss(weight=BioW, ignore_index=-100, label_smoothing=LabelSmooth)
    CeCls = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LabelSmooth)
    Mse = nn.MSELoss(reduction="none")

    M = B["AttentionMask"].float()
    G = B["EmoBio"].clone()
    G[M == 0] = -100
    LBio = CeBio(O["Bio"].transpose(1, 2), G)

    FlatLogits = O["Cls"].reshape(-1, O["Cls"].size(-1))
    FlatTargets = B["EmoCls"].reshape(-1)
    Valid = FlatTargets != -100
    LCls = CeCls(FlatLogits[Valid], FlatTargets[Valid]) if Valid.any() else torch.tensor(0., device=Dev)

    Msk = (B["EmoMsk"] > 0) & (M > 0)
    if Msk.any():
        LVal = Mse(O["Inten"], B["EmoInt"])
        LInt = (LVal * Msk).sum() / Msk.sum().clamp(min=1.0)
    else:
        LInt = torch.tensor(0., device=Dev)

    P = O["Bio"].softmax(-1)
    PB, PI = P[:, :, 1], P[:, :, 2]
    PBI_Mean = ((PB + PI) * M).sum() / M.sum().clamp(min=1.0)
    CoveragePen = CoverageAlpha * (1.0 - PBI_Mean)

    SeqLens = M.sum(dim=1).clamp(min=1.0)
    SeqBi = (PB + PI).sum(dim=1) / SeqLens
    Deficit = torch.clamp(MinTagRate - SeqBi, min=0.0)
    NoTagPen = LambdaNoTag * Deficit.mean()

    Total = LBio + LCls + 0.25 * LInt + CoveragePen + NoTagPen
    return torch.nan_to_num(Total)

@torch.no_grad()
def ComputeF1(Logits, Gold, Mask):
    P = Logits.argmax(-1)
    Mask = Mask.bool()
    PosP, PosG = (P > 0), (Gold > 0)
    Tp = (PosP & PosG)[Mask].sum().item()
    Fp = (PosP & ~PosG)[Mask].sum().item()
    Fn = (~PosP & PosG)[Mask].sum().item()
    Pr = Tp / (Tp + Fp + 1e-9)
    Rc = Tp / (Tp + Fn + 1e-9)
    return 2 * Pr * Rc / (Pr + Rc + 1e-9)

# Training function
def Train():
    Emos = ReadList(EmotionsTxt)
    Tok = AutoTokenizer.from_pretrained(BaseModel, use_fast=True)
    Tr = EmotionDataset(ReadTsv(TrainTSV), Tok, Emos)
    Va = EmotionDataset(ReadTsv(ValTSV), Tok, Emos)
    Col = Collator()
    TrDl = DataLoader(Tr, batch_size=BatchSize, shuffle=True, collate_fn=Col)
    VaDl = DataLoader(Va, batch_size=BatchSize, shuffle=False, collate_fn=Col)

    Model = EmotionTagger(BaseModel, len(Emos))
    if FreezeEncoder:
        for P in Model.Enc.parameters(): P.requires_grad = False
        for L in Model.Enc.encoder.layer[-UnfreezeTopLayers:]:
            for P in L.parameters(): P.requires_grad = True
    Model.to(Device)

    Head = list(Model.Bio.parameters()) + list(Model.Cls.parameters()) + list(Model.Inten.parameters())
    Enc = [P for P in Model.Enc.parameters() if P.requires_grad]
    Opt = torch.optim.AdamW(
        [{"params": Head, "lr": LrHeads, "weight_decay": WeightDecay},
         {"params": Enc, "lr": LrEncoder, "weight_decay": WeightDecay}]
    )
    Sched = get_linear_schedule_with_warmup(Opt, int(Warmup * len(TrDl) * Epochs), len(TrDl) * Epochs)

    Best, Bad = float("inf"), 0
    BestPath = f"{OutputDir}/best"
    os.makedirs(BestPath, exist_ok=True)

    for Ep in range(1, Epochs + 1):
        Model.train()
        Tot = 0
        for I, B in enumerate(TrDl, 1):
            B = {K: V.to(Device) for K, V in B.items()}
            O = Model(B["InputIds"], B["AttentionMask"])
            Loss = ComputeLoss(O, B)
            Opt.zero_grad()
            Loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.0)
            for P in Model.parameters():
                if P.grad is not None:
                    P.grad.add_(GradNoiseStd * torch.randn_like(P.grad))
            Opt.step()
            Sched.step()
            Tot += Loss.item()
            if I % 50 == 0:
                print(f"Ep{Ep} step{I}/{len(TrDl)} loss={Loss.item():.5f}")

        Model.eval()
        Vl, F1S, Nf = 0, 0, 0
        with torch.no_grad():
            for B in VaDl:
                B = {K: V.to(Device) for K, V in B.items()}
                O = Model(B["InputIds"], B["AttentionMask"])
                Vl += float(ComputeLoss(O, B))
                F1S += float(ComputeF1(O["Bio"], B["EmoBio"], B["AttentionMask"]))
                Nf += 1
        Vl /= max(1, Nf)
        F1S /= max(1, Nf)
        print(f"[Ep{Ep}] val_loss={Vl:.4f} F1={F1S:.3f}")
        if Vl < Best:
            Best = Vl
            Bad = 0
            torch.save(Model.state_dict(), f"{BestPath}/model.bin")
            Tok.save_pretrained(BestPath)
            print("[Best] saved")
        else:
            Bad += 1
            if Bad >= EarlyStopPatience:
                print(f"[EarlyStop] epoch {Ep}")
                break

    print("\nLoading best model for testing:\n\n")
    Model = EmotionTagger(BaseModel, len(Emos))
    Model.load_state_dict(torch.load(f"{BestPath}/model.bin", map_location=Device))
    Model.to(Device)
    Model.eval()

    def Reconstruct(EncIds, AttnMask, OutLogits, Threshold=0.40):
        BioProbs = OutLogits["Bio"].softmax(-1)[0].cpu().numpy()
        ClsIds = OutLogits["Cls"].argmax(-1)[0].cpu().numpy()
        Intens = OutLogits["Inten"][0].cpu().numpy()

        Toks = Tok.convert_ids_to_tokens(EncIds[0])
        Words = [T for T in Toks if T not in ["<s>", "</s>", "<pad>"]]
        TextOut, OpenEmo = [], None
        BCount, ICount = 0, 0

        for I, T in enumerate(Words):
            TokTxt = T.replace("Ġ", " ")
            PB = BioProbs[I, 1] if I < BioProbs.shape[0] else 0.0
            PI = BioProbs[I, 2] if I < BioProbs.shape[0] else 0.0

            if PB >= Threshold:
                Emo = Emos[ClsIds[I]] if 0 <= ClsIds[I] < len(Emos) else "NEUTRAL"
                Intensity = int(max(1, min(9, round(float(Intens[I]) * 8 + 1))))
                TextOut.append(f"<<{Emo} Intensity={Intensity}>>")
                OpenEmo = Emo
                BCount += 1
            elif PI >= Threshold:
                ICount += 1

            TextOut.append(TokTxt)

            EndOfSpan = (I == len(Words) - 1) or (BioProbs[I + 1, 1] < Threshold and BioProbs[I + 1, 2] < Threshold)
            if OpenEmo and EndOfSpan:
                TextOut.append(f"<<\\{OpenEmo}>>")
                OpenEmo = None

        if (BCount + ICount) == 0:
            Bi = np.maximum(BioProbs[:len(Words), 1], BioProbs[:len(Words), 2])
            J = int(np.argmax(Bi))
            Emo = Emos[ClsIds[J]] if 0 <= ClsIds[J] < len(Emos) else "NEUTRAL"
            Intensity = int(max(1, min(9, round(float(Intens[J]) * 8 + 1))))
            Words[J] = f"<<{Emo} Intensity={Intensity}>>{Words[J].replace('Ġ', ' ')}<<\\{Emo}>>"
            TextOut = ["".join(Words)]
            BCount, ICount = 1, 0

        Txt = "".join(TextOut)
        Txt = re.sub(r"\s+([.,!?;:])", r"\1", Txt)
        Txt = re.sub(r"\s{2,}", " ", Txt).strip()
        Txt = re.split(r"={10,}\s*Input\s*:", Txt, flags=re.IGNORECASE)[0].strip()
        Txt = (
            Txt.encode("utf-8", "ignore").decode("utf-8")
                .replace("â€™", "’")
                .replace("âĢĻ", "—")
                .replace("âĢĵ", "–")
                .replace("â€¦", "…")
                .replace("â€œ", "“")
                .replace("â€", "”")
        )
        return Txt, BCount, ICount

    print("\nTesting random validation samples:\n\n")
    Emos = ReadList(EmotionsTxt)
    for Idx, S in enumerate(random.sample(Va.Samples, min(3, len(Va.Samples))), 1):
        Enc = Tok(S["OrigInput"], return_tensors="pt", truncation=True, max_length=MaxTokens).to(Device)
        with torch.no_grad():
            O = Model(Enc["input_ids"], Enc["attention_mask"])
        Thr = 0.40
        while True:
            Text, Bc, Ic = Reconstruct(Enc["input_ids"], Enc["attention_mask"], O, Threshold=Thr)
            if (Bc + Ic) > 0 or Thr <= 0.05:
                break
            Thr -= 0.05
        print(f"\nSample {Idx}\n")
        print(f"Persona: {S['Persona']}")
        print(f"(Predicted {Bc} B-tags, {Ic} I-tags) [Final threshold={Thr:.2f}]")
        print("\nExpected Output:\n" + S["ExpectedOutput"])
        print("\nPredicted Output:\n" + Text)
        print("------------------------")

    print("\nDone.")

Train()
