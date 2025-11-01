# BGSFX_Tagger_v3_NoIntensity.py
# -------------------------------
# Trains a RoBERTa-based span tagger to insert <<BGSFX=...>> tags (no Intensity).
# Uses SFXSTYLE context (Cinematic, etc.) instead of PERSONA.

import os, re, csv, random, torch, numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ---------- SETTINGS ----------
TrainTSV = "TaggedTextDataset_train.tsv"
ValTSV = "TaggedTextDataset_val.tsv"
BgsfxTxt = "BackgroundSFX.txt"
BaseModel = "roberta-base"
OutputDir = "bgsfx_tagger_v3"

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
Device = "cuda" if torch.cuda.is_available() else "cpu"
Seed = 42

BioClassWeights = (1.0, 3.0, 3.0)
MinTagRate = 0.03
LambdaNoTag = 2.0
CoverageAlpha = 0.25

# ---------- HELPERS ----------
def SetSeed(S=42):
    random.seed(S); np.random.seed(S); torch.manual_seed(S); torch.cuda.manual_seed_all(S)
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

def NormalizeText(S): return re.sub(r"\s+", " ", S).strip()

# ---------- TAG HELPERS ----------
SfxStyleTag = re.compile(r"<<\s*SFXSTYLE\s*=\s*([^>]+)>>", re.IGNORECASE)
def ExtractSfxStyle(Txt):
    M = SfxStyleTag.search(Txt)
    return M.group(1).strip() if M else "Neutral"

def StripNonBgsfxTags(T):
    T = re.sub(r"<<\s*(PERSONA|FGSFX)[^>]*>>", "", T, flags=re.I)
    T = re.sub(r"<<\\\s*(PERSONA|FGSFX)\s*>>", "", T, flags=re.I)
    T = re.sub(r"<<\s*[A-Z]{2,}\s*(?:Intensity\s*=\s*\d+)?\s*>>", "", T)
    T = re.sub(r"<<\\\s*[A-Z]{2,}\s*>>", "", T)
    return T

BgsfxOpen = re.compile(r"<<\s*BGSFX\s*=\s*([A-Za-z0-9_]+)[^>]*>>")
BgsfxClose = re.compile(r"<<\\\s*BGSFX\s*>>")

def ParseBgsfxSpans(Tagged):
    Spans, Text, Stack = [], [], []
    i = 0
    while i < len(Tagged):
        m1 = BgsfxOpen.match(Tagged, i)
        if m1:
            SFX = m1.group(1).upper()
            Stack.append((SFX, len(Text)))
            i = m1.end(); continue
        m2 = BgsfxClose.match(Tagged, i)
        if m2 and Stack:
            SFX, st = Stack.pop()
            Spans.append((st, len(Text), {"SFX": SFX}))
            i = m2.end(); continue
        Text.append(Tagged[i])
        i += 1
    return "".join(Text), Spans

# ---------- DATASET ----------
class BGSFXDataset(Dataset):
    def __init__(self, Rows, Tok, SfxList):
        self.Tok = Tok
        self.SfxToId = {S.upper(): I for I, S in enumerate(SfxList)}
        self.Samples = []
        for R in Rows:
            SfxStyle = ExtractSfxStyle(R["Input"])
            Out = StripNonBgsfxTags(R["Output"])
            Clean, Sp = ParseBgsfxSpans(NormalizeText(Out))
            Prefix = f"SFXStyle: {SfxStyle}. Story: "
            Enc = Tok(Prefix + Clean, return_offsets_mapping=True, truncation=True, max_length=MaxTokens)
            N = len(Enc["input_ids"])
            Bio = np.zeros(N, int)
            Cls = np.full(N, -100, int)
            Shift = len(Prefix)
            for St, En, M in Sp:
                A, B = St + Shift, En + Shift
                Idxs = [I for I, (X, Y) in enumerate(Enc["offset_mapping"]) if not (Y <= A or X >= B)]
                if not Idxs: continue
                Bio[Idxs[0]] = 1
                for T in Idxs[1:]: Bio[T] = 2
                Sid = self.SfxToId.get(M["SFX"])
                if Sid is not None:
                    Cls[Idxs] = Sid
            self.Samples.append({
                "Enc": Enc,
                "SfxBio": Bio,
                "SfxCls": Cls,
                "SFXStyle": SfxStyle,
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
        Ids, Mask, Bio, Cls = [], [], [], []
        for S in B:
            Ids.append(Pad(S["Enc"]["input_ids"], 1))
            Mask.append(Pad(S["Enc"]["attention_mask"], 0))
            Bio.append(Pad(S["SfxBio"].tolist(), 0))
            Cls.append(Pad(S["SfxCls"].tolist(), -100))
        return {
            "InputIds": torch.tensor(Ids),
            "AttentionMask": torch.tensor(Mask),
            "SfxBio": torch.tensor(Bio),
            "SfxCls": torch.tensor(Cls)
        }

# ---------- MODEL ----------
class BGSFXTagger(nn.Module):
    def __init__(self, Base, NumSfx):
        super().__init__()
        self.Enc = AutoModel.from_pretrained(Base)
        H = self.Enc.config.hidden_size
        self.Drop = nn.Dropout(DropoutP)
        self.Bio = nn.Linear(H, 3)
        self.Cls = nn.Linear(H, NumSfx)
    def forward(self, InputIds, AttentionMask):
        H = self.Enc(input_ids=InputIds, attention_mask=AttentionMask).last_hidden_state
        H = self.Drop(H)
        return {"Bio": self.Bio(H), "Cls": self.Cls(H)}

# ---------- LOSS ----------
def ComputeLoss(O, B):
    Dev = B["AttentionMask"].device
    BioW = torch.tensor(BioClassWeights, device=Dev)
    CeBio = nn.CrossEntropyLoss(weight=BioW, ignore_index=-100, label_smoothing=LabelSmooth)
    CeCls = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LabelSmooth)
    M = B["AttentionMask"].float()
    G = B["SfxBio"].clone(); G[M == 0] = -100
    LBio = CeBio(O["Bio"].transpose(1, 2), G)

    FlatLogits = O["Cls"].reshape(-1, O["Cls"].size(-1))
    FlatTargets = B["SfxCls"].reshape(-1)
    Valid = FlatTargets != -100
    LCls = CeCls(FlatLogits[Valid], FlatTargets[Valid]) if Valid.any() else torch.tensor(0., device=Dev)

    P = O["Bio"].softmax(-1)
    PB, PI = P[:, :, 1], P[:, :, 2]
    PBI_Mean = ((PB + PI) * M).sum() / M.sum().clamp(min=1.0)
    CoveragePen = CoverageAlpha * (1.0 - PBI_Mean)

    SeqLens = M.sum(dim=1).clamp(min=1.0)
    SeqBi = (PB + PI).sum(dim=1) / SeqLens
    Deficit = torch.clamp(MinTagRate - SeqBi, min=0.0)
    NoTagPen = LambdaNoTag * Deficit.mean()

    return torch.nan_to_num(LBio + LCls + CoveragePen + NoTagPen)

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

# ---------- TRAIN ----------
def Train():
    SfxList = ReadList(BgsfxTxt)
    Tok = AutoTokenizer.from_pretrained(BaseModel, use_fast=True)
    Tr = BGSFXDataset(ReadTsv(TrainTSV), Tok, SfxList)
    Va = BGSFXDataset(ReadTsv(ValTSV), Tok, SfxList)
    Col = Collator()
    TrDl = DataLoader(Tr, batch_size=BatchSize, shuffle=True, collate_fn=Col)
    VaDl = DataLoader(Va, batch_size=BatchSize, shuffle=False, collate_fn=Col)

    Model = BGSFXTagger(BaseModel, len(SfxList))
    if FreezeEncoder:
        for P in Model.Enc.parameters(): P.requires_grad = False
        for L in Model.Enc.encoder.layer[-UnfreezeTopLayers:]:
            for P in L.parameters(): P.requires_grad = True
    Model.to(Device)

    Head = list(Model.Bio.parameters()) + list(Model.Cls.parameters())
    Enc = [P for P in Model.Enc.parameters() if P.requires_grad]
    Opt = torch.optim.AdamW(
        [{"params": Head, "lr": LrHeads, "weight_decay": WeightDecay},
         {"params": Enc, "lr": LrEncoder, "weight_decay": WeightDecay}]
    )
    Sched = get_linear_schedule_with_warmup(Opt, int(Warmup * len(TrDl) * Epochs), len(TrDl) * Epochs)

    Best, Bad = float("inf"), 0
    BestPath = f"{OutputDir}/best"; os.makedirs(BestPath, exist_ok=True)

    for Ep in range(1, Epochs + 1):
        Model.train(); Tot = 0
        for I, B in enumerate(TrDl, 1):
            B = {K: V.to(Device) for K, V in B.items()}
            O = Model(B["InputIds"], B["AttentionMask"])
            Loss = ComputeLoss(O, B)
            Opt.zero_grad(); Loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.0)
            for P in Model.parameters():
                if P.grad is not None:
                    P.grad.add_(GradNoiseStd * torch.randn_like(P.grad))
            Opt.step(); Sched.step()
            Tot += Loss.item()
            if I % 50 == 0:
                print(f"Ep{Ep} step{I}/{len(TrDl)} loss={Loss.item():.5f}")

        Model.eval(); Vl, F1S, Nf = 0, 0, 0
        with torch.no_grad():
            for B in VaDl:
                B = {K: V.to(Device) for K, V in B.items()}
                O = Model(B["InputIds"], B["AttentionMask"])
                Vl += float(ComputeLoss(O, B))
                F1S += float(ComputeF1(O["Bio"], B["SfxBio"], B["AttentionMask"]))
                Nf += 1
        Vl /= max(1, Nf); F1S /= max(1, Nf)
        print(f"[Ep{Ep}] val_loss={Vl:.4f} F1={F1S:.3f}")
        if Vl < Best:
            Best = Vl; Bad = 0
            torch.save(Model.state_dict(), f"{BestPath}/model.bin")
            Tok.save_pretrained(BestPath)
            print("[Best] saved")
        else:
            Bad += 1
            if Bad >= EarlyStopPatience:
                print(f"[EarlyStop] epoch {Ep}"); break

    print("\n=== Validation Samples ===\n")
    Model.load_state_dict(torch.load(f"{BestPath}/model.bin", map_location=Device))
    Model.eval()

    def Reconstruct(EncIds, AttnMask, Out, Thr=0.4):
        BioProbs = Out["Bio"].softmax(-1)[0].cpu().numpy()
        ClsIds = Out["Cls"].argmax(-1)[0].cpu().numpy()
        Toks = Tok.convert_ids_to_tokens(EncIds[0])
        Words = [T.replace("Ġ", " ") for T in Toks if T not in ["<s>", "</s>", "<pad>"]]
        OutTxt, Open = [], None
        for i, W in enumerate(Words):
            PB = BioProbs[i, 1]
            if PB >= Thr:
                SFX = SfxList[ClsIds[i]] if 0 <= ClsIds[i] < len(SfxList) else "UNKNOWN"
                OutTxt.append(f"<<BGSFX={SFX}>>")
                Open = SFX
            OutTxt.append(W)
            if Open and (i == len(Words)-1 or BioProbs[i+1,1] < Thr):
                OutTxt.append("<<\\BGSFX>>"); Open = None
        return re.sub(r"\s+", " ", "".join(OutTxt)).strip()

    for i, S in enumerate(random.sample(Va.Samples, min(3, len(Va.Samples))), 1):
        Enc = Tok(S["OrigInput"], return_tensors="pt", truncation=True, max_length=MaxTokens).to(Device)
        with torch.no_grad():
            O = Model(Enc["input_ids"], Enc["attention_mask"])
        Text = Reconstruct(Enc["input_ids"], Enc["attention_mask"], O)
        print(f"\nSample {i}")
        print(f"SFXStyle: {S['SFXStyle']}")
        print("Expected:\n", S["ExpectedOutput"])
        print("Predicted:\n", Text)
        print("---------------------------")

    print("\nTraining complete.\n")

# ---------- MAIN ----------
if __name__ == "__main__":
    print("=== BGSFX Tagger Training (No Intensity version) ===")
    Train()
