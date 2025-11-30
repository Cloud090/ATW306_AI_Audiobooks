# This module runs the Foreground SFX tagging model, a fine-tined RoBERTa-based model designed to identify context-based positions in the provided text where
# a foreground sound could be added. 

# The model returns Begin/Inside/Outside probabilities for each token - even though no spans are needed for foreground sound effects, reusing the BIO technique for training saved
# rewriting the training logic from scratch for the new model. The "Begin" probabilities can then be mapped into the most likely sound effect from
# a predefined list (loaded from "ForegroundSFX.txt"). These can then be converted into tags, in the format "<<FGSFX Sound=[Sound]>>" (Note: No closing tags needed for foreground sounds).

import os
import re
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Set up base model name, and locations of needed files.
BaseModel = "roberta-base"
BaseDir = os.path.dirname(os.path.abspath(__file__))
ModelDir = os.path.join(BaseDir, "FGSFXTagger")
SfxListPath = os.path.join(BaseDir, "ForegroundSFX.txt")


MaxTokens = 450
Device = "cuda" if torch.cuda.is_available() else "cpu"

# Dropout (kept due to training process), threshold needed for sound effect to appear, and threshold needed for a new sound effect.
DropoutP = 0.2
ThresholdDefault = 0.40
TagChangeDelta = 0.15

# Function to read a text file, returning a list of lines.
def ReadList(Path):
    with open(Path, "r", encoding="utf-8") as F:
        return [L.strip() for L in F if L.strip()]

# Function to clean text - collapses multiple spaces into single space, and trims white space from front and end of text.
def NormalizeText(S):
    return re.sub(r"\s+", " ", S).strip()

# Function to remove all tags unrelated to foreground sound effects from the text
def StripNonSfxTags(T):
    T = re.sub(r"<<\s*(HAPPY|SAD|FEAR|ANGRY|DISGUST)[^>]*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\\\s*(HAPPY|SAD|FEAR|ANGRY|DISGUST)\s*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\s*BGSFX[^>]*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\\\s*(BGSFX)\s*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\s*SFXSTYLE\s*=\s*[^>]+>>", "", T, flags=re.IGNORECASE)
    return T

# Function combining above two functions.
def PreprocessInput(Raw):
    Txt = NormalizeText(StripNonSfxTags(Raw))
    return Txt

# Sets up a class for the FGSFX tagging model.
class FGTagger(nn.Module):
    def __init__(self, Base, NumTags):
        super().__init__()

        # Load the pretrained model encoder
        self.Enc = AutoModel.from_pretrained(Base)

        # Hidden dimension size of the encoder output (needed for classifying)
        H = self.Enc.config.hidden_size

        # Apply a dropout (used in training, so remains for inference)
        self.Drop = nn.Dropout(DropoutP)

        # Layer to predict beginning, inside, and outside tags - Beginning tags used to find FGSFX tag locations, other probabilities calculated for consistency with other models.
        self.Bio = nn.Linear(H, 3)

        # Layer to predict sound effect classification
        self.Cls = nn.Linear(H, NumTags)

    def forward(self, InputIds, AttentionMask):
        # Run input through the encoder
        H = self.Enc(input_ids=InputIds, attention_mask=AttentionMask).last_hidden_state

        # Apply the dropout
        H = self.Drop(H)

        # Return lists for each token - "beginning, inside, and outside" probabilities, and sound effect classifier probabilities.
        return {"Bio": self.Bio(H), "Cls": self.Cls(H)}

# Gets list of foreground sound effects from file
SfxTags = ReadList(SfxListPath)
print(f"[Loaded {len(SfxTags)} FGSFX tags from {SfxListPath}]")

# Loads the tokeniser saved with the trained model.
Tok = AutoTokenizer.from_pretrained(ModelDir)

# Loads the trained model architecture, using the above class.
Model = FGTagger(BaseModel, len(SfxTags))

# Loads the trained weights
State = torch.load(os.path.join(ModelDir, "model.bin"), map_location=Device)

# Loads the weights into the model.
Model.load_state_dict(State, strict=False)

# Moves the model to the appropriate device (should be GPU)
Model.to(Device).eval()

print(f"[Model loaded successfully from {ModelDir}\\model.bin]")

# Converts token predictions into tagged text
def Reconstruct(EncIds, AttnMask, OutLogits, SfxTags, Threshold=ThresholdDefault):
    # Converts the ""beginning, inside, and outside" results into a probabilities list
    BioProbs = OutLogits["Bio"].softmax(-1)[0].cpu().numpy()

    # Converts the sound effect classifier results into a probabilities list.
    ClsIds = OutLogits["Cls"].argmax(-1)[0].cpu().numpy()

    # Convert token IDs back to text-based tokens.
    Toks = Tok.convert_ids_to_tokens(EncIds[0])

    # Filter out non-text tokens 
    Words = [T for T in Toks if T not in ["<s>", "</s>", "<pad>"]]

    # Sets up variables for tag analysis
    TextOut, PrevTag, PrevPB = [], None, 0.0
    BCount, ICount = 0, 0

    # Iterate through the text token list
    for I, T in enumerate(Words):
        # If text token has word start character, replaces it with a space.
        TokTxt = T.replace("Ġ", " ")

        # Retrieves probabilities
        PB = BioProbs[I, 1] if I < BioProbs.shape[0] else 0.0
        PI = BioProbs[I, 2] if I < BioProbs.shape[0] else 0.0

        # Retrieves detected sound effect token
        Tag = SfxTags[ClsIds[I]] if 0 <= ClsIds[I] < len(SfxTags) else "Unknown"

        # If the probability is higher than the threshold, checks if either no previous tag exists, or that the threshold difference is over the delta (to avoid too many sound effects
        # in a short span). If true, sets "should_change" to true.
        if PB >= Threshold:
            should_change = False
            if PrevTag is None:
                should_change = True
            elif Tag != PrevTag and (PB - PrevPB) > TagChangeDelta:
                should_change = True

            # If "should_change" is true, adds the sound effect tag, based on the sound effect classifier result.
            if should_change:
                TextOut.append(f"<<FGSFX Sound={Tag}>>")
                PrevTag, PrevPB = Tag, PB
                BCount += 1

        # Counts "inside" tokens.
        elif PI >= Threshold:
            ICount += 1

        # Appends the token to the text output list.
        TextOut.append(TokTxt)

    # If no sound effect tags were created, add one at the highest probability token
    if (BCount + ICount) == 0:
        Bi = np.maximum(BioProbs[:len(Words), 1], BioProbs[:len(Words), 2])
        J = int(np.argmax(Bi))
        Tag = SfxTags[ClsIds[J]] if 0 <= ClsIds[J] < len(SfxTags) else "Unknown"
        Words[J] = f"<<FGSFX Sound={Tag}>>{Words[J].replace('Ġ', ' ')}"
        TextOut = ["".join(Words)]
        BCount, ICount = 1, 0

    # Combines the text output list into a tagged text string, cleans up punctuation, and returns it along with diagnostic metadata.
    Txt = "".join(TextOut)
    Txt = re.sub(r"\s+([.,!?;:])", r"\1", Txt)
    Txt = re.sub(r"\s{2,}", " ", Txt).strip()
    Txt = re.split(r"={10,}\s*Input\s*:", Txt, flags=re.IGNORECASE)[0].strip()

    return Txt, BCount, ICount

# Run the foreground sound effect tagging process on a single block of text.
def ProcessSingleText(RawText, SFXStyle="Balanced"):
    # Preprocess the text - collapse multiple spaces to single, remove preexisting tags.
    Txt = PreprocessInput(RawText)

    # From the provided SFXStyle, calculates the threshold for sound effect tags
    SFXStyle = SFXStyle.lower().strip()
    if SFXStyle == "none":
        # If the SFXStyle is "none", no tags should be added - returns unedited text without running the model.
        print("[SFXStyle: None] Returning unedited text.")
        return {"text_tagged": Txt, "b_tags": 0, "i_tags": 0, "threshold": None}
    elif SFXStyle == "subtle":
         # If the SFXStyle is "subtle", the threshold is increased by 10% (less probability of added tags)
        threshold = ThresholdDefault + 0.10
    elif SFXStyle == "cinematic":
         # If the SFXStyle is "cinematic", the threshold is decreased by 10% (more probability of added tags)
        threshold = ThresholdDefault - 0.10
    else:
        # If the persona is "neutral", or anything else, the threshold is unchanged from default.
        threshold = ThresholdDefault

    # Encode the text using the saved tokeniser.
    Enc = Tok(Txt, return_tensors="pt", truncation=True, max_length=MaxTokens).to(Device)

    # Run the text through the model, saving the output.
    with torch.no_grad():
        Outputs = Model(Enc["input_ids"], Enc["attention_mask"])

    # Reconstruct the tagged text using the model output, sound effect list, and threshold calculated earlier.
    Text, BCount, ICount = Reconstruct(Enc["input_ids"], Enc["attention_mask"], Outputs, SfxTags, Threshold=threshold)
    TxtOut = Text.replace("Ġ", " ").strip()

    # Return the tagged text, with diagnostic metadata
    return {
        "text_tagged": TxtOut,
        "b_tags": BCount,
        "i_tags": ICount,
        "threshold": threshold,
        "style": SFXStyle
    }

# Function to process a list of text, using the ProcessSingleText function above on every entry.
def ProcessTextArray(TextList, SFXStyle="Balanced"):
    Results = []
    for Txt in TextList:
        Results.append(ProcessSingleText(Txt, SFXStyle=SFXStyle))
    return Results
