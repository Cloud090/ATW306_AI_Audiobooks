# This module runs the Background SFX tagging model, a fine-tined RoBERTa-based model designed to identify context-based positions in the provided text where
# a section of text could have emotional emphasis.

import os
import re
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Set up base model name, and locations of needed files.
BaseModel = "roberta-base"
BaseDir = os.path.dirname(os.path.abspath(__file__))
ModelDir = os.path.join(BaseDir, "EmotionTagger")
EmotionsTxt = os.path.join(BaseDir, "Emotions.txt")


MaxTokens = 450
Device = "cuda" if torch.cuda.is_available() else "cpu"

# Dropout (kept due to training process), threshold needed for emotion tag to appear, and threshold needed for emotion tags to change to a new emotion.
DropoutP = 0.2
ThresholdDefault = 0.40
TagChangeDelta = 0.01

# Function to read a text file, returning a list of lines.
def ReadList(Path):
    with open(Path, "r", encoding="utf-8") as F:
        return [L.strip() for L in F if L.strip()]

# Function to clean text - collapses multiple spaces into single space, and trims white space from front and end of text.
def NormalizeText(S):
    return re.sub(r"\s+", " ", S).strip()

# Function to remove all tags unrelated to emotion from the text
def StripNonEmotionTags(T):
    T = re.sub(r"<<\s*(BGSFX|FGSFX)[^>]*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\\\s*(BGSFX|FGSFX)\s*>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\s*SFXSTYLE\s*=\s*[^>]+>>", "", T, flags=re.IGNORECASE)
    T = re.sub(r"<<\s*PERSONA\s*=\s*[^>]+>>", "", T, flags=re.IGNORECASE)
    return T

# Function combining above two functions.
def PreprocessInput(Raw):
    Txt = NormalizeText(StripNonEmotionTags(Raw))
    return Txt

# Sets up a class for the emotion tagging model.
class EmotionTagger(nn.Module):
    def __init__(self, Base, NumEmos):
        super().__init__()

        # Load the pretrained model encoder
        self.Enc = AutoModel.from_pretrained(Base)

        # Hidden dimension size of the encoder output (needed for classifying)
        H = self.Enc.config.hidden_size

        # Apply a dropout (used in training, so remains for inference)
        self.Drop = nn.Dropout(DropoutP)

        # Layer to predict beginning, inside, and outside tags, to create spans
        self.Bio = nn.Linear(H, 3)

        # Layer to predict emotion classification
        self.Cls = nn.Linear(H, NumEmos)

        # Layer to predict emotion intensity
        self.Inten = nn.Linear(H, 1)

    def forward(self, InputIds, AttentionMask):
        # Run input through the encoder
        H = self.Enc(input_ids=InputIds, attention_mask=AttentionMask).last_hidden_state

        # Apply the dropout
        H = self.Drop(H)

        # Return lists for each token - "beginning, inside, and outside" probabilities, emotion classifier probabilities, and emotional intensity probabilities.
        return {
            "Bio": self.Bio(H),
            "Cls": self.Cls(H),
            "Inten": torch.sigmoid(self.Inten(H)).squeeze(-1)
        }

# Gets list of emotions from file
Emos = ReadList(EmotionsTxt)
print(f"[Loaded {len(Emos)} emotions from {EmotionsTxt}]")

# Loads the tokeniser saved with the trained model.
Tok = AutoTokenizer.from_pretrained(ModelDir)

# Loads the trained model architecture, using the above class.
Model = EmotionTagger(BaseModel, len(Emos))

# Loads the trained weights
State = torch.load(os.path.join(ModelDir, "model.bin"), map_location=Device)

# Loads the weights into the model.
Model.load_state_dict(State, strict=False)

# Moves the model to the appropriate device (should be GPU)
Model.to(Device).eval()

print(f"[Model loaded successfully from {ModelDir}\\model.bin]")

# Converts token predictions into tagged text
def Reconstruct(EncIds, AttnMask, OutLogits, Emos, Threshold=ThresholdDefault, IntensityOffset=0):
    # Converts the ""beginning, inside, and outside" results into a probabilities list
    BioProbs = OutLogits["Bio"].softmax(-1)[0].cpu().numpy()

    # Converts the emotion classifier results into a probabilities list.
    ClsIds = OutLogits["Cls"].argmax(-1)[0].cpu().numpy()

    # Converts the intensity classifier results into a probabilities list.
    Intens = OutLogits["Inten"][0].cpu().numpy()

    # Convert token IDs back to text-based tokens.
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
        # Convert intensity probability into a 1-9 scale, taking into account the intensity offset from the selected tone.
        Intensity = int(max(1, min(9, round(IntVal * 8 + 1 + IntensityOffset))))

        # If the probability that this word starts a span is above the threshold, checks if the span should change.
        if PB >= Threshold:
            # Defaults to "should not change", unless the below is met.
            should_change = False

            # If no tag exists, it should change.
            if PrevEmo is None:
                should_change = True

            # If the new emotion is different to the previous emotion, and either the intensity is 2 or more difference or the probability change is above the tag change delta
            # then it should change.
            elif Emo != PrevEmo:
                if abs(Intensity - (PrevInten or 0)) >= 2 or (PB - PrevPB) > TagChangeDelta:
                    should_change = True

            # If it should change and an emotion tag exists, this adds the end tag. If the new emotion isn't neutral, it then inserts an emotion tag, with calculated intensity.
            if should_change:
                if PrevEmo:
                    TextOut.append(f"<<\\{PrevEmo}>>")
                if Emo != "NEUTRAL":
                    TextOut.append(f"<<{Emo} Intensity={Intensity}>>")
                    PrevEmo, PrevInten, PrevPB = Emo, Intensity, PB
                    BCount += 1

        # If the probability that this word is inside a span is above the threshold, adds 1 to the inside count.
        elif PI >= Threshold:
            ICount += 1

        # Appends the word to the TextOut list, with a space.
        TextOut.append(w + " ")

    # After all words have been assessed, if an emotion tag has opened, adds a closing tag.
    if PrevEmo:
        TextOut.append(f"<<\\{PrevEmo}>>")

    # Combines the text output list into a tagged text string, cleans up punctuation, and returns it along with diagnostic metadata.
    Txt = "".join(TextOut)
    Txt = re.sub(r"\s+([.,!?;:])", r"\1", Txt)
    Txt = re.sub(r"\s{2,}", " ", Txt).strip()
    Txt = re.sub(r"<<Neutral Intensity=\d+>>", "", Txt)
    Txt = re.sub(r"<<\\Neutral>>", "", Txt)
    return Txt, BCount, ICount

# Run the emotional tagging process on a single block of text.
def ProcessSingleText(RawText, Tone="Neutral"):
    # Preprocess the text - collapse multiple spaces to single, remove preexisting tags.
    Txt = PreprocessInput(RawText)

    # From the provided tone (persona), calculates the threshold for emotion tags, and the intensity offset.
    Tone = Tone.lower().strip()
    if Tone == "unemotional":
        # If the persona is "unemotional", no tags should be added - returns unedited text without running the model.
        print("[Mode: Unemotional] Returning unedited text.")
        return {"text_tagged": Txt, "b_tags": 0, "i_tags": 0, "threshold": None}
    elif Tone == "calm":
        # If the persona is "calm", the threshold is increased by 10% (less probability of added tags), and intensity of emotions is reduced by 2.
        threshold = ThresholdDefault + 0.10
        intensity_offset = -2
    elif Tone == "dramatic":
        # If the persona is "dramatic", the threshold is decreased by 10% (more probability of added tags), and intensity of emotions is increased by 2.
        threshold = ThresholdDefault - 0.10
        intensity_offset = +2
    else:
        # If the persona is "neutral", or anything else, the threshold and intensity are unchanged from defaults.
        threshold = ThresholdDefault
        intensity_offset = 0

    # Encode the text using the saved tokeniser.
    Enc = Tok(Txt, return_tensors="pt", truncation=True, max_length=MaxTokens).to(Device)

    # Run the text through the model, saving the output.
    with torch.no_grad():
        Outputs = Model(Enc["input_ids"], Enc["attention_mask"])

    # Reconstruct the tagged text using the model output, emotion list, threshold, and intensity modifier calculated earlier.
    Text, BCount, ICount = Reconstruct(
        Enc["input_ids"], Enc["attention_mask"], Outputs, Emos,
        Threshold=threshold, IntensityOffset=intensity_offset
    )

    # Return the tagged text, with diagnostic metadata
    return {
        "text_tagged": Text.strip(),
        "b_tags": BCount,
        "i_tags": ICount,
        "threshold": threshold,
        "tone": Tone
    }

# Function to process a list of text, using the ProcessSingleText function above on every entry.
def ProcessTextArray(TextList, Tone="Neutral"):
    Results = []
    for Txt in TextList:
        Results.append(ProcessSingleText(Txt, Tone=Tone))
    return Results
