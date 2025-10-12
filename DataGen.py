import tkinter as Tk
from tkinter import messagebox
import random
import os
import time
import google.generativeai as GenAI
import re
import string
import difflib

def LoadFile(FilePath):
    if not os.path.exists(FilePath):
        return f"[Error: {FilePath} not found]"
    with open(FilePath, "r", encoding="utf-8") as F:
        return F.read().strip()

def LoadFileLines(FilePath):
    if not os.path.exists(FilePath):
        return [f"[Error: {FilePath} not found]"]
    with open(FilePath, "r", encoding="utf-8") as F:
        return [Line.strip() for Line in F if Line.strip()]

def LoadAPIKey():
    try:
        with open("GeminiKey.txt", "r", encoding="utf-8") as F:
            return F.read().strip()
    except FileNotFoundError:
        messagebox.showerror("Error", "GeminiKey.txt not found. Please create it with your API key.")
        return None

def GeneratePrompt(NumOutputs=5):
    Description = LoadFile("Description.txt")
    Personas = LoadFile("Personas.txt")
    SFXStyles = LoadFile("SFXStyle.txt")
    Emotions = LoadFile("Emotions.txt")
    FGSFX = LoadFile("ForegroundSFX.txt")
    BGSFX = LoadFile("BackgroundSFX.txt")
    Genres = LoadFileLines("Genres.txt")
    Genre = random.choice(Genres) if Genres else "Unknown"
    PromptParts = [
        Description,
        f"\n\nAvailable personas:\n{Personas}",
        f"\n\nAvailable SFX Styles:\n{SFXStyles}",
        f"\n\nAvailable emotions:\n{Emotions}",
        f"\n\nAvailable foreground SFX:\n{FGSFX}",
        f"\n\nAvailable background SFX:\n{BGSFX}",
        f"\n\nPlease create a two paragraph input in the {Genre} genre. "
        f"Then create {NumOutputs} tagged outputs using varied combinations of the personas and SFX styles provided."
    ]
    FinalPrompt = "\n".join(PromptParts)
    with open("TestPrompt.txt", "w", encoding="utf-8") as F:
        F.write(FinalPrompt)
    return FinalPrompt, Genre

def RequestTest():
    try:
        NumOutputs = int(SpinboxOutputs.get())
        FinalPrompt, Genre = GeneratePrompt(NumOutputs)
        APIKey = LoadAPIKey()
        if not APIKey:
            return

        GenAI.configure(api_key=APIKey)
        Model = GenAI.GenerativeModel("gemma-3-27b-it")

        print(f"Requesting response... (Genre: {Genre})")
        StartTime = time.time()
        Response = Model.generate_content(FinalPrompt)
        EndTime = time.time()
        Elapsed = round(EndTime - StartTime, 2)

        with open("TestResponse.txt", "w", encoding="utf-8") as F:
            F.write(Response.text if Response and Response.text else "[No response]")

        print(f"Response saved to TestResponse.txt | Genre: {Genre} | Time taken: {Elapsed} seconds\n")

    except Exception as E:
        print(f"Error during request: {E}")


def CheckIOPairs():
    if not os.path.exists("TestResponse.txt"):
        raise FileNotFoundError("TestResponse.txt not found.")
    with open("TestResponse.txt", "r", encoding="utf-8") as F:
        Content = F.read().strip()
    Sections = [S.strip() for S in Content.split('---') if S.strip()]
    if len(Sections) < 2:
        raise ValueError("Response too short — expected at least one Input and one Output section.")
    ExpectedIsInput = True
    for I, Section in enumerate(Sections):
        HeaderLine = Section.split("\n", 1)[0].strip().upper()
        IsInput = HeaderLine.startswith("INPUT") or "PERSONA" in Section.split("\n", 3)[0]
        IsOutput = HeaderLine.startswith("OUTPUT")
        if ExpectedIsInput and not IsInput:
            raise ValueError(f"Section {I+1} expected to be Input but appears to be Output.")
        if not ExpectedIsInput and not IsOutput:
            raise ValueError(f"Section {I+1} expected to be Output but appears to be Input.")
        ExpectedIsInput = not ExpectedIsInput
    if len(Sections) % 2 != 0:
        raise ValueError(f"Found {len(Sections)} total sections — must be an even number (Input/Output pairs).")
    if any(len(S) < 10 for S in Sections):
        raise ValueError("One or more sections appear empty or too short.")
    return Sections

def CleanText(Text):
    Text = re.sub(r"<<.*?>>", "", Text)
    Text = Text.translate(str.maketrans("", "", string.punctuation))
    Text = Text.lower()
    Text = re.sub(r"\s+", " ", Text).strip()
    return Text

def CheckEmotionTags(Sections):
    if not os.path.exists("Emotions.txt"):
        raise FileNotFoundError("Emotions.txt not found.")
    with open("Emotions.txt", "r", encoding="utf-8") as F:
        ValidEmotions = [Line.strip().upper() for Line in F if Line.strip()]
    Errors = []
    TotalTags = 0
    OpenPattern = r"<<([A-Z]+)\s+Intensity=(\d+)>>"
    IgnorePrefixes = ("FGSFX ", "BGSFX ", "PERSONA ", "SFXSTYLE ")
    for I, Section in enumerate(Sections):
        IsOutput = (I % 2 == 1)
        AllTags = re.findall(r"<<([^>]+)>>", Section)
        AllTags = [T for T in AllTags if not T.startswith(IgnorePrefixes)]
        OpenTags = [
            M for M in re.findall(OpenPattern, Section)
            if M[0] not in ("FGSFX", "BGSFX", "PERSONA", "SFXSTYLE")
        ]
        TotalTags += len(OpenTags)
        if not IsOutput:
            EmotionTagsInInput = [T for T in AllTags if "INTENSITY=" in T]
            if EmotionTagsInInput:
                Errors.append(f"Emotion-like tags found in Input section {I+1}: {', '.join(EmotionTagsInInput[:3])}")
            continue
        for Emotion, Intensity in OpenTags:
            if Emotion not in ValidEmotions:
                Errors.append(f"Invalid emotion tag '{Emotion}' in Output {I+1}.")
            if not Intensity.isdigit():
                Errors.append(f"Non-numeric intensity '{Intensity}' in Output {I+1} for {Emotion}.")
        Stack = []
        for Match in re.finditer(r"<<(\\)?([A-Z]+).*?>>", Section):
            IsClose, Emotion = Match.groups()
            if Emotion in ("FGSFX", "BGSFX", "PERSONA", "SFXSTYLE"):
                continue
            if Emotion not in ValidEmotions:
                continue
            if IsClose:
                if not Stack or Stack[-1] != Emotion:
                    Errors.append(f"Unmatched closing tag for {Emotion} in Output {I+1}.")
                else:
                    Stack.pop()
            else:
                Stack.append(Emotion)
        if Stack:
            for Emotion in Stack:
                Errors.append(f"Unclosed tag for {Emotion} in Output {I+1}.")
        if len(OpenTags) == 0:
            Errors.append(f"No emotion tags found in Output {I+1} (possible missing annotation).")
    return Errors, TotalTags

def CheckSFXTags(Sections):
    if not os.path.exists("ForegroundSFX.txt") or not os.path.exists("BackgroundSFX.txt"):
        raise FileNotFoundError("ForegroundSFX.txt or BackgroundSFX.txt not found.")

    with open("ForegroundSFX.txt", "r", encoding="utf-8") as F:
        ValidFG = [Line.strip().lower() for Line in F if Line.strip()]
    with open("BackgroundSFX.txt", "r", encoding="utf-8") as F:
        ValidBG = [Line.strip().lower() for Line in F if Line.strip()]

    Errors = []
    FGPattern = r"<<FGSFX\s+Sound=(.*?)>>"
    BGOpenPattern = r"<<BGSFX\s+Sound=(.*?)>>"
    BGClosePattern = r"<<\\BGSFX>>"

    for I, Section in enumerate(Sections):
        IsOutput = (I % 2 == 1)
        if not IsOutput:
            continue

        # --- Foreground SFX checks ---
        FGTags = re.findall(FGPattern, Section)
        for Sound in FGTags:
            SoundClean = Sound.strip().lower()
            if SoundClean not in ValidFG:
                Errors.append(f"Invalid foreground sound '{Sound}' in Output {I+1}.")
        # Ensure no closing tag for FGSFX
        if re.search(r"<<\\FGSFX>>", Section):
            Errors.append(f"Incorrect closing tag <<\\FGSFX>> found in Output {I+1} (FGSFX should be standalone).")

        # --- Background SFX checks ---
        BGOpens = re.findall(BGOpenPattern, Section)
        BGCloses = re.findall(BGClosePattern, Section)
        if len(BGOpens) != len(BGCloses):
            Errors.append(f"Mismatched BGSFX open/close tags in Output {I+1}: {len(BGOpens)} open, {len(BGCloses)} close.")
        # Validate background names
        for Sound in BGOpens:
            SoundClean = Sound.strip().lower()
            if SoundClean not in ValidBG:
                Errors.append(f"Invalid background sound '{Sound}' in Output {I+1}.")
    return Errors


def CheckTextSimilarity(Sections):
    Errors = []
    for I in range(0, len(Sections), 2):
        if I + 1 >= len(Sections):
            continue

        InputRaw = Sections[I]
        OutputRaw = Sections[I + 1]

        # Remove section headers like "Input:" or "Output:"
        InputRaw = re.sub(r"^\s*Input:\s*", "", InputRaw, flags=re.IGNORECASE)
        OutputRaw = re.sub(r"^\s*Output:\s*", "", OutputRaw, flags=re.IGNORECASE)

        InputText = CleanText(InputRaw)
        OutputText = CleanText(OutputRaw)

        if InputText != OutputText:
            print("\n--- TEXT MISMATCH DETECTED ---")
            print(f"Input {I//2 + 1} (Cleaned):")
            print(InputText[:1000] + ("..." if len(InputText) > 1000 else ""))
            print()
            print(f"Output {I//2 + 1} (Cleaned):")
            print(OutputText[:1000] + ("..." if len(OutputText) > 1000 else ""))
            print("-" * 60)

            InputWords = InputText.split()
            OutputWords = OutputText.split()
            Diff = list(difflib.ndiff(InputWords, OutputWords))
            Added = [W[2:] for W in Diff if W.startswith("+ ")]
            Removed = [W[2:] for W in Diff if W.startswith("- ")]

            if Added or Removed:
                AddedPreview = " ".join(Added[:10])
                RemovedPreview = " ".join(Removed[:10])
                print(f"   + Added sample: {AddedPreview}")
                print(f"   - Removed sample: {RemovedPreview}")
                print(f"   Total added: {len(Added)}, removed: {len(Removed)}")
                print("-" * 60)

            ErrorMsg = f"Text mismatch between Input {I//2 + 1} and Output {I//2 + 1}."
            Errors.append(ErrorMsg)

    return Errors

def ResponseCheck():
    try:
        Sections = CheckIOPairs()
        Errors, TotalTags = CheckEmotionTags(Sections)
        SFXErrors = CheckSFXTags(Sections)
        SimilarityErrors = CheckTextSimilarity(Sections)
        
        AllErrors = Errors + SFXErrors + SimilarityErrors

        print("\n" + "=" * 70)
        print("RESPONSE VALIDATION REPORT")
        print("=" * 70)
        print(f"Total Sections: {len(Sections)}")
        print(f"Expected I/O Pairs: {len(Sections)//2}")
        print(f"Detected Emotion Tags: {TotalTags}")
        print("-" * 70)
        if AllErrors:
            print(f"{len(AllErrors)} issue(s) detected:\n")
            for Index, Error in enumerate(AllErrors, 1):
                Context = ""
                Match = re.search(r"section (\d+)", Error)
                if Match:
                    SectionNum = int(Match.group(1))
                    if 0 <= SectionNum - 1 < len(Sections):
                        SectionText = Sections[SectionNum - 1]
                        Preview = SectionText[:200].replace("\n", " ")
                        Context = f"Context (first 200 chars): {Preview}..."
                print(f"{Index}. {Error}")
                if Context:
                    print(f"   {Context}\n")
            print("-" * 70)
            print("Review these issues before using this dataset output.\n")
        else:
            print("Response passed all validation checks.")
            print(f"All {len(Sections)//2} Input/Output pairs are correctly structured.")
            print(f"All emotion tags and base text are valid.\n")
        print("=" * 70 + "\n")
    except FileNotFoundError as E:
        messagebox.showerror("Error", str(E))
    except ValueError as E:
        print(f"\nStructural issue detected:\n{E}\n")
    except Exception as E:
        print(f"\nValidation process failed:\n{E}\n")

def AutoRequestLoop():
    TimeValue = EntryTime.get().strip()
    try:
        H, M = map(int, TimeValue.split(":"))
        RunSeconds = H * 3600 + M * 60
    except Exception:
        print("Invalid time format. Please use hh:mm (e.g., 00:10).")
        return

    StartTime = time.time()
    EndTime = StartTime + RunSeconds
    TotalRuns = 0
    SuccessfulPairs = 0

    print(f"\nStarting automated request loop for {H}h {M}m ({RunSeconds} seconds)\n")

    try:
        while time.time() < EndTime:
            TotalRuns += 1
            print(f"\n===== RUN {TotalRuns} START =====")
            RequestTest()
            print("Validating response by pair...")

            try:
                Sections = CheckIOPairs()
                PairCount = len(Sections) // 2
                print(f"Detected {PairCount} input/output pairs.")
                ValidPairs = []

                for i in range(0, len(Sections), 2):
                    InputSection = Sections[i]
                    OutputSection = Sections[i + 1]
                    PairSections = [InputSection, OutputSection]

                    Errors, _ = CheckEmotionTags(PairSections)
                    SFXErrors = CheckSFXTags(PairSections)
                    SimilarityErrors = CheckTextSimilarity(PairSections)
                    PairErrors = Errors + SFXErrors + SimilarityErrors

                    if not PairErrors:
                        ValidPairs.append((InputSection, OutputSection))
                        SuccessfulPairs += 1
                        print(f"  Pair {(i//2) + 1}: Valid")
                    else:
                        print(f"  Pair {(i//2) + 1}: Invalid ({len(PairErrors)} issue(s))")

                # Append all valid pairs to dataset
                if ValidPairs:
                    with open("TaggedTextDataset.txt", "a", encoding="utf-8") as D:
                        for InSec, OutSec in ValidPairs:
                            D.write(f"{InSec}\n\n---\n\n{OutSec}\n\n" + ("=" * 70) + "\n\n")
                    print(f"Appended {len(ValidPairs)} valid pair(s) from this run.")
                else:
                    print("No valid pairs to append from this run.")

            except Exception as E:
                print(f"Validation error during run {TotalRuns}: {E}")

            Elapsed = round(time.time() - StartTime, 2)
            print(f"Elapsed: {Elapsed}s / {RunSeconds}s total duration.")
            print("=" * 70)

        print("\nAutomation complete.")
        print(f"Total runs attempted: {TotalRuns}")
        print(f"Total valid input/output pairs saved: {SuccessfulPairs}\n")

    except KeyboardInterrupt:
        print("\nAutomation interrupted by user.")



Root = Tk.Tk()
Root.title("Emotional Storyteller Prompt Builder")
Root.geometry("400x500")

LabelTitle = Tk.Label(Root, text="Emotional Storyteller Dataset Tool", font=("Arial", 14))
LabelTitle.pack(pady=15)

FrameOutputs = Tk.Frame(Root)
FrameOutputs.pack(pady=5)

LabelOutputs = Tk.Label(FrameOutputs, text="Number of tagged outputs:", font=("Arial", 12))
LabelOutputs.pack(side=Tk.LEFT, padx=5)

SpinboxOutputs = Tk.Spinbox(FrameOutputs, from_=1, to=20, width=5, font=("Arial", 12))
SpinboxOutputs.delete(0, Tk.END)
SpinboxOutputs.insert(0, "5")
SpinboxOutputs.pack(side=Tk.LEFT)

FrameTime = Tk.Frame(Root)
FrameTime.pack(pady=5)

LabelTime = Tk.Label(FrameTime, text="Interval (hh:mm):", font=("Arial", 12))
LabelTime.pack(side=Tk.LEFT, padx=5)

EntryTime = Tk.Entry(FrameTime, width=7, font=("Arial", 12))
EntryTime.insert(0, "00:10")
EntryTime.pack(side=Tk.LEFT)


BtnPrompt = Tk.Button(Root, text="Test Prompt",
                      command=lambda: (GeneratePrompt(int(SpinboxOutputs.get())),
                                       messagebox.showinfo("Success", "Prompt saved to TestPrompt.txt")),
                      font=("Arial", 12))
BtnPrompt.pack(pady=10)

BtnRequest = Tk.Button(Root, text="Request Test", command=RequestTest, font=("Arial", 12))
BtnRequest.pack(pady=10)

BtnCheck = Tk.Button(Root, text="Response Check", command=ResponseCheck, font=("Arial", 12))
BtnCheck.pack(pady=10)

BtnAuto = Tk.Button(Root, text="Add to dataset", command=AutoRequestLoop, font=("Arial", 12))
BtnAuto.pack(pady=10)

Root.mainloop()
