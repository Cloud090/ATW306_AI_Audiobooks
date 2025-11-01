import re
import nltk
from transformers import AutoTokenizer

# Ensure sentence tokenizer data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def GetTokenizer(ModelName: str = "roberta-base"):
    if not hasattr(GetTokenizer, "_Cache"):
        GetTokenizer._Cache = {}
    if ModelName not in GetTokenizer._Cache:
        GetTokenizer._Cache[ModelName] = AutoTokenizer.from_pretrained(ModelName)
    return GetTokenizer._Cache[ModelName]


def Tokenize(Text: str, ModelName: str = "roberta-base"):
    Tokenizer = GetTokenizer(ModelName)
    return Tokenizer.tokenize(Text)


def CountTokens(Text: str, ModelName: str = "roberta-base") -> int:
    return len(Tokenize(Text, ModelName))


def SplitText(Text: str, Limit: int = 400, ModelName: str = "roberta-base"):
    # Extract tags
    PersonaTag = None
    SfxTag = None

    PersonaMatch = re.search(r"<<\s*PERSONA\s*=\s*([^>]+)>>", Text, re.IGNORECASE)
    SfxMatch = re.search(r"<<\s*SFXSTYLE\s*=\s*([^>]+)>>", Text, re.IGNORECASE)

    if PersonaMatch:
        PersonaTag = PersonaMatch.group(0).strip()
    if SfxMatch:
        SfxTag = SfxMatch.group(0).strip()

    # Remove tags
    CleanText = re.sub(r"<<\s*(PERSONA|SFXSTYLE)\s*=\s*[^>]+>>", "", Text, flags=re.IGNORECASE).strip()

    print(f"[DEBUG] Persona tag detected: {PersonaTag}")
    print(f"[DEBUG] SFX tag detected: {SfxTag}")
    print("[DEBUG] Beginning text split...\n")

    # Split remaining text
    Paragraphs = CleanText.split("\n")
    Chunks = []

    for Para in Paragraphs:
        Para = Para.strip()
        if not Para:
            continue

        if CountTokens(Para, ModelName) <= Limit:
            Chunks.append(Para)
            continue

        Sentences = nltk.sent_tokenize(Para)
        CurrentChunk = ""
        CurrentTokens = 0

        for Sent in Sentences:
            SentTokens = CountTokens(Sent, ModelName)
            if SentTokens > Limit:
                # Sentence too long → split by words
                Words = Sent.split()
                SubChunk = ""
                SubTokens = 0
                for W in Words:
                    WTokens = CountTokens(W, ModelName)
                    if SubTokens + WTokens > Limit:
                        Chunks.append(SubChunk.strip())
                        SubChunk = W + " "
                        SubTokens = WTokens
                    else:
                        SubChunk += W + " "
                        SubTokens += WTokens
                if SubChunk.strip():
                    Chunks.append(SubChunk.strip())
                continue

            if CurrentTokens + SentTokens > Limit:
                Chunks.append(CurrentChunk.strip())
                CurrentChunk = Sent + " "
                CurrentTokens = SentTokens
            else:
                CurrentChunk += Sent + " "
                CurrentTokens += SentTokens

        if CurrentChunk.strip():
            Chunks.append(CurrentChunk.strip())

    # Reattach tags to each chunk
    TaggedChunks = []
    for chunk in Chunks:
        Prefix = ""
        if PersonaTag:
            Prefix += PersonaTag + "\n"
        if SfxTag:
            Prefix += SfxTag + "\n"
        TaggedChunks.append(f"{Prefix}{chunk.strip()}")

    print(f"[DEBUG] Final chunk count: {len(TaggedChunks)}")
    for i, c in enumerate(TaggedChunks, 1):
        preview = c.replace("\n", " ")[:100]
        print(f"  [Chunk {i:02}] {preview}...")
    print("")

    return TaggedChunks
