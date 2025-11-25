import re
import nltk
from transformers import AutoTokenizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

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
    # Clean unwanted metadata tags
    CleanText = re.sub(r"<<\s*(PERSONA|SFXSTYLE)\s*=\s*[^>]+>>", "", Text, flags=re.IGNORECASE).strip()

    print("[DEBUG] Beginning text split (no persona/SFXSTYLE tags)...\n")

    # Split by paragraphs
    Paragraphs = CleanText.split("\n")
    Chunks = []

    for Para in Paragraphs:
        Para = Para.strip()
        if not Para:
            continue

        if CountTokens(Para, ModelName) <= Limit:
            Chunks.append(Para)
            continue

        # Split into sentences
        Sentences = nltk.sent_tokenize(Para)
        CurrentChunk = ""
        CurrentTokens = 0

        for Sent in Sentences:
            SentTokens = CountTokens(Sent, ModelName)
            if SentTokens > Limit:
                # Sentence itself too long → split by words
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

    # Clean and print results
    print(f"[DEBUG] Final chunk count: {len(Chunks)}")
    for i, c in enumerate(Chunks, 1):
        preview = c.replace("\n", " ")[:100]
        print(f"  [Chunk {i:02}] {preview}...")
    print("")

    return Chunks
