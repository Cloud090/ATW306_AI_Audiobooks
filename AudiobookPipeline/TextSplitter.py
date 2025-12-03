# Text splitting utility for preparing long input text for the audiobook pipeline.
# Removes metadata tags and breaks text into manageable chunks based on a token limit.
# Splits in stages: paragraphs -> sentences -> words, ensuring each chunk stays under the limit.
# Designed to keep segments small enough for emotion tagging and TTS processing to remain stable.

import re
import nltk
from transformers import AutoTokenizer


# Download tokenizers with quiet mode to avoid clutter
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Cache tokenizers to avoid redundant loading
def GetTokenizer(ModelName: str = "roberta-base"):
    if not hasattr(GetTokenizer, "_Cache"):
        GetTokenizer._Cache = {}
    if ModelName not in GetTokenizer._Cache:
        GetTokenizer._Cache[ModelName] = AutoTokenizer.from_pretrained(ModelName)
    return GetTokenizer._Cache[ModelName]


# Tokenize text using cached tokenizer
def Tokenize(Text: str, ModelName: str = "roberta-base"):
    Tokenizer = GetTokenizer(ModelName)
    return Tokenizer.tokenize(Text)


# Count the number of tokens a string produces
def CountTokens(Text: str, ModelName: str = "roberta-base") -> int:
    return len(Tokenize(Text, ModelName))


# Split text into chunks based on token limit usiing roberta-base.
def SplitText(Text: str, Limit: int = 400, ModelName: str = "roberta-base"):

    # Clean unwanted metadata tags like <<PERSONA=...>> or <<SFXSTYLE=...>>
    CleanText = re.sub(r"<<\s*(PERSONA|SFXSTYLE)\s*=\s*[^>]+>>", "", Text, flags=re.IGNORECASE).strip()

    print("[DEBUG] Beginning text split (no persona/SFXSTYLE tags)...\n")

    # Split by paragraphs
    Paragraphs = CleanText.split("\n")
    Chunks = []

    # Process each paragraph
    for Para in Paragraphs:
        Para = Para.strip()
        if not Para:
            continue

        # If paragraph fits within limit, add directly
        if CountTokens(Para, ModelName) <= Limit:
            Chunks.append(Para)
            continue

        # Split into sentences and build chunks within limit
        Sentences = nltk.sent_tokenize(Para)
        CurrentChunk = ""
        CurrentTokens = 0

        # Process each sentence
        for Sent in Sentences:
            SentTokens = CountTokens(Sent, ModelName)

            # If single sentence exceeds limit, split by words
            if SentTokens > Limit:
                Words = Sent.split()
                SubChunk = ""
                SubTokens = 0

                # Process each word
                for W in Words:
                    WTokens = CountTokens(W, ModelName)

                    # If new word exceeds limit, start new chunk
                    if SubTokens + WTokens > Limit:
                        Chunks.append(SubChunk.strip())
                        SubChunk = W + " "
                        SubTokens = WTokens
                    else:
                        SubChunk += W + " "
                        SubTokens += WTokens

                # Add final leftover word chunk
                if SubChunk.strip():
                    Chunks.append(SubChunk.strip())
                continue

            # If adding sentence exceeds the limit, finalise current chunk
            if CurrentTokens + SentTokens > Limit:
                Chunks.append(CurrentChunk.strip())
                CurrentChunk = Sent + " "
                CurrentTokens = SentTokens
            else:
                CurrentChunk += Sent + " "
                CurrentTokens += SentTokens

        # Add any remaining chunk from sentence loop
        if CurrentChunk.strip():
            Chunks.append(CurrentChunk.strip())

    # Clean and print results
    print(f"[DEBUG] Final chunk count: {len(Chunks)}")
    for i, c in enumerate(Chunks, 1):
        preview = c.replace("\n", " ")[:100]
        print(f"  [Chunk {i:02}] {preview}...")
    print("")

    return Chunks
