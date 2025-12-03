# Merges emotion tags with background and foreground SFX tags into one final tagged text block.
# Ensures tags appear in the correct order without breaking the underlying text.
# Cleans up common UTF-8 artifacts and whitespace afterwards.

import re

# Fix various UTF-8 artifacts that appear in text
def FixUTF8Artifacts(text: str) -> str:
    if not text:
        return text

    # Known broken character sequences and their fixes
    replacements = {
        "â": "’",
        "âœ": "“",
        "â": "”",
        "â": "—",
        "â": "–",
        "â¦": "…",

        "âĢĻ": "—",
        "âĢ¦": "…",
        "âĢĶ": "–",
        "âĢľ": "“",
        "âĢĿ": "”",
        "âĢĵ": "’",
    }

    # Apply replacements
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove any leftover "â" followed by random bytes
    text = re.sub(r"â[\u0100-\uffff]{1,3}", "", text)

    return text


# Spacing cleanup
# Remove double spaces and spaces before punctuation
def CleanWhitespace(text: str) -> str:
    # Remove double spaces
    text = re.sub(r"\s{2,}", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    return text.strip()

# Combine emotion, background, and foreground tags into a single text stream
# Prioritises emotion tags, then background/foreground tags
# Merges tags without disrupting the main text flow
def CombineTags(emotion_text: str, bg_text: str, fg_text: str):
    merged = []
    i = j = k = 0
    len_e, len_b, len_f = len(emotion_text), len(bg_text), len(fg_text)

    # Iterate through all texts until all are fully processed
    while i < len_e or j < len_b or k < len_f:

        ce = emotion_text[i] if i < len_e else ""
        cb = bg_text[j] if j < len_b else ""
        cf = fg_text[k] if k < len_f else ""

        # Background Tags
        if cb == "<" and bg_text[j:j+2] == "<<" and not emotion_text[i:i+2] == "<<":
            tag_end = bg_text.find(">>", j)
            if tag_end != -1:
                merged.append(bg_text[j:tag_end+2])
                j = tag_end + 2
                continue

        # Foreground Tags
        if cf == "<" and fg_text[k:k+2] == "<<" and not emotion_text[i:i+2] == "<<":
            tag_end = fg_text.find(">>", k)
            if tag_end != -1:
                merged.append(fg_text[k:tag_end+2])
                k = tag_end + 2
                continue

        # Emotion tags
        if ce == "<" and emotion_text[i:i+2] == "<<":
            tag_end = emotion_text.find(">>", i)
            if tag_end != -1:
                merged.append(emotion_text[i:tag_end+2])
                i = tag_end + 2
                continue

        # Pull next character from emotion text
        if i < len_e:
            merged.append(ce)
            i += 1

        # Advance background/foreground indices to stay in sync with emotion text
        if j < len_b:
            j += 1
        if k < len_f:
            k += 1

    # Final text cleanup
    result = "".join(merged)
    result = FixUTF8Artifacts(result)
    result = CleanWhitespace(result)

    return {"combined_text": result, "length": len(result)}
