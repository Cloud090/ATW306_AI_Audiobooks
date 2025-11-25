import re
import json


def intensity_to_tag(v: int) -> str:
    # Convert intensity (1–9) into int_lo / int_md / int_hi.
    if v <= 3:
        return "int_lo"
    elif v <= 6:
        return "int_md"
    else:
        return "int_hi"

def _convert_to_segments(input_text: str, speaker_id: str):
    # Regex to catch open/close emotion tags
    token_re = re.compile(
        r"<<\s*(?P<open_emo>[A-Za-z]+)\s+Intensity=(?P<int>\d+)\s*>>"
        r"|<<\\\s*(?P<close_emo>[A-Za-z]+)\s*>>",
        re.DOTALL
    )

    pos = 0
    results = []

    current_emotion = None
    current_intensity = None

    for match in token_re.finditer(input_text):
        start, end = match.span()

        # Text before this tag
        text_chunk = input_text[pos:start]
        pos = end

        if text_chunk.strip():
            if current_emotion is None:
                # Neutral line
                results.append({
                    "text": f"<spk={speaker_id}> <neutral> {text_chunk.strip()}"
                })
            else:
                # Emotion line
                int_tag = intensity_to_tag(current_intensity)
                results.append({
                    "text": f"<spk={speaker_id}> <{current_emotion.lower()}> <{int_tag}> {text_chunk.strip()}"
                })

        # Process tag
        open_emo = match.group("open_emo")
        close_emo = match.group("close_emo")

        if open_emo:
            current_emotion = open_emo
            current_intensity = int(match.group("int"))

        elif close_emo:
            current_emotion = None
            current_intensity = None

    # Trailing text
    tail = input_text[pos:].strip()
    if tail:
        if current_emotion is None:
            results.append({
                "text": f"<spk={speaker_id}> <neutral> {tail}"
            })
        else:
            int_tag = intensity_to_tag(current_intensity)
            results.append({
                "text": f"<spk={speaker_id}> <{current_emotion.lower()}> <{int_tag}> {tail}"
            })

    return results


#  Convert single text to JSON

def convert_tagged_text(input_text: str, speaker_id: str = "1001") -> str:
    segments = _convert_to_segments(input_text, speaker_id)
    return json.dumps(segments, indent=2, ensure_ascii=False)


#  Convert text list to JSON

def convert_list_of_tagged_texts(list_of_strings, speaker_id: str = "1001"):
    final = []
    for block in list_of_strings:
        segments = _convert_to_segments(block, speaker_id)
        final.append(segments)
    return final
