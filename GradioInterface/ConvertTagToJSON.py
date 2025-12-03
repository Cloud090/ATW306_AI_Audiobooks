# This module contains functions to convert text with emotion tags into a JSON format usable by the TextToSpeech module.

import re
import json

# The text to speech functions use "int_lo", "int_md" and "int_hi" tags instead of numeric intensity levels. This functions converts the intensity level to the 
# appropriate tag - 1-3 returns "int_lo", 4-6 returns "int_md", 7-9 returns "int_hi".
def intensity_to_tag(v: int) -> str:
    if v <= 3:
        return "int_lo"
    elif v <= 6:
        return "int_md"
    else:
        return "int_hi"

# This function takes emotion tagged text and a speaker id and returns a dictionary list, with the text split into individual lines separated into single emotions (as the text to speech
# model can only generate one emotion at a time). 
def _convert_to_segments(input_text: str, speaker_id: str):
    # Regex to find opening and closing emotion tags
    token_re = re.compile(
        r"<<\s*(?P<open_emo>[A-Za-z]+)\s+Intensity=(?P<int>\d+)\s*>>"
        r"|<<\\\s*(?P<close_emo>[A-Za-z]+)\s*>>",
        re.DOTALL
    )

    # Variables to help separate tagged text - current position, results list, current emotion, current intensity.
    pos = 0
    results = []

    current_emotion = None
    current_intensity = None

    # Iterate through all tags found by the above regex.
    for match in token_re.finditer(input_text):
        # Gets start and end index of current tag match.
        start, end = match.span()

        # Extract the text from the current position to the start of the matched tag (i.e. all unprocessed text before the match).
        text_chunk = input_text[pos:start]

        # Sets current position to the end of the matched tag.
        pos = end

        # If the retrieved text isn't empty
        if text_chunk.strip():
            # If no current emotion exists, adds an entry with a "<neutral>" tag and a speaker id tag.
            if current_emotion is None:
                results.append({
                    "text": f"<spk={speaker_id}> <neutral> {text_chunk.strip()}"
                })
            # If a current emotion does exist, adds an entry with a speaker id tag, and emotion tag, and an intensity tag. 
            else:
                int_tag = intensity_to_tag(current_intensity)
                results.append({
                    "text": f"<spk={speaker_id}> <{current_emotion.lower()}> <{int_tag}> {text_chunk.strip()}"
                })

        # Checks if the current tag match is an open emotion tag, or a closing tag.
        open_emo = match.group("open_emo")
        close_emo = match.group("close_emo")

        # If it's an open emotion tag, sets the current emotion to the tag emotion, and the current intensity to the corresponding intensity tag ("int_lo" for 1-3, "int_md" for 4-6, 
        # "int_hi" for 7-9).
        if open_emo:
            current_emotion = open_emo
            current_intensity = int(match.group("int"))

        # If it's a closing tag, sets current emotion and intensity to "None".
        elif close_emo:
            current_emotion = None
            current_intensity = None

    # Process the trailing text after the last tag match, if any exists.
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

    # Returns the results
    return results


# Convert single text to JSON format, using the above _convert_to_segments function.
def convert_tagged_text(input_text: str, speaker_id: str = "1001") -> str:
    segments = _convert_to_segments(input_text, speaker_id)
    return json.dumps(segments, indent=2, ensure_ascii=False)


# Convert a text list to JSON format, using the _convert_to_segments function.
def convert_list_of_tagged_texts(list_of_strings, speaker_id: str = "1001"):
    final = []
    for block in list_of_strings:
        segments = _convert_to_segments(block, speaker_id)
        final.append(segments)
    return final
