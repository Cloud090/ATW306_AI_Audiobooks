import gradio as gr
import time
import os
import threading
import requests
import json

import RenderLink
import TextSplitter
import EmotionTagger
import BGSFXTagger
import FGSFXTagger
import TagCombiner
import ConvertTagToJSON
import TextToSpeech


CurrentStatus = "Idle"

def SetStatus(message: str):
    global CurrentStatus
    CurrentStatus = message
    print(f"[STATUS] {message}")


def GetStatus():
    return {"status": CurrentStatus}

def GenerateAudiobook(text, persona, sfxStyle, voiceAudio=None, voiceName=None):
    print("\n\nNew Audiobook Generation\n")
    print(f"Persona: {persona}")
    print(f"SFX Style: {sfxStyle}")
    print(f"Voice Name (speaker id): {voiceName}\n")

    valid_tones = ["Unemotional", "Calm", "Neutral", "Dramatic"]
    tone = persona.strip() if persona in valid_tones else "Neutral"

    valid_sfx_styles = ["None", "Subtle", "Balanced", "Cinematic"]
    sfx_style = sfxStyle.strip().title() if sfxStyle.title() in valid_sfx_styles else "Balanced"

    # Split text
    SetStatus("Splitting text...")
    yield ({"status": "running", "message": "Splitting text..."}, None)

    try:
        chunks = TextSplitter.SplitText(text, Limit=400, ModelName="roberta-base")
        SetStatus(f"Text split — {len(chunks)} sections.")
        yield ({"status": "running", "message": f"Split into {len(chunks)} chunks."}, None)
    except Exception as e:
        SetStatus(f"Split failed: {e}")
        yield ({"status": "error", "message": f"Split failed: {e}"}, None)
        return

    if not chunks:
        SetStatus("No chunks generated.")
        yield ({"status": "error", "message": "No chunks generated."}, None)
        return

    # Taggers
    SetStatus("Running taggers...")
    yield ({"status": "running", "message": "Running all taggers..."}, None)

    combined_tagged_texts = []
    emotion_tagged_texts = []

    total_emo_tags = 0
    total_bg_tags = 0
    total_fg_tags = 0

    for idx, chunk in enumerate(chunks, start=1):
        emo_result = EmotionTagger.ProcessSingleText(chunk, Tone=tone)
        emo_text = emo_result.get("text_tagged", "")
        emotion_tagged_texts.append(emo_text)
        total_emo_tags += emo_result.get("b_tags", 0) + emo_result.get("i_tags", 0)

        bg_result = BGSFXTagger.ProcessSingleText(chunk, SFXStyle=sfx_style)
        total_bg_tags += bg_result.get("b_tags", 0)

        fg_result = FGSFXTagger.ProcessSingleText(chunk, SFXStyle=sfx_style)
        total_fg_tags += fg_result.get("b_tags", 0)

        combined_result = TagCombiner.CombineTags(
            emo_text,
            bg_result.get("text_tagged", ""),
            fg_result.get("text_tagged", "")
        )
        combined_tagged_texts.append(combined_result.get("combined_text", ""))

    SetStatus("Taggers complete.")
    yield (
        {
            "status": "running",
            "message": (
                f"Taggers complete — Emo: {total_emo_tags}, "
                f"BGSFX: {total_bg_tags}, FGSFX: {total_fg_tags}"
            ),
        },
        None
    )

    # Convert emotion tags to JSON
    SetStatus("Converting to JSON for TTS")
    yield ({"status": "running", "message": "Formatting emotion tags..."}, None)

    speaker_id = voiceName if voiceName else "1002"

    try:
        converted_json_blocks = ConvertTagToJSON.convert_list_of_tagged_texts(
            emotion_tagged_texts, speaker_id=speaker_id
        )

        yield (
            {
                "status": "running",
                "message": "Converted to Orpheus JSON.",
                "orpheus_input": converted_json_blocks,
            },
            None
        )

    except Exception as e:
        SetStatus(f"Conversion failed: {e}")
        yield ({"status": "error", "message": f"Conversion failed: {e}"}, None)
        return

    # Generate TTS
    SetStatus("Generating full narration...")
    yield ({"status": "running", "message": "Generating narration via Orpheus..."}, None)

    output_audio_path = None

    try:
        all_segments = []
        for block in converted_json_blocks:
            all_segments.extend(block)

        output_root = "tts_outputs"
        os.makedirs(output_root, exist_ok=True)

        run_id = int(time.time())
        tts_folder = os.path.join(output_root, f"run_{run_id}")
        os.makedirs(tts_folder, exist_ok=True)

        TextToSpeech.speak_json(all_segments, output_folder=tts_folder)

        combined_path = os.path.join(tts_folder, "combined.wav")
        if os.path.isfile(combined_path):
            output_audio_path = combined_path

        yield (
            {
                "status": "running",
                "message": "Full-book narration generated.",
                "tts_output_folder": tts_folder,
                "output_audio": output_audio_path,
            },
            None
        )

    except Exception as e:
        SetStatus(f"TTS failed: {e}")
        yield ({"status": "error", "message": f"TTS failed: {e}"}, None)
        return

    # Finish
    SetStatus("Completed.")
    print("Audiobook generation completed.\n")

    # Audio output
    yield (
        {
            "status": "success",
            "message": "Completed audiobook generation.",
            "chunk_count": len(chunks),
            "tagged_texts": combined_tagged_texts,
            "emotion_tagged_texts": emotion_tagged_texts,
            "orpheus_formatted": converted_json_blocks,
            "output_audio": output_audio_path,
        },
        output_audio_path
    )


# Gradio & Render

def WaitForShareUrl(apiInterface, timeout=60):
    for _ in range(timeout // 2):
        url = getattr(apiInterface, "share_url", None) or getattr(apiInterface, "share_link", None)
        if url and url.startswith("http"):
            return url
        time.sleep(2)
    return None


def KeepAliveLoop(renderBase="https://audioapi-g2ru.onrender.com"):
    while True:
        try:
            requests.get(f"{renderBase}/current", timeout=10)
        except:
            pass
        time.sleep(600)


print("Launching Emotional Audiobook API")

ApiMain = gr.Interface(
    fn=GenerateAudiobook,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Textbox(label="Persona"),
        gr.Textbox(label="SFX Style"),
        gr.Audio(type="filepath", label="Voice Sample (optional)"),
        gr.Textbox(label="Voice Name (speaker id)"),
    ],
    outputs=[
        gr.JSON(label="Streaming API Output"),
        gr.Audio(label="Final Audio", type="filepath")
    ],
    title="Emotional Audiobook API",
    description="Generates emotion-tagged audiobook audio using Orpheus.",
    live=True,
)

ApiStatus = gr.Interface(fn=GetStatus, inputs=[], outputs=gr.JSON(label="Current Status"))

App = gr.TabbedInterface([ApiMain, ApiStatus], ["Generate", "Status"])

App.launch(share=True, prevent_thread_lock=True, inline=False)

ShareUrl = WaitForShareUrl(App)
if ShareUrl:
    RenderLink.SetGradioURL(ShareUrl)

thread = threading.Thread(target=KeepAliveLoop, daemon=True)
thread.start()

print("Server running...")
while True:
    time.sleep(60)
