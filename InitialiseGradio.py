import gradio as gr
import time
import os
import threading
import requests
import json
import sys

# Local module - Functions to connect to Render URL, setting or retrieving the Gradio API URL
import RenderLink

# Local module - Function to split text into "chunks" of no more than 400 tokens. Attempts to break by paragraph, if paragraph is too long then by sentence, if 
# sentence is too long then by word
import TextSplitter

# Local module - Runs the emotion tagging model, inserting tags in the format <<[Emotion] Intensity=[x]>> and <<\[Emotion]>>, based on context.
import EmotionTagger

# Local module – Runs the background SFX tagging model, inserting tags in the format <<BGSFX Sound=[sound]>> and <<\BGSFX>>, based on context. 
import BGSFXTagger

# Local module – Runs the foreground SFX tagging model, inserting tags in the format <<FGSFX Sound=[sound]>> (no closing tag), based on context. 
import FGSFXTagger

# Local module – Merges emotion, background, and foreground tags into a single combined tagged text string.
import TagCombiner

# Local module – Converts emotion-tagged text into JSON list format, for use in the text to speech model.
import ConvertTagToJSON

# Local module - Runs a trained text to spoken audio model, based on Orpheus.
import TextToSpeech

# Local module - Uses pythons "faster_whisper" module to transcribe the spoken audio with timestamps, then inserts sound effects at the appropriate times,
# based on the BGSFXTagger and FGSFXTagger results.
import SFXModule

# The CurrentStatus variable tracks which stage the audiobook generation pipeline is up to. The "SetStatus" and "GetStatus" functions below exist so that the 
# current pipeline status can be set or requested at any time.
CurrentStatus = "Idle"


def SetStatus(message: str):
    global CurrentStatus
    CurrentStatus = message
    print(f"Current Status:{message}")


def GetStatus():
    return {"status": CurrentStatus}

# Function to convert seconds to HH:MM format.
def format_time(seconds: float):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# This is the main audiobook pipeline function, which the Gradio API calls when a "/predict" request is received. It's provided with text, a persona (which 
# influences emotion tag placements), an sfx style (which influences sound effect placement), and a voice name (one of a selection of voices available with 
# the text to speech model), and interacts with a group of local modules to add emotion and sound effect tags to text, convert the text to spoken audio, and
# then add sound effects.
def GenerateAudiobook(text, persona, sfxStyle, voiceAudio=None, voiceName=None):
    print("\n\nNew Audiobook Generation\n")
    print(f"Persona: {persona}")
    print(f"SFX Style: {sfxStyle}")
    print(f"Voice Name: {voiceName}\n")

    valid_tones = ["Unemotional", "Calm", "Neutral", "Dramatic"]
    tone = persona.strip() if persona in valid_tones else "Neutral"

    valid_sfx_styles = ["None", "Subtle", "Balanced", "Cinematic"]
    sfx_style = (
        sfxStyle.strip().title()
        if sfxStyle.title() in valid_sfx_styles
        else "Balanced"
    )

    # Split text
    SetStatus("Splitting text")
    yield ({"status": "running", "message": "Splitting text"}, None)

    try:
        chunks = TextSplitter.SplitText(text, Limit=400, ModelName="roberta-base")
        SetStatus(f"Text split — {len(chunks)} sections.")
        yield (
            {"status": "running", "message": f"Split into {len(chunks)} chunks."},
            None,
        )
    except Exception as e:
        SetStatus(f"Split failed: {e}")
        yield ({"status": "error", "message": f"Split failed: {e}"}, None)
        return

    if not chunks:
        SetStatus("No chunks generated.")
        yield ({"status": "error", "message": "No chunks generated."}, None)
        return

    # Taggers
    SetStatus("Running tagging models")
    yield ({"status": "running", "message": "Running tagging models"}, None)

    combined_tagged_texts = []
    emotion_tagged_texts = []
    bgsfx_tagged_texts = []
    fgsfx_tagged_texts = []

    total_emo_tags = 0
    total_bg_tags = 0
    total_fg_tags = 0

    for idx, chunk in enumerate(chunks, start=1):
        emo_result = EmotionTagger.ProcessSingleText(chunk, Tone=tone)
        emo_text = emo_result.get("text_tagged", "")
        emotion_tagged_texts.append(emo_text)
        total_emo_tags += emo_result.get("b_tags", 0) + emo_result.get("i_tags", 0)

        sys.stdout.write("\nEmotion Tags:\n")
        sys.stdout.write(str(emo_text) + "\n")
        sys.stdout.flush()

        bg_result = BGSFXTagger.ProcessSingleText(chunk, SFXStyle=sfx_style)
        bg_text = bg_result.get("text_tagged", "")
        total_bg_tags += bg_result.get("b_tags", 0)
        bgsfx_tagged_texts.append(bg_result.get("text_tagged", ""))

        sys.stdout.write("\nBGSFX Tags:\n")
        sys.stdout.write(str(bg_text) + "\n")
        sys.stdout.flush()

        fg_result = FGSFXTagger.ProcessSingleText(chunk, SFXStyle=sfx_style)
        fg_text = fg_result.get("text_tagged", "")
        total_fg_tags += fg_result.get("b_tags", 0)
        fgsfx_tagged_texts.append(fg_result.get("text_tagged", ""))

        sys.stdout.write("\nFGSFX Tags:\n")
        sys.stdout.write(str(fg_text) + "\n")
        sys.stdout.flush()

        combined_result = TagCombiner.CombineTags(
            emo_text,
            bg_result.get("text_tagged", ""),
            fg_result.get("text_tagged", "")
        )
        combined_text = combined_result.get("combined_text", "")
        combined_tagged_texts.append(combined_text)

        sys.stdout.write("\nCombined Tags:\n")
        sys.stdout.write(str(combined_text) + "\n")
        sys.stdout.flush()

    SetStatus("Text tagging complete.")
    yield ({"status": "running", "message": "Text tagging complete"}, None)

    # Convert emotion tags to JSON
    SetStatus("Formatting emotion tags for speech generation")
    yield (
        {"status": "running", "message": "Formatting emotion tags for speech generation"},
        None,
    )

    speaker_id = voiceName if voiceName else "1002"

    try:
        converted_json_blocks = ConvertTagToJSON.convert_list_of_tagged_texts(
            emotion_tagged_texts, speaker_id=speaker_id
        )

    except Exception as e:
        SetStatus(f"Conversion failed: {e}")
        yield ({"status": "error", "message": f"Conversion failed: {e}"}, None)
        return

    # -------------------------------------------------------
    # CHUNK-BY-CHUNK TTS WITH TIMING
    # -------------------------------------------------------
    SetStatus("Generating spoken audio")
    yield (
        {"status": "running", "message": "Generating spoken audio"},
        None,
    )

    output_root = "tts_outputs"
    os.makedirs(output_root, exist_ok=True)

    run_id = int(time.time())
    tts_folder = os.path.join(output_root, f"run_{run_id}")
    os.makedirs(tts_folder, exist_ok=True)

    wav_paths = []
    start_time = time.time()
    total_chunks = len(converted_json_blocks)

    try:
        for idx, block in enumerate(converted_json_blocks, start=1):

            # Timing stats
            elapsed = time.time() - start_time
            elapsed_str = format_time(elapsed)

            avg_time_per_chunk = 0
            if (idx>1):
                avg_time_per_chunk = elapsed / (idx-1)
            total_predicted = avg_time_per_chunk * total_chunks
            total_predicted_str = format_time(total_predicted)

            message = ""
            if (total_predicted_str == "00:00"):
                message = (f"Generating audio ({idx} of {total_chunks})")
            else:
                message = (f"Generating audio ({idx} of {total_chunks}, {elapsed_str} / {total_predicted_str})")

            SetStatus(message)
            yield ({"status": "running", "message": message}, None)

            # Output folder for this chunk
            chunk_temp_folder = os.path.join(tts_folder, f"temp_{idx:03d}")
            os.makedirs(chunk_temp_folder, exist_ok=True)

            # Run TTS for this single chunk
            TextToSpeech.speak_json(block, output_folder=chunk_temp_folder)

            combined = os.path.join(chunk_temp_folder, "combined.wav")
            wav_paths.append(combined)

        # Combine WAVs
        import soundfile as sf
        import numpy as np

        merged_audio = []

        for p in wav_paths:
            audio, _ = sf.read(p)
            merged_audio.append(audio.astype("float32"))

        final_audio = np.concatenate(merged_audio)
        final_path = os.path.join(tts_folder, "combined_full.wav")
        sf.write(final_path, final_audio, 24000)

        output_audio_path = final_path

        yield (
            {
                "status": "running",
                "message": "Spoken audio generated.",
                "tts_output_folder": tts_folder,
                "output_audio": output_audio_path,
            },
            None,
        )

    except Exception as e:
        SetStatus(f"TTS failed: {e}")
        yield ({"status": "error", "message": f"TTS failed: {e}"}, None)
        return

    # -------------------------------------------------------
    # SFX OVERLAY
    # -------------------------------------------------------
    SetStatus("Overlaying sound effects")
    yield ({"status": "running", "message": "Overlaying sound effects"}, None)

    full_bgsfx_script = "\n".join(bgsfx_tagged_texts)
    full_fgsfx_script = "\n".join(fgsfx_tagged_texts)

    try:
        sfx_output_path = os.path.join(tts_folder, "final_sfx.wav")

        SFXModule.run_add_sfx(
            audio_path=output_audio_path,
            bgsfx_text=full_bgsfx_script,
            fgsfx_text=full_fgsfx_script,
            output_path=sfx_output_path,
            fade_ms=120,
            debug_align=False
        )

        output_audio_path = sfx_output_path

        yield (
            {
                "status": "running",
                "message": "SFX added successfully.",
                "output_audio": output_audio_path,
            },
            None,
        )

    except Exception as e:
        SetStatus(f"SFX stage failed: {e}")
        yield ({"status": "error", "message": f"SFX stage failed: {e}"}, None)
        return

    # Finish
    SetStatus("Completed.")
    print("Audiobook generation completed.\n")

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
        output_audio_path,
    )


# ------------------------------------------------------------
# Gradio & Render
# ------------------------------------------------------------

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
        gr.Audio(label="Final Audio", type="filepath"),
    ],
    title="Emotional Audiobook API",
    description="Generates emotion-tagged audiobook audio with dynamic SFX.",
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

print("Server running.")
while True:
    time.sleep(60)
