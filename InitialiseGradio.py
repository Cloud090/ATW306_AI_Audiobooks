# This script, InitialiseGradio.py, creates a Gradio API capable of generating an emotional audiobook with sound effects. Running the script initialises the API,
# updates a static endpoint with the Gradio API address so a frontend can find it, and handles audiobook generation requests. To generate an audiobook, a
# Frontend Interface send a request to the Gradio API. The request is forwarded to this scripts "GenerateAudiobook" function, which creates the audiobook, and
# sends ongoing status updates and a final result back to the Frontend Interface.

# The script also sends a regular "keep alive" request to the static endpoint every ten minutes, to prevent it from timing out while the API is running.


import gradio as gr
import time
import os
import threading
import requests
import json
import sys
import soundfile as sf
import numpy as np

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

    # Checks if the persona is from the current list, defaults to "Neutral" if not.
    valid_tones = ["Unemotional", "Calm", "Neutral", "Dramatic"]
    tone = persona.strip() if persona in valid_tones else "Neutral"

    # Checks if the sfx style is from the current list, defaults to "Balanced" if not.
    valid_sfx_styles = ["None", "Subtle", "Balanced", "Cinematic"]
    sfx_style = (
        sfxStyle.strip().title()
        if sfxStyle.title() in valid_sfx_styles
        else "Balanced"
    )

    # Step 1 - Split text
    SetStatus("Splitting text")
    yield ({"status": "running", "message": "Splitting text"}, None)

    # Uses the SplitText function from the local TextSplitter module to split the text into chunks of under 400 tokens length.
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

    # Step 2 - Run text tagging models.
    SetStatus("Running tagging models")
    yield ({"status": "running", "message": "Running tagging models"}, None)

    # Sets blank lists to store completed tagged texts, and counter variables to count tags generated.
    combined_tagged_texts = []
    emotion_tagged_texts = []
    bgsfx_tagged_texts = []
    fgsfx_tagged_texts = []

    total_emo_tags = 0
    total_bg_tags = 0
    total_fg_tags = 0


    # Iterates through every text chunk, using the three local modules to generate emotion tags, background sound tags, and foreground sound tags 
    # Respectively. These are then combined using the "TagCombiner" local module.
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

    # Step 3 - Convert emotion tags to JSON format used by the Text To Speech local module.
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

    # Step 4 - Generate spoken audio, using the local TextToSpeech module
    SetStatus("Generating spoken audio")
    yield (
        {"status": "running", "message": "Generating spoken audio"},
        None,
    )

    # Set output folder, creating it if it doesn't already exist.
    output_root = "tts_outputs"
    os.makedirs(output_root, exist_ok=True)

    # Creates subfolder based on current timestamp.
    run_id = int(time.time())
    tts_folder = os.path.join(output_root, f"run_{run_id}")
    os.makedirs(tts_folder, exist_ok=True)

    # Create a list to store generated sound files.
    wav_paths = []

    # Stores start time and total number of chunks to be processed, for time estimation.
    start_time = time.time()
    total_chunks = len(converted_json_blocks)

    # Iterates through every JSON block.
    try:
        for idx, block in enumerate(converted_json_blocks, start=1):

            # Timing stats
            elapsed = time.time() - start_time
            elapsed_str = format_time(elapsed)

            # Calculate remaining time (average time per chunk x total number of chunks)
            avg_time_per_chunk = 0
            if (idx>1):
                avg_time_per_chunk = elapsed / (idx-1)
            total_predicted = avg_time_per_chunk * total_chunks
            total_predicted_str = format_time(total_predicted)

            # Generates Gradio message, for status updates.
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

        # After all individual audio files are created, this combines them into a single .wav file.
        merged_audio = []

        for p in wav_paths:
            audio, _ = sf.read(p)
            merged_audio.append(audio.astype("float32"))

        final_audio = np.concatenate(merged_audio)
        final_path = os.path.join(tts_folder, "combined_full.wav")
        sf.write(final_path, final_audio, 24000)

        # Sends the final path to back to the Frontend Interface as an intermediary before sound effects are added.
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

    # Error handling in case of failure.
    except Exception as e:
        SetStatus(f"TTS failed: {e}")
        yield ({"status": "error", "message": f"TTS failed: {e}"}, None)
        return

    # Step 5 - Add sound effects/
    SetStatus("Overlaying sound effects")
    yield ({"status": "running", "message": "Overlaying sound effects"}, None)

    # Joins sound effect tagged text files together, as these are processed in a single block.
    full_bgsfx_script = "\n".join(bgsfx_tagged_texts)
    full_fgsfx_script = "\n".join(fgsfx_tagged_texts)

    try:
        # Sets path of final audio file
        sfx_output_path = os.path.join(tts_folder, "final_sfx.wav")

        # Runs the add_sfx function of the local module SFXModule - this finds the timestamps corresponding to the audio position of the listed soud effect tags,
        # and adds them, using pydub.
        SFXModule.run_add_sfx(
            audio_path=output_audio_path,
            bgsfx_text=full_bgsfx_script,
            fgsfx_text=full_fgsfx_script,
            output_path=sfx_output_path,
            fade_ms=120,
            debug_align=False
        )

        # Passes the final audio back to the Frontend Interface.
        output_audio_path = sfx_output_path

        yield (
            {
                "status": "running",
                "message": "SFX added successfully.",
                "output_audio": output_audio_path,
            },
            None,
        )

    # Error handling in case of failure.
    except Exception as e:
        SetStatus(f"SFX stage failed: {e}")
        yield ({"status": "error", "message": f"SFX stage failed: {e}"}, None)
        return

    # Finish - sends final completed message to the Frontend Interface.
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

# After the Gradio API is initialised, this function checks if the public URL has been created, returning it if found. If it hasn't been found in 60 seconds,
# the function times out, returning a None value.
def WaitForShareUrl(apiInterface, timeout=60):
    for _ in range(timeout // 2):
        url = getattr(apiInterface, "share_url", None) or getattr(apiInterface, "share_link", None)
        if url and url.startswith("http"):
            return url
        time.sleep(2)
    return None

# This function, run asynchroniously so as not to disrupt other tasks, sends a request to the static endpoint every ten minutes, keeping it from timing out.
def KeepAliveLoop(renderBase="https://audioapi-g2ru.onrender.com"):
    while True:
        try:
            requests.get(f"{renderBase}/current", timeout=10)
        except:
            pass
        time.sleep(600)

# All functions have now been defined, and the script is ready to generate the Gradio API.
print("Launching Emotional Audiobook API")

# Initialises the Gradio API, with virtual "textboxes" for Text, Persona, SFX Style, Voice Sample (not used - potential for later), and Voice Name. It also sets up 
# output fields the Frontend Interface can use - "Streaming API Output" for messages, and "Final Audio" for the generated audiobook.
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

# Secondary Gradio interface, used to retrieve the current status of the main API.
ApiStatus = gr.Interface(fn=GetStatus, inputs=[], outputs=gr.JSON(label="Current Status"))

# Creates a combined version of the two interfaces.
App = gr.TabbedInterface([ApiMain, ApiStatus], ["Generate", "Status"])

# Launches the combined API
App.launch(share=True, prevent_thread_lock=True, inline=False)

# Waits for and saves the public Gradio URL, when available.
ShareUrl = WaitForShareUrl(App)

# Sets the static endpoint to the current URL, using the local RenderLink module.
if ShareUrl:
    RenderLink.SetGradioURL(ShareUrl)

# Asynchroniously starts the function to repeatedly make a request to the static endpoint every ten minutes, keeping it from timing out.
thread = threading.Thread(target=KeepAliveLoop, daemon=True)
thread.start()

print("Server running.")

# Repeating while loop, to keep the script from ending while the API runs.
while True:
    time.sleep(60)
