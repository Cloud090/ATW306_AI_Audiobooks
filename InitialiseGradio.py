import gradio as gr
import time
import os
import threading
import requests
import RenderLink
import TextSplitter
import EmotionTagger

CurrentStatus = "Idle"


def SetStatus(message):
    global CurrentStatus
    CurrentStatus = message
    print(f"[STATUS] {message}")


def GetStatus():
    return {"status": CurrentStatus}


def GenerateAudiobook(text, persona, sfxStyle, voiceAudio=None, voiceName=None):
    global CurrentStatus

    print("\n\nNew Audiobook Generation\n")
    print(f"Persona: {persona}")
    print(f"SFX Style: {sfxStyle}")
    print(f"Voice Audio: {voiceAudio}")
    print(f"Voice Name: {voiceName}")
    print("\n")

    # Split
    SetStatus("Splitting text...")
    yield {"status": "running", "message": "Splitting text..."}

    try:
        chunks = TextSplitter.SplitText(text, Limit=400, ModelName="roberta-base")
        SetStatus(f"Text split complete — {len(chunks)} sections created.")
        print(f"[DEBUG] Split into {len(chunks)} chunks:")
        for i, c in enumerate(chunks, 1):
            print(f"  [Chunk {i:02}] {repr(c[:100])}...")
        yield {"status": "running", "message": f"Text split complete — {len(chunks)} sections created."}
    except Exception as e:
        SetStatus(f"Text splitting failed: {e}")
        print(f"[ERROR] {e}")
        yield {"status": "error", "message": f"Text splitting failed: {e}"}
        return

    # Emotion tagging
    SetStatus("Analyzing emotional tone...")
    yield {"status": "running", "message": "Analyzing emotional tone..."}

    try:
        print(f"[DEBUG] Starting EmotionTagger.ProcessTextArray with {len(chunks)} chunks...")
        tagged_results = EmotionTagger.ProcessTextArray(chunks)

        # Log tag results
        tagged_texts = []
        for i, r in enumerate(tagged_results, 1):
            tagged = r.get("text_tagged", "[No tagged text returned]")
            tagged_texts.append(tagged)
            print(f"\n[Tagged {i:02}]")
            print(tagged)
            print("\n")

        SetStatus("Emotion tagging complete.")
        yield {"status": "running", "message": "Emotion tagging complete."}
    except Exception as e:
        SetStatus(f"Emotion tagging failed: {e}")
        print(f"[ERROR] {e}")
        yield {"status": "error", "message": f"Emotion tagging failed: {e}"}
        return

    # Tagged text to voice (to be created)
    SetStatus("Generating emotional narration...")
    print("[DEBUG] Simulating emotional narration generation...")
    yield {"status": "running", "message": "Generating emotional narration..."}
    time.sleep(2)
    print("[DEBUG] Narration generation simulated complete.")

    # Add sound effects (to be created)
    SetStatus("Adding sound effects...")
    print("[DEBUG] Simulating sound effect layering...")
    yield {"status": "running", "message": "Adding sound effects..."}
    time.sleep(2)
    print("[DEBUG] Sound effects simulation complete.")

    # Finalise (to be created)
    SetStatus("Completed audiobook generation.")
    print("[SUCCESS] Audiobook generation finished successfully.\n")
    yield {
        "status": "success",
        "message": "Completed audiobook generation.",
        "chunk_count": len(chunks),
        "tagged_texts": tagged_texts,
        "output_audio": "final_audiobook.wav"
    }


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
            response = requests.get(f"{renderBase}/current", timeout=10)
            print(f"[PING] Keep-alive returned {response.status_code}")
        except Exception as e:
            print(f"[PING ERROR] {e}")
        time.sleep(600)  # 10 min


print("Launching Emotional Audiobook API...")

ApiMain = gr.Interface(
    fn=GenerateAudiobook,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Textbox(label="Persona"),
        gr.Textbox(label="SFX Style"),
        gr.Audio(type="filepath", label="Voice Sample (optional)"),
        gr.Textbox(label="Voice Name"),
    ],
    outputs=gr.JSON(label="Streaming API Output"),
    title="Emotional Audiobook API",
    description="Headless API endpoint for audiobook generation with live status updates.",
    live=False,
)

ApiStatus = gr.Interface(
    fn=GetStatus,
    inputs=[],
    outputs=gr.JSON(label="Current Status"),
    title="Status Endpoint",
    description="Returns the current audiobook generation status.",
)

App = gr.TabbedInterface(
    [ApiMain, ApiStatus],
    tab_names=["Generate", "Status"]
)

App.launch(share=True, prevent_thread_lock=True, inline=False)

print("Waiting for Gradio to provide share URL ...")
ShareUrl = WaitForShareUrl(App)

if ShareUrl:
    print(f"[INFO] Gradio Share URL: {ShareUrl}")
    print("[INFO] Publishing to Render ...")
    result = RenderLink.SetGradioURL(ShareUrl)
    print(f"[INFO] Render update result: {result}")

    thread = threading.Thread(target=KeepAliveLoop, daemon=True)
    thread.start()
else:
    print("[WARN] Could not detect share URL from Gradio interface object.")

print("Server running — leave this open while active.")
while True:
    time.sleep(60)
