import tkinter as tk
from tkinter import filedialog
from gradio_client import Client
import RenderLink
import os
import threading
import time
import requests

PollingFlag = False


def LoadOptionsFromFile(filepath):
    if not os.path.exists(filepath):
        return []
    options = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                option = line.split("–")[0].split(".", 1)[1].strip()
                if option:
                    options.append(option)
    return options


def LoadVoicePresets(filepath):
    if not os.path.exists(filepath):
        return [], {}

    display_names = []
    id_map = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "–" in line:
                parts = line.split("–")
                display = parts[0].strip()
                number = parts[1].strip()
                if display and number.isdigit():
                    display_names.append(display)
                    id_map[display] = number

    return display_names, id_map


def PollStatus(StatusUrl):
    global PollingFlag
    while PollingFlag:
        try:
            response = requests.get(StatusUrl, timeout=5)
            if response.status_code == 200:
                data = response.json()
                message = data.get("status", "")
                if message:
                    OutputBox.insert(tk.END, message + "\n")
                    OutputBox.see(tk.END)
        except Exception as e:
            OutputBox.insert(tk.END, f"Status poll error: {e}\n")
        time.sleep(1)


def GenerateAudiobook():
    def Task():
        global PollingFlag

        try:
            OutputBox.delete("1.0", tk.END)

            # Get Render API link
            status, ApiUrl = RenderLink.GetGradioURL()
            if status != "success" or not ApiUrl:
                OutputBox.insert(tk.END, "Could not retrieve API URL from Render.\n")
                return

            OutputBox.insert(tk.END, f"Connected to {ApiUrl}\n")

            PredictUrl = ApiUrl.rstrip("/") + "/predict"
            StatusUrl = ApiUrl.rstrip("/") + "/status"

            # Start status polling
            PollingFlag = True
            threading.Thread(target=PollStatus, args=(StatusUrl,), daemon=True).start()

            client = Client(ApiUrl)

            # Gather inputs
            raw_text = TextEntry.get("1.0", tk.END).strip()
            persona_tone = PersonaVar.get()
            sfx_style = SfxVar.get()
            voice_id = VoiceIDVar.get()

            OutputBox.insert(tk.END, "Sending request...\n")

            # Call backend
            json_result, audio_path = client.predict(
                raw_text,
                persona_tone,
                sfx_style,
                None,       # no voice sample
                voice_id,   # speaker id
                api_name="/predict"
            )

            PollingFlag = False
            OutputBox.insert(tk.END, "\nFinal result received.\n\n")

            # Show JSON result
            if isinstance(json_result, dict):
                for k, v in json_result.items():
                    OutputBox.insert(tk.END, f"{k}: {v}\n")

            if audio_path:
                OutputBox.insert(tk.END, f"\nLocal Gradio audio file path:\n{audio_path}\n")

                if not os.path.isfile(audio_path):
                    OutputBox.insert(tk.END, "ERROR: File does not exist locally.\n")
                    return

                download_folder = "DownloadedAudio"
                os.makedirs(download_folder, exist_ok=True)

                local_filename = os.path.join(download_folder, "audiobook.wav")

                try:
                    # Copy directly from Gradio's local temp file
                    import shutil
                    shutil.copy(audio_path, local_filename)

                    OutputBox.insert(tk.END, f"\nCopied to: {local_filename}\n")
                    OutputBox.insert(tk.END, "Audiobook saved successfully!\n")

                except Exception as e:
                    OutputBox.insert(tk.END, f"Audio save failed: {e}\n")

        except Exception as e:
            PollingFlag = False
            OutputBox.insert(tk.END, f"Error: {e}\n")

    threading.Thread(target=Task, daemon=True).start()

root = tk.Tk()
root.title("Emotional Audiobook Frontend (Dynamic GUI)")
root.geometry("750x600")

tk.Label(root, text="Story Text:").pack(anchor="w", padx=10, pady=(10, 0))
TextEntry = tk.Text(root, wrap="word", height=8)
TextEntry.pack(fill="x", padx=10, pady=5)

ControlFrame = tk.Frame(root)
ControlFrame.pack(fill="x", padx=10)

# Persona & SFX options
PersonaOptions = ["Unemotional", "Calm", "Neutral", "Dramatic"]
SfxOptions = ["None", "Subtle", "Balanced", "Cinematic"]

tk.Label(ControlFrame, text="Tone (Persona):").grid(row=0, column=0, sticky="w")
PersonaVar = tk.StringVar(value=PersonaOptions[2])
tk.OptionMenu(ControlFrame, PersonaVar, *PersonaOptions).grid(row=0, column=1, sticky="w")

tk.Label(ControlFrame, text="SFX Style:").grid(row=0, column=2, sticky="w", padx=(20, 0))
SfxVar = tk.StringVar(value=SfxOptions[2])
tk.OptionMenu(ControlFrame, SfxVar, *SfxOptions).grid(row=0, column=3, sticky="w")

# Load voice presets
VoiceDisplayNames, VoiceIDMap = LoadVoicePresets("VoiceSelection.txt")

tk.Label(ControlFrame, text="Voice Preset:").grid(row=1, column=0, sticky="w", pady=(10, 0))

VoiceNameVar = tk.StringVar(value=VoiceDisplayNames[0] if VoiceDisplayNames else "")
VoiceIDVar = tk.StringVar(value=VoiceIDMap.get(VoiceNameVar.get(), ""))

# Update ID whenever user picks a name
def OnVoiceSelected(choice):
    VoiceIDVar.set(VoiceIDMap.get(choice, ""))

VoiceDropdown = tk.OptionMenu(
    ControlFrame,
    VoiceNameVar,
    VoiceDisplayNames[0] if VoiceDisplayNames else "",
    *VoiceDisplayNames,
    command=OnVoiceSelected
)

VoiceDropdown.grid(row=1, column=1, sticky="w", pady=(10, 0))

tk.Button(root, text="Generate Audiobook", bg="#0078D7", fg="white",
          height=2, command=GenerateAudiobook).pack(fill="x", padx=10, pady=10)

tk.Label(root, text="Output Log:").pack(anchor="w", padx=10)
OutputBox = tk.Text(root, wrap="word", height=15, bg="#111", fg="#0f0")
OutputBox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

root.mainloop()
