import tkinter as tk
from tkinter import filedialog
from gradio_client import Client
import RenderLink
import os
import threading
import time
import requests

PollingFlag = False  # Controls the polling loop

def LoadOptionsFromFile(filepath):
    if not os.path.exists(filepath):
        return []
    options = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                # Take part after the number
                option = line.split("–")[0].split(".", 1)[1].strip()
                if option:
                    options.append(option)
    return options


def SelectAudio():
    path = filedialog.askopenfilename(
        title="Select Voice Sample",
        filetypes=[("Audio Files", "*.wav;*.mp3"), ("All Files", "*.*")]
    )
    if path:
        VoicePathVar.set(path)


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

            PollingFlag = True
            threading.Thread(target=PollStatus, args=(StatusUrl,), daemon=True).start()

            client = Client(ApiUrl)

            # Get field info
            raw_text = TextEntry.get("1.0", tk.END).strip()
            persona = PersonaVar.get()
            sfxStyle = SfxVar.get()
            voicePath = VoicePathVar.get() or None
            voiceName = VoiceNameVar.get()

            # Create input tags
            tagged_text = f"<<PERSONA = {persona}>>\n<<SFXSTYLE = {sfxStyle}>>\n{raw_text}"
            print(tagged_text)

            OutputBox.insert(tk.END, "Sending request...\n")

            # Send text to API
            result = client.predict(
                tagged_text,
                persona,
                sfxStyle,
                voicePath,
                voiceName,
                api_name="/predict"
            )

            PollingFlag = False
            OutputBox.insert(tk.END, "\nFinal result received.\n")

            if isinstance(result, dict):
                for key, value in result.items():
                    OutputBox.insert(tk.END, f"{key}: {value}\n")
                if result.get("output_audio"):
                    OutputBox.insert(tk.END, f"Output file: {result['output_audio']}\n")

        except Exception as e:
            PollingFlag = False
            OutputBox.insert(tk.END, f"Error: {e}\n")

    threading.Thread(target=Task, daemon=True).start()


# GUI
root = tk.Tk()
root.title("Emotional Audiobook Frontend (Dynamic GUI)")
root.geometry("750x600")

tk.Label(root, text="Story Text:").pack(anchor="w", padx=10, pady=(10, 0))
TextEntry = tk.Text(root, wrap="word", height=8)
TextEntry.pack(fill="x", padx=10, pady=5)

ControlFrame = tk.Frame(root)
ControlFrame.pack(fill="x", padx=10)

PersonasFile = "Personas.txt"
SfxFile = "SFXStyle.txt"

PersonaOptions = LoadOptionsFromFile(PersonasFile) or ["Calm", "Balanced", "Energetic"]
SfxOptions = LoadOptionsFromFile(SfxFile) or ["Balanced Mix", "Minimalist", "Cinematic"]

tk.Label(ControlFrame, text="Persona:").grid(row=0, column=0, sticky="w")
PersonaVar = tk.StringVar(value=PersonaOptions[0])
tk.OptionMenu(ControlFrame, PersonaVar, *PersonaOptions).grid(row=0, column=1, sticky="w")

tk.Label(ControlFrame, text="SFX Style:").grid(row=0, column=2, sticky="w", padx=(20, 0))
SfxVar = tk.StringVar(value=SfxOptions[0])
tk.OptionMenu(ControlFrame, SfxVar, *SfxOptions).grid(row=0, column=3, sticky="w")

tk.Label(ControlFrame, text="Voice Preset:").grid(row=1, column=0, sticky="w", pady=(10, 0))
VoiceNameVar = tk.StringVar(value="Voice A")
tk.OptionMenu(ControlFrame, VoiceNameVar, "Voice A", "Voice B", "Voice C").grid(row=1, column=1, sticky="w", pady=(10, 0))

tk.Button(ControlFrame, text="Select Voice Sample", command=SelectAudio).grid(row=1, column=2, padx=(20, 0), pady=(10, 0))
VoicePathVar = tk.StringVar()
tk.Entry(ControlFrame, textvariable=VoicePathVar, width=30).grid(row=1, column=3, sticky="w", pady=(10, 0))

tk.Button(root, text="Generate Audiobook", bg="#0078D7", fg="white",
          height=2, command=GenerateAudiobook).pack(fill="x", padx=10, pady=10)

tk.Label(root, text="Output Log:").pack(anchor="w", padx=10)
OutputBox = tk.Text(root, wrap="word", height=15, bg="#111", fg="#0f0")
OutputBox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

root.mainloop()
