import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import SFXModule


def gui_log(widget, text):
    widget.configure(state="normal")
    widget.insert(tk.END, text + "\n")
    widget.configure(state="disabled")
    widget.see(tk.END)


def run_sfx_thread(
    logbox,
    audio_path,
    bgsfx_text,
    fgsfx_text,
    output_path
):
    try:
        gui_log(logbox, "=== Starting SFX generation ===")

        # run_add_sfx no longer accepts verbose
        result_path = SFXModule.run_add_sfx(
            audio_path=audio_path,
            bgsfx_text=bgsfx_text,
            fgsfx_text=fgsfx_text,
            output_path=output_path,
            debug_align=True
        )

        gui_log(logbox, "=== Completed successfully ===")
        gui_log(logbox, f"Saved to: {result_path}")

    except Exception as e:
        gui_log(logbox, f"[ERROR] {str(e)}")


def main():
    root = tk.Tk()
    root.title("SFX Module Test")
    root.geometry("900x700")

    default_audio = "combined.wav"

    default_bgsfx_text = ""
    if os.path.isfile("BGSFXTags.txt"):
        with open("BGSFXTags.txt", "r", encoding="utf-8") as f:
            default_bgsfx_text = f.read()

    default_fgsfx_text = ""
    if os.path.isfile("FGSFXTags.txt"):
        with open("FGSFXTags.txt", "r", encoding="utf-8") as f:
            default_fgsfx_text = f.read()

    frm_audio = tk.Frame(root)
    frm_audio.pack(fill="x", padx=10, pady=5)

    tk.Label(frm_audio, text="Input Audio:").pack(anchor="w")

    audio_var = tk.StringVar(value=default_audio)
    audio_entry = tk.Entry(frm_audio, textvariable=audio_var, width=60)
    audio_entry.pack(side="left", padx=5)

    def browse_audio():
        path = filedialog.askopenfilename(
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.m4a"),
                ("All Files", "*.*")
            ]
        )
        if path:
            audio_var.set(path)

    tk.Button(frm_audio, text="Browse", command=browse_audio).pack(side="left", padx=5)

    frm_output = tk.Frame(root)
    frm_output.pack(fill="x", padx=10, pady=5)

    tk.Label(frm_output, text="Output:").pack(anchor="w")

    output_var = tk.StringVar(value="FinalMix.wav")
    out_entry = tk.Entry(frm_output, textvariable=output_var, width=60)
    out_entry.pack(side="left", padx=5)

    def browse_output():
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )
        if path:
            output_var.set(path)

    tk.Button(frm_output, text="Browse", command=browse_output).pack(side="left", padx=5)

    tk.Label(root, text="BGSFX Tagged Text:").pack(anchor="w", padx=10)
    bgsfx_textbox = scrolledtext.ScrolledText(root, height=8, width=100)
    bgsfx_textbox.pack(padx=10, pady=5)
    bgsfx_textbox.insert("1.0", default_bgsfx_text)

    tk.Label(root, text="FGSFX Tagged Text:").pack(anchor="w", padx=10)
    fgsfx_textbox = scrolledtext.ScrolledText(root, height=8, width=100)
    fgsfx_textbox.pack(padx=10, pady=5)
    fgsfx_textbox.insert("1.0", default_fgsfx_text)

    tk.Label(root, text="Log:").pack(anchor="w", padx=10)
    logbox = scrolledtext.ScrolledText(root, height=10, width=100, state="disabled")
    logbox.pack(padx=10, pady=5)

    def run_clicked():
        audio_path = audio_var.get().strip()
        output_path = output_var.get().strip()
        bgsfx_text = bgsfx_textbox.get("1.0", tk.END).strip()
        fgsfx_text = fgsfx_textbox.get("1.0", tk.END).strip()

        if not audio_path or not os.path.isfile(audio_path):
            messagebox.showerror("Error", "Please select a valid narration audio file.")
            return

        if not output_path:
            messagebox.showerror("Error", "Please specify an output file path.")
            return

        t = threading.Thread(
            target=run_sfx_thread,
            args=(logbox, audio_path, bgsfx_text, fgsfx_text, output_path),
            daemon=True
        )
        t.start()

    tk.Button(root, text="Generate SFX Mix", command=run_clicked, height=2).pack(pady=10)

    root.mainloop()


main()
