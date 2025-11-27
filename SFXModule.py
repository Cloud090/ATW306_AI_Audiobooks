import os
import re
from faster_whisper import WhisperModel
from pydub import AudioSegment

BGSFX_FOLDER = "BGSFXSounds"
FGSFX_FOLDER = "FGSFXSounds"

BGSFXVol = 50
FGSFXVol = 50

def norm(w: str):
    return re.sub(r"[^\w']+", "", w.lower().strip())

def parse_bgsfx_script(text):
    token_pattern = re.compile(r"<<[^>]+>>|\b[\w']+\b")
    start_re = re.compile(r"<<\s*BGSFX\s+Sound=([^>]+)>>", re.IGNORECASE)
    end_re = re.compile(r"<<\s*\\BGSFX\s*>>", re.IGNORECASE)
    script_words = []
    events = []
    stack = []
    idx = 0
    for m in token_pattern.finditer(text):
        tok = m.group(0)
        if tok.startswith("<<"):
            m1 = start_re.match(tok)
            if m1:
                stack.append({"sound": m1.group(1).strip(), "start_idx": idx})
                continue
            if end_re.match(tok):
                if stack:
                    ev = stack.pop()
                    end_idx = idx - 1
                    if end_idx >= ev["start_idx"]:
                        events.append({
                            "sound": ev["sound"],
                            "start_word_idx": ev["start_idx"],
                            "end_word_idx": end_idx
                        })
                continue
        else:
            script_words.append(norm(tok))
            idx += 1
    return script_words, events

def parse_fgsfx_script(text):
    token_pattern = re.compile(r"<<[^>]+>>|\b[\w']+\b")
    tag_re = re.compile(r"<<\s*FGSFX\s+Sound=([^>]+)>>", re.IGNORECASE)
    words = []
    events = []
    idx = 0
    for m in token_pattern.finditer(text):
        tok = m.group(0)
        if tok.startswith("<<"):
            m1 = tag_re.match(tok)
            if m1:
                events.append({"sound": m1.group(1).strip(), "word_idx": idx})
                continue
        else:
            words.append(norm(tok))
            idx += 1
    return words, events

def align_script_to_transcript(script_words, spoken_words, debug_label="ALIGN", debug=False, max_window=4):
    script_items = [{"w": w, "idx": i} for i, w in enumerate(script_words)]
    trans_items = [{"w": norm(w.word), "idx": i, "obj": w} for i, w in enumerate(spoken_words)]
    mapping = {}
    def dbg(msg):
        if not debug: return
        s_idx = script_items[0]["idx"] if script_items else None
        t_idx = trans_items[0]["idx"] if trans_items else None
        s_word = script_items[0]["w"] if script_items else "∅"
        t_word = trans_items[0]["w"] if trans_items else "∅"
        print(f"[{debug_label}][s0={s_idx} t0={t_idx}] ({s_word!r} vs {t_word!r}) {msg}")
    while script_items and trans_items:
        s0 = script_items[0]
        t0 = trans_items[0]
        dbg("TOP")
        if s0["w"] == t0["w"]:
            mapping[s0["idx"]] = t0["idx"]
            script_items.pop(0)
            trans_items.pop(0)
            dbg(f"Direct match {s0['w']}")
            continue
        max_i = min(max_window, len(script_items))
        max_j = min(max_window, len(trans_items))
        best_pair = None
        best_cost = None
        for i in range(max_i):
            for j in range(max_j):
                if script_items[i]["w"] == trans_items[j]["w"]:
                    cost = i + j
                    if best_pair is None or cost < best_cost:
                        best_pair = (i, j)
                        best_cost = cost
        if best_pair:
            i, j = best_pair
            dbg(f"Window match script_offset={i}, trans_offset={j}")
            for _ in range(i):
                dropped = script_items.pop(0)
                mapping[dropped["idx"]] = None
                dbg(f"Dropped script {dropped}")
            for _ in range(j):
                extra = trans_items.pop(0)
                dbg(f"Skip trans {extra}")
            if script_items and trans_items and script_items[0]["w"] == trans_items[0]["w"]:
                s_head = script_items.pop(0)
                t_head = trans_items.pop(0)
                mapping[s_head["idx"]] = t_head["idx"]
                dbg(f"Aligned {s_head['w']}")
                continue
            else:
                if trans_items:
                    trans_items.pop(0)
                continue
        trans_items.pop(0)
        dbg("Skip transcript word")
    for leftover in script_items:
        mapping[leftover["idx"]] = None
    return mapping

def trim_trailing_silence(audio, silence_thresh=-50.0, chunk_size=10):
    length = len(audio)
    trim_point = length
    while trim_point > 0:
        seg = audio[trim_point - chunk_size:trim_point]
        if seg.dBFS < silence_thresh:
            trim_point -= chunk_size
        else:
            break
    trim_point = min(length, trim_point + 200)
    return audio[:trim_point]

def run_add_sfx(
    audio_path,
    bgsfx_text,
    fgsfx_text,
    output_path,
    fade_ms=120,
    silence_threshold=-50.0,
    silence_chunk_ms=10,
    pad_after_last=5.0,
    debug_align=False,
    verbose=True
):
    if verbose: print("Parsing tag text...")
    bgsfx_words, bgsfx_events = parse_bgsfx_script(bgsfx_text)
    fgsfx_words, fgsfx_events = parse_fgsfx_script(fgsfx_text)

    if verbose: print("Transcribing audio with Whisper...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    spoken = []
    for seg in segments:
        if seg.words:
            spoken.extend(seg.words)

    if verbose: print("Aligning BGSFX tags...")
    map_bgsfx = align_script_to_transcript(
        bgsfx_words, spoken, debug_label="BGSFX", debug=debug_align
    )
    results_bgsfx = []
    for ev in bgsfx_events:
        sw, ew = ev["start_word_idx"], ev["end_word_idx"]
        if sw not in map_bgsfx or ew not in map_bgsfx:
            continue
        si, ei = map_bgsfx[sw], map_bgsfx[ew]
        if si is None or ei is None:
            continue
        t1 = spoken[si].start
        t2 = spoken[ei].end
        results_bgsfx.append({"sound": ev["sound"], "start": t1, "end": t2})

    if verbose: print("Aligning FGSFX tags...")
    map_fgsfx = align_script_to_transcript(
        fgsfx_words, spoken, debug_label="FGSFX", debug=debug_align
    )
    results_fgsfx = []
    for ev in fgsfx_events:
        wi = ev["word_idx"]
        if wi not in map_fgsfx:
            continue
        ti = map_fgsfx[wi]
        if ti is None:
            continue
        t = spoken[ti].start
        results_fgsfx.append({"sound": ev["sound"], "time": t})

    if verbose: print("Building audio mix...")
    base = AudioSegment.from_file(audio_path)
    base_ms = len(base)

    narration_peak = base.max_dBFS if base.max_dBFS != float("-inf") else -20.0
    bgsfx_target_db = narration_peak - (20 - (BGSFXVol / 5))
    fgsfx_target_db = narration_peak - (20 - (FGSFXVol / 5))

    last_time = 0
    for b in results_bgsfx:
        last_time = max(last_time, b["end"])
    for f in results_fgsfx:
        last_time = max(last_time, f["time"])

    target_len = max(base_ms, int((last_time + pad_after_last) * 1000))
    if target_len > base_ms:
        pad = AudioSegment.silent(duration=target_len - base_ms)
        base = base + pad

    for ev in results_bgsfx:
        snd_path = os.path.join(BGSFX_FOLDER, f"{ev['sound']}.mp3")
        if not os.path.isfile(snd_path):
            if verbose: print(f"[WARN] Missing BGSFX file: {snd_path}")
            continue
        snd = AudioSegment.from_file(snd_path)
        if snd.max_dBFS != float("-inf"):
            diff = bgsfx_target_db - snd.max_dBFS
            snd = snd.apply_gain(diff)
        snd = snd.fade_in(fade_ms).fade_out(fade_ms)
        start_ms = int(ev["start"] * 1000)
        end_ms = int(ev["end"] * 1000)
        dur = end_ms - start_ms
        if dur > 0:
            snd = snd[:dur]
            base = base.overlay(snd, position=start_ms)

    for ev in results_fgsfx:
        snd_path = os.path.join(FGSFX_FOLDER, f"{ev['sound']}.mp3")
        if not os.path.isfile(snd_path):
            if verbose: print(f"[WARN] Missing FGSFX file: {snd_path}")
            continue
        snd = AudioSegment.from_file(snd_path)
        if snd.max_dBFS != float("-inf"):
            diff = fgsfx_target_db - snd.max_dBFS
            snd = snd.apply_gain(diff)
        start_ms = int(ev["time"] * 1000)
        base = base.overlay(snd, position=start_ms)

    final_audio = trim_trailing_silence(
        base, silence_thresh=silence_threshold, chunk_size=silence_chunk_ms
    )
    final_audio.export(output_path, format="wav")
    if verbose: print(f"Saved final mix: {output_path}")
    return output_path
