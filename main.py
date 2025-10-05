"""
SailRR: A Real-Time, End-to-End Voice Assistant.

This script captures audio from a microphone, transcribes it to text, generates
a response using a small language model, and synthesizes the response back into
speech, creating a continuous conversational loop.

It creates and overwrites two temporary files in its directory:
- live_prompt.wav: The user's recorded speech.
- response.wav: The AI's generated speech.

Usage:
    python voice_agent.py
"""

import torch
import whisper
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from TTS.api import TTS
import sounddevice as sd
from scipy.io.wavfile import write, read
import gc
import os
import numpy as np
import re
import traceback

# --- File paths ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LIVE_PROMPT_PATH = os.path.join(SCRIPT_DIR, "live_prompt.wav")
RESPONSE_PATH = os.path.join(SCRIPT_DIR, "response.wav")

print("Libraries loaded.")

# --- Function 1: Record + Transcribe from Microphone ---
def listen_and_transcribe():
    fs = 16000
    seconds = 7
    print(f"\nPlease speak. Recording for {seconds} seconds...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    write(LIVE_PROMPT_PATH, fs, myrecording)
    print("Recording complete.")

    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base").to("cuda")
    result = whisper_model.transcribe(LIVE_PROMPT_PATH)
    user_text = result["text"].strip()
    print(f" You said: {user_text}")

    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()
    return user_text

# --- Function 2: Generate concise SLM response ---
def get_slm_response(prompt):
    print("Loading SLM (microsoft/phi-2)...")
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    full_prompt = f"You are a helpful and concise assistant. Provide a short, natural response to this: {prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=False).to("cuda")

    print("Thinking...")
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=100)
    end_time = time.time()
    print(f"Latency for 100-token generation: {end_time - start_time:.2f} seconds.")

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response_text.split(prompt)[-1].strip()
    print(f"SailRR says: {response_text}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return response_text

# --- Helper function to clean and normalize SLM text for TTS ---
def clean_text_for_tts(text: str) -> str:
    if not text:
        return ""

    t = text.replace("\r", "\n")
    t = re.sub(r"(?mi)^\s*##\s*\w+.*$", " ", t)
    t = re.sub(r"(?i)\b(user|assistant|input|output)\s*:\s*", " ", t)
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    t = re.sub(r"[#@]{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+[^\w\s\.\,\?\!\'\-]{1}\s+", " ", t)

    if t and t[-1] not in ".?!":
        words = t.split()
        if len(words) <= 3:
            t += "."
        else:
            last = words[-1]
            if len(last) < 3 or last.islower():
                words = words[:-1]
                t = " ".join(words)
                if not t.endswith((".", "?", "!")):
                    t += "."
            else:
                t += "."

    t = t.strip()
    if len(t) > 8000:
        t = t[:8000].rsplit(" ", 1)[0] + "."
    return t

# --- Function 3: Speak with Coqui Tacotron2-DDC ---
def speak(text):
    print("Preparing Coqui TTS (Tacotron2-DDC)...")
    safe_text = clean_text_for_tts(text)
    if not safe_text:
        print("Nothing to speak after cleaning.")
        return

    print(f"Cleaned text: {safe_text!r}")

    print("Loading Coqui TTS model (Tacotron2-DDC, English-only)...")
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to("cuda")

    print("Generating audio...")
    try:
        tts.tts_to_file(text=safe_text, file_path=RESPONSE_PATH, split_sentences=False)
    except TypeError:
        tts.tts_to_file(text=safe_text, file_path=RESPONSE_PATH)
    finally:
        del tts
        gc.collect()
        torch.cuda.empty_cache()

    print("Playing audio...")
    samplerate, data = read(RESPONSE_PATH)
    sd.play(data, samplerate)
    sd.wait()

def run_voice_assistant():
    try:
        user_prompt = listen_and_transcribe()
        if user_prompt and len(user_prompt.strip()) > 2:
            ai_response = get_slm_response(user_prompt)
            speak(ai_response)
        else:
            print("No clear speech detected. Please try again.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting the SailRR Voice Assistant...")
    run_voice_assistant()
