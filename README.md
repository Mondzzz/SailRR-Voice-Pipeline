# SailRR: A Real-Time Voice Assistant Pipeline

This project is a complete, end-to-end voice assistant that runs locally on a CUDA-enabled GPU. It actively listens for voice commands through a microphone, processes the request using a small language model (SLM), and speaks the response back to the user.

### Features

* **Real-Time Interaction:** Records audio directly from a microphone.
* **End-to-End Pipeline:** Integrates Speech-to-Text (Whisper), Language Model (Phi-2), and Text-to-Speech (Coqui TTS) in a seamless flow.
* **Memory Optimized:** Uses 4-bit quantization for the language model and aggressively clears GPU VRAM to run on consumer-grade hardware.
* **Text Sanitization:** Includes a text-cleaning function to normalize the model's output for more reliable speech synthesis.

---
### How It Works

The assistant operates in a simple, linear sequence:

1.  **Listen:** The script records 7 seconds of audio from the default microphone and saves it as `live_prompt.wav`.
2.  **Transcribe:** OpenAI's **Whisper** model processes the audio file to convert the speech into a text prompt.
3.  **Think:** The text prompt is fed to Microsoft's **Phi-2**, a powerful small language model, which generates a concise and helpful response.
4.  **Speak:** The generated text is cleaned and then synthesized into audio using **Coqui's Tacotron2-DDC** model. The resulting `response.wav` file is played back through the user's speakers.

---
### Setup and Installation

**Prerequisites:**
* Python 3.8+
* An NVIDIA GPU with CUDA installed.
* [FFmpeg](https://ffmpeg.org/download.html) (often required for audio processing by the underlying libraries).

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/SailRR-Voice-Pipeline.git](https://github.com/your-username/SailRR-Voice-Pipeline.git)
    cd SailRR-Voice-Pipeline
    ```

2.  **Install the required libraries:**
    *(It is highly recommended to use a Python virtual environment)*
    ```bash
    pip install -r requirements.txt
    ```

---
### Usage

Simply run the script from your terminal. It will begin recording automatically.

```bash
python voice_agent.py
