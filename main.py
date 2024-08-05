import json
import os
import subprocess
import time

import sounddevice as sd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configuration
CONFIG = {
    "model_name": "aiola/whisper-medusa-v1",
    "sample_rate": 16000,
    "chunk_duration": 5,  # Process 5 seconds of audio at a time
    "ollama_model": "mistral-nemo:latest"
}

# Load model and processor
model_name = CONFIG["model_name"]
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Audio settings
sample_rate = 16000
chunk_duration = 5  # Process 5 seconds of audio at a time


# Function to process audio chunk
def process_audio(audio_chunk):
    # Normalize the audio chunk
    audio_chunk = audio_chunk / torch.max(torch.abs(audio_chunk))

    input_features = processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt").input_features
    input_features = input_features.to(device)

    with torch.no_grad():
        generated_ids = model.generate(input_features)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


# Callback function for audio stream
def audio_callback(indata, status):
    if status:
        print(f"Error in audio stream: {status}")
        return

    try:
        audio_chunk = torch.from_numpy(indata[:, 0]).float()
        transcription = process_audio(audio_chunk)

        with open("live_translation.txt", "a") as f:
            f.write(transcription + "\n")

        print(f"Translated: {transcription}")
    except Exception as e:
        print(f"Error processing audio: {e}")


# Function to summarize content using Ollama
def summarize_with_ollama(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    prompt = f"Please summarize the following text concisely:\n\n{content}"

    ollama_command = [
        "ollama", "run", "mistral-nemo:latest",
        json.dumps({"prompt": prompt, "stream": False})
    ]

    try:
        result = subprocess.run(ollama_command, capture_output=True, text=True, check=True)
        try:
            return json.loads(result.stdout)['response']
        except json.JSONDecodeError as je:
            print(f"Error parsing Ollama output: {je}")
            print(f"Raw output: {result.stdout}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama: {e}")
        print(f"Ollama stderr: {e.stderr}")
        return None


# Start audio stream
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate,
                    blocksize=int(sample_rate * chunk_duration)):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped listening. Generating summary...")

# Generate summary after recording stops
summary = summarize_with_ollama("live_translation.txt")

if summary:
    print("\nSummary of the meeting:")
    print(summary)

    # Optionally, save the summary to a file
    with open("meeting_summary.txt", "w") as f:
        f.write(summary)
    print("Summary saved to 'meeting_summary.txt'")
else:
    print("Failed to generate summary.")


# Clean up the translation file
def cleanup_files():
    try:
        os.remove("live_translation.txt")
        print("Cleaned up temporary translation file.")
    except FileNotFoundError:
        print("No temporary file to clean up.")
    except Exception as e:
        print(f"Error cleaning up file: {e}")


cleanup_files()
