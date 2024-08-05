import json
import os
import subprocess
import time

import sounddevice as sd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load configuration
with open('config.json', 'r') as config_file:
    CONFIG = json.load(config_file)

# Load model and processor
model_name = CONFIG["model_name"]
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Audio settings
sample_rate = CONFIG["sample_rate"]
chunk_duration = CONFIG["chunk_duration"]


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

        with open(CONFIG["live_translation_file"], "a") as f:
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
        "ollama", "run", CONFIG["ollama_model"],
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
summary = summarize_with_ollama(CONFIG["live_translation_file"])

if summary:
    print("\nSummary of the meeting:")
    print(summary)

    # Save the summary to a file
    with open(CONFIG["summary_output_file"], "w") as f:
        f.write(summary)
    print(f"Summary saved to '{CONFIG['summary_output_file']}'")
else:
    print("Failed to generate summary.")


# Clean up the translation file
def cleanup_files():
    try:
        os.remove(CONFIG["live_translation_file"])
        print(f"Cleaned up temporary translation file: {CONFIG['live_translation_file']}")
    except FileNotFoundError:
        print("No temporary file to clean up.")
    except Exception as e:
        print(f"Error cleaning up file: {e}")


cleanup_files()
