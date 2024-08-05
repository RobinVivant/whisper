import json
import logging
import os
import subprocess
import time
from typing import Dict, Any

import numpy as np
import sounddevice as sd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_config() -> Dict[str, Any]:
    with open('config.json', 'r') as config_file:
        return json.load(config_file)


CONFIG = load_config()


def load_model_and_processor(model_name: str):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(DEVICE)
    return processor, model


# Load model and processor
model_name = CONFIG["model_name"]
processor, model = load_model_and_processor(model_name)

# Audio settings
sample_rate = CONFIG["sample_rate"]
chunk_duration = CONFIG["chunk_duration"]


def process_audio(audio_chunk: torch.Tensor) -> str:
    # Normalize the audio chunk
    audio_chunk = audio_chunk / torch.max(torch.abs(audio_chunk))

    input_features = processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt").input_features
    input_features = input_features.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(input_features)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


def audio_callback(indata: np.ndarray, status: sd.CallbackFlags) -> None:
    if status:
        logging.error(f"Error in audio stream: {status}")
        return

    try:
        audio_chunk = torch.from_numpy(indata[:, 0]).float()
        transcription = process_audio(audio_chunk)

        with open(CONFIG["live_translation_file"], "a") as f:
            f.write(transcription + "\n")

        logging.info(f"Translated: {transcription}")
    except Exception as e:
        logging.error(f"Error processing audio: {e}")


def summarize_with_ollama(file_path: str) -> str:
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
            logging.error(f"Error parsing Ollama output: {je}")
            logging.error(f"Raw output: {result.stdout}")
            return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Ollama: {e}")
        logging.error(f"Ollama stderr: {e.stderr}")
        return None


def cleanup_files():
    try:
        os.remove(CONFIG["live_translation_file"])
        logging.info(f"Cleaned up temporary translation file: {CONFIG['live_translation_file']}")
    except FileNotFoundError:
        logging.info("No temporary file to clean up.")
    except Exception as e:
        logging.error(f"Error cleaning up file: {e}")


def main():
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate,
                            blocksize=int(sample_rate * chunk_duration), dtype='float32')

    with stream:
        logging.info("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Stopped listening. Generating summary...")

    # Generate summary after recording stops
    summary = summarize_with_ollama(CONFIG["live_translation_file"])

    if summary:
        logging.info("\nSummary of the meeting:")
        logging.info(summary)

        # Save the summary to a file
        with open(CONFIG["summary_output_file"], "w") as f:
            f.write(summary)
        logging.info(f"Summary saved to '{CONFIG['summary_output_file']}'")
    else:
        logging.error("Failed to generate summary.")

    cleanup_files()


if __name__ == "__main__":
    main()
