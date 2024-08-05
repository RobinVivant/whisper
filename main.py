import json
import logging
import os
import subprocess
import time
import signal
from typing import Dict, Any, List

import numpy as np
import sounddevice as sd
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

logging.info(f"Using device: {DEVICE}")


def load_config() -> Dict[str, Any]:
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        
        required_keys = ["model_name", "sample_rate", "chunk_duration", "ollama_model", "live_translation_file", "summary_output_file"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required key in config: {key}")
        
        return config
    except FileNotFoundError:
        logging.error("config.json file not found.")
        raise
    except json.JSONDecodeError:
        logging.error("Error parsing config.json. Please check the file format.")
        raise
    except KeyError as e:
        logging.error(str(e))
        raise

try:
    CONFIG = load_config()
except Exception as e:
    logging.error(f"Failed to load configuration: {e}")
    sys.exit(1)


def load_model_and_processor(model_name_param: str):
    logging.info(f"Loading model and processor from {model_name_param}")
    loaded_processor = AutoProcessor.from_pretrained(model_name_param)
    loaded_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_param)

    if DEVICE.type == "mps":
        # Convert the model to float32 for MPS compatibility
        loaded_model = loaded_model.float()

    loaded_model = loaded_model.to(DEVICE)
    logging.info(f"Model loaded and moved to device: {DEVICE}")
    return loaded_processor, loaded_model


# Load model and processor
model_name = CONFIG["model_name"]
processor, model = load_model_and_processor(model_name)

# Audio settings
sample_rate = CONFIG["sample_rate"]
chunk_duration = CONFIG["chunk_duration"]


def process_audio(audio_chunk: torch.Tensor) -> str:
    logging.debug(f"Processing audio chunk of shape: {audio_chunk.shape}")

    # Ensure audio_chunk is on CPU for numpy conversion
    audio_chunk = audio_chunk.cpu()

    # Normalize the audio chunk
    audio_chunk = audio_chunk / torch.max(torch.abs(audio_chunk))

    # Process on CPU
    input_features = processor(audio_chunk.numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features

    # Move to device for model inference
    input_features = input_features.to(DEVICE)

    logging.debug(f"Input features shape: {input_features.shape}")

    with torch.no_grad():
        generated_ids = model.generate(input_features, max_length=448)

    # Move generated_ids back to CPU for decoding
    generated_ids = generated_ids.cpu()

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logging.debug(f"Raw transcription: '{transcription}'")

    return transcription.strip()


def audio_callback(indata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
    if status:
        logging.error(f"Error in audio stream: {status}")
        return

    try:
        logging.debug(f"Received audio chunk of shape: {indata.shape}")
        if indata.size == 0:
            logging.warning("Received empty audio chunk")
            return
        audio_chunk = torch.from_numpy(indata[:, 0]).float()
        transcription = process_audio(audio_chunk)

        if transcription:
            with open(CONFIG["live_translation_file"], "a") as f:
                f.write(transcription + "\n")
            print(f"Transcribed: {transcription}")  # Display on console
        else:
            logging.warning("No transcription generated for this audio chunk.")
    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        logging.exception("Detailed error information:")


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
            return result.stdout  # Return raw output if JSON parsing fails
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Ollama: {e}")
        logging.error(f"Ollama stderr: {e.stderr}")
        return f"Error: {e.stderr}"


def cleanup_files():
    try:
        os.remove(CONFIG["live_translation_file"])
        logging.info(f"Cleaned up temporary translation file: {CONFIG['live_translation_file']}")
    except FileNotFoundError:
        logging.info("No temporary file to clean up.")
    except Exception as e:
        logging.error(f"Error cleaning up file: {e}")


def list_audio_devices() -> List[Dict[str, Any]]:
    devices = sd.query_devices()
    input_devices = [device for device in devices if device['max_input_channels'] > 0]
    return input_devices


def select_audio_device(devices: List[Dict[str, Any]]) -> int:
    print("Available input devices:")
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")

    while True:
        try:
            selection = int(input("Select the number of the input device to use: "))
            if 0 <= selection < len(devices):
                return selection
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    logging.info(f"Initializing audio stream with sample rate: {sample_rate}, chunk duration: {chunk_duration}")

    input_devices = list_audio_devices()
    device_index = select_audio_device(input_devices)
    selected_device = input_devices[device_index]

    logging.info(f"Selected input device: {selected_device['name']}")

    stream = sd.InputStream(callback=audio_callback,
                            device=device_index,
                            channels=1,
                            samplerate=sample_rate,
                            blocksize=int(sample_rate * chunk_duration),
                            dtype='float32')

    recording = True

    def stop_recording(signum, frame):
        nonlocal recording
        recording = False
        print("\nStopping recording...")

    signal.signal(signal.SIGINT, stop_recording)

    with stream:
        print("Listening... Press Ctrl+C to stop.")
        while recording:
            time.sleep(0.1)
        stream.stop()
        print("Stopped listening.")

    if os.path.exists(CONFIG["live_translation_file"]) and os.path.getsize(CONFIG["live_translation_file"]) > 0:
        print("Transcriptions found.")
        summarize = input("Do you want to generate a summary? (y/n): ").lower().strip() == 'y'
        
        if summarize:
            print("Generating summary...")
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
    else:
        print("No transcriptions were generated. The audio input might not have been captured correctly.")

    cleanup_files()


if __name__ == "__main__":
    main()
