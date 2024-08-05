# Real-Time Speech Transcription and Summarization

## Overview

This project provides a real-time speech transcription and summarization tool using the Whisper model for transcription and Ollama for summarization. It captures audio input, transcribes it in real-time, and generates a summary of the entire conversation at the end.

## Features

- Real-time audio capture and transcription
- Live display of transcribed text
- Automatic summarization of the entire conversation
- Configurable settings via JSON file
- Support for different Whisper models
- Utilizes Ollama for text summarization

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- sounddevice
- numpy
- Ollama (for summarization)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required Python packages:
   ```
   pip install torch transformers sounddevice numpy
   ```

3. Install Ollama following the instructions at [Ollama's official website](https://ollama.ai/).

4. Configure the `config.json` file with your preferred settings.

## Configuration

Edit the `config.json` file to customize the following settings:

- `model_name`: The Whisper model to use (e.g., "aiola/whisper-medusa-v1")
- `sample_rate`: Audio sample rate (default: 16000)
- `chunk_duration`: Duration of each audio chunk in seconds (default: 5)
- `ollama_model`: Ollama model for summarization (e.g., "mistral-nemo:latest")
- `live_translation_file`: Temporary file for storing live transcriptions
- `summary_output_file`: File to save the final summary

## Usage

Run the main script:

```
python main.py
```

The program will start capturing audio from your default input device. Speak clearly, and you'll see the transcribed text appear in real-time. To stop the recording, press Ctrl+C. The program will then generate a summary of the entire conversation.

## How It Works

1. The script initializes the Whisper model and processor based on the configuration.
2. It sets up an audio stream to capture input from the microphone.
3. Each audio chunk is processed in real-time:
   - The audio is transcribed using the Whisper model.
   - The transcription is appended to a temporary file and displayed in the console.
4. When the user stops the recording (Ctrl+C), the script:
   - Stops the audio stream.
   - Sends the entire transcription to Ollama for summarization.
   - Displays the summary and saves it to a file.
5. Temporary files are cleaned up at the end.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI's Whisper](https://github.com/openai/whisper) for the speech recognition model
- [Ollama](https://ollama.ai/) for the summarization capabilities
- All contributors and open-source libraries used in this project

## Support

If you encounter any problems or have any questions, please open an issue in the GitHub repository.
