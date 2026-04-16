# Speakernotes Whisper

Automatically generate speaker notes from presentation dry-run recordings. Upload an audio or video file of your practice run, provide your slide titles, and get AI-generated speaker notes — ready to paste or embed directly into your PowerPoint.

## How it works

1. **Upload** an audio/video recording of your presentation dry-run
2. **Transcribe** the recording using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (runs locally)
3. **Provide slide titles** — type them in or upload a `.pptx` file to extract them automatically
4. **Generate notes** — either via the Claude API (automatic) or by copying a prompt to use on claude.ai
5. **Download** the notes as a `.txt` file or embedded into a new `.pptx` file

## Requirements

- Python 3.12+
- [ffmpeg](https://ffmpeg.org/) (for audio/video processing)
- An [Anthropic API key](https://console.anthropic.com/) (optional — you can use "Copy prompt" mode without one)

## Installation

```bash
# Install ffmpeg (macOS)
brew install ffmpeg

# Clone the repo
git clone https://github.com/your-username/speakernotes-whisper.git
cd speakernotes-whisper

# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

If you use [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root to set your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Alternatively, you can paste the key directly into the app's sidebar at runtime.

## Usage

```bash
speakernotes-whisper
```

This opens the app in your browser at `http://localhost:8501`.

### Sidebar options

| Setting | Description |
|---|---|
| **Mode** | `Claude API` — generates notes automatically; `Copy prompt` — builds a prompt you can paste into your AI tool of choice. |
| **Whisper model** | Larger models are more accurate but slower. `base` is a good default. |
| **Claude model** | Choose between Sonnet (balanced), Opus (best quality), or Haiku (fastest/cheapest) |

### Supported file formats

Audio/video: `.mp3`, `.mp4`, `.wav`, `.m4a`, `.ogg`, `.webm`, `.mkv`, `.mov`

## Development

Run the test suite:

```bash
pytest
```

### Project structure

```
speakernotes_whisper/
├── app.py           # Streamlit UI
├── transcribe.py    # Whisper transcription (with CUDA/MPS/CPU auto-detection)
├── claude_client.py # Claude API integration and prompt assembly
└── pptx_writer.py   # PowerPoint title extraction and note embedding
tests/
├── test_transcribe.py
├── test_claude_client.py
└── test_pptx_writer.py
```

## License

MIT
