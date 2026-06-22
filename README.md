# Jarvis — Voice AI Tutoring Robot

A voice-activated educational assistant that runs on Raspberry Pi. Jarvis listens for a wake word, transcribes speech via Azure STT, queries an Azure OpenAI deployment, and streams the response back as synthesized speech — behaving like a patient, conversational tutor.

## Hardware

- Raspberry Pi (tested on RPi 4/5 with 64-bit OS)
- USB microphone (tested with ReSpeaker Lite)
- USB speaker or audio output device
- (Optional) Raspberry Pi Camera Module for homework scanning

## Prerequisites

- Python 3.8+
- PipeWire audio server
- Azure account with:
  - Azure Cognitive Services (Speech-to-Text and Text-to-Speech)
  - Azure OpenAI deployment

## Installation

```bash
git clone <repo-url>
cd Ai-Agent
./setup.sh
```

`setup.sh` installs system packages, creates a Python virtual environment at `.venv/`, installs Python dependencies, and configures PipeWire WebRTC echo cancellation for full-duplex audio.

## Configuration

```bash
cp config/.env.template .env
```

Edit `.env` and fill in your credentials.

### Required variables

| Variable | Description |
|---|---|
| `AZURE_SPEECH_KEY` | Azure Speech Services API key |
| `AZURE_SPEECH_REGION` | Azure region (e.g. `eastus`) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | API version (e.g. `2024-12-01-preview`) |
| `AUDIO_INPUT_DEVICE_INDEX` | PyAudio input device index |

### Finding your audio device index

```bash
source .venv/bin/activate
python tests/test_list_devices.py
```

Set `AUDIO_INPUT_DEVICE_INDEX` to the index of the `pulse` or `pipewire` device. Do **not** set `AUDIO_OUTPUT_DEVICE_INDEX` — the code auto-discovers the PipeWire output so TTS audio flows through echo cancellation correctly.

### Wake word

The default wake word is `hey_jarvis` (built into openwakeword). To use a different built-in model, set `WAKE_WORD_MODEL` to the model name string. No external API key is required.

### Key optional variables

| Variable | Default | Description |
|---|---|---|
| `WAKE_WORD_THRESHOLD` | `0.5` | Wake word confidence threshold |
| `SILENCE_TIMEOUT_MS` | `1500` | Silence (ms) before ending speech capture |
| `BARGE_IN_ENERGY_THRESHOLD` | `1500` | RMS threshold to trigger barge-in |
| `BARGE_IN_ECHO_THRESHOLD` | `0.75` | Containment score to reject bot-echo barge-ins |
| `TTS_VOICE` | `en-US-JennyNeural` | Azure TTS voice name |
| `TTS_SPEECH_RATE` | `1.0` | Speech rate multiplier |
| `MAX_CONVERSATION_HISTORY` | `20` | Number of turns kept in LLM context |
| `FACE_ENABLED` | `true` | Show PyQt5 face animation window |
| `GUARDRAILS_ENABLED` | `false` | Enable NeMo Guardrails content filtering |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## LLM Tools

Jarvis exposes two tools to the language model. The LLM decides when to call them based on what the student says.

**`begin_onboarding`** — Starts or resumes a structured tutoring session. The LLM calls this when the student explicitly asks to begin a lesson, start a study session, or resume tutoring. It triggers a multi-step onboarding flow: syllabus ingestion, diagnostic questions to gauge prior knowledge, and personalised study plan generation. Not called for general knowledge questions.

**`capture_photo`** — Takes a photo using the robot's camera and feeds the image into the next LLM turn. The LLM calls this when the student wants Jarvis to look at something physical — a homework sheet, a problem on paper, a diagram. Requires a camera module; the image is extracted to text before being sent to the model to minimise token usage.

## Running

```bash
source .venv/bin/activate
cd src
python main.py
```

**Interaction flow:**
1. Jarvis prints `Listening for wake word...`
2. Say **"Hey Jarvis"**
3. Ask your question; pause when done (VAD detects silence automatically)
4. Jarvis responds aloud
5. Trigger the wake word again to continue
6. `Ctrl+C` to shut down

## Project Structure

```
Ai-Agent/
├── src/
│   ├── main.py                  # Main orchestrator
│   ├── audio/                   # Wake word detection, VAD, audio playback
│   ├── azure_services/          # STT, LLM, and TTS clients
│   ├── conversation/            # Conversation history management
│   ├── vision/                  # Camera and homework scanning
│   ├── memory/                  # Study session persistence
│   ├── guardrails/              # NeMo Guardrails integration
│   └── visuals/                 # PyQt5 UI and face animation
├── config/
│   ├── .env.template            # Environment variable template
│   ├── system_prompt.txt        # Tutor persona and instructions
│   ├── requirements.txt         # Python dependencies
│   └── guardrails/              # Guardrails rules (config.yml, main.co)
├── tests/                       # Audio and component tests
├── logs/                        # Timestamped run logs (auto-created)
├── setup.sh                     # One-shot installer
└── .env                         # Runtime configuration (not committed)
```

## Testing Audio

```bash
source .venv/bin/activate

# Test microphone input
python tests/test_microphone.py

# Test speaker output
python tests/test_speaker.py
```

## Tuning

**Bot responds to its own voice (ghost turns):** Lower `BARGE_IN_ECHO_THRESHOLD` toward `0.60` to make the echo guard more aggressive.

**Barge-in fires too easily:** Raise `BARGE_IN_ENERGY_THRESHOLD` or increase `BARGE_IN_CHUNKS_NEEDED`.

**Speech capture cuts off too soon:** Increase `SILENCE_TIMEOUT_MS`.
