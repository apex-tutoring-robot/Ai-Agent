# Jarvis 🤖 - AI-Powered Raspberry Pi Tutoring Robot

Jarvis is a voice-activated AI tutoring robot for K-8 students, built to run on a Raspberry Pi with a display, microphone, speaker, and camera. It listens for a wake word, holds natural spoken conversations, remembers each student by name and face, teaches math/science topics on an animated whiteboard, and reacts with an expressive animated face.

## 🌟 Features

- **Wake Word Detection** - always listening via [openWakeWord](https://github.com/dscripka/openWakeWord) (open-source, no API key or account needed)
- **Voice Activity Detection** - WebRTC VAD-based audio capture, with barge-in/interruption handling and (on Linux/Pi) SpeexDSP echo cancellation
- **Memory** - remembers students by spoken name (fuzzy-matched) and by face (a photo-derived signature), and remembers each student's past tutoring sessions, mastery per concept, and recent misconceptions
- **Tutoring Loop** - greets, checks in on a returning student, teaches a topic, tests understanding with a comprehension check, hints/re-explains as needed, and wraps up naming what was covered
- **Spaced Retrieval Practice** - periodically re-surfaces a concept the student hasn't reviewed recently, instead of only ever moving forward
- **Homework Help** - on request, takes a photo of a physical homework paper, reads it via vision, and walks the student through solving it themselves (hints if they're stuck)
- **Animated Whiteboard** - writes out equations step by step (formula → substitution → result), draws labeled geometry diagrams, underlines the final answer, and sketches + animates real-world objects (a car driving, a person walking, a ball, a plant growing) for non-geometry word problems
- **Expressive Face** - 6 emotional states (happy, sad, excited, confused, listening, thinking) plus continuous lip-sync driven by actual audio output, rendered in a Qt GUI (fullscreen on the Pi)
- **Volume Control by Voice** - "louder" / "quieter" / "normal", handled as an LLM tool call
- **Content Safety** - a fast synchronous keyword pre-check plus async NeMo Guardrails topic/safety checking on every response
- **Azure AI Integration** - Speech-to-Text, GPT-4o-mini (via Azure OpenAI, including vision for Homework Help) for tutoring/evaluation, and streaming Text-to-Speech
- **Streaming Architecture** - LLM → TTS → audio playback all stream, so speech starts before the full response is generated
- **Privacy Protection** - PII anonymization before anything is logged or sent upstream

## 📋 Prerequisites

### Hardware
- Raspberry Pi (a real display, mic, and speaker are needed for the full experience; a headless/no-GUI mode also works - see `FACE_ENABLED` below)
- Arducam 5MP (OV5647) CSI camera module - used for the student's profile photo and for Homework Help's paper capture (optional: everything else works without a camera, just without those two features)
- USB or onboard microphone and speaker
- Internet connection

### Software
- Python 3.13
- Raspberry Pi OS (or compatible Linux) for the real deployment; also runs on Windows for development (no camera or SpeexDSP support there - see "What's not covered" under Testing)

### Required Accounts & Keys
1. **Azure Cognitive Services** account (Speech-to-Text and Text-to-Speech)
2. **Azure OpenAI** service access and a GPT-4o-mini-class deployment (needs vision support for Homework Help)

No wake-word account is needed - openWakeWord's models download automatically on first run.

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/apex-tutoring-robot/Ai-Agent.git
cd Ai-Agent
```

### 2. Install System Dependencies (Raspberry Pi / Linux)
```bash
sudo apt-get update
sudo apt-get install -y \
    portaudio19-dev libasound2-dev python3-pyqt5 \
    libportaudio2 libportaudiocpp0 libspeexdsp-dev \
    python3-dev libopenblas-dev liblapack-dev libgtk-3-0 ffmpeg \
    libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libx11-xcb1
sudo apt install -y python3-picamera2   # camera support - must be apt, not pip
```

### 3. Install Python Dependencies
The real, current dependency list lives in `config/requirements.txt` (the root `requirements.txt` predates most of this project and is stale):
```bash
pip install -r config/requirements.txt
```
On the Pi, prefer the apt-installed `python3-pyqt5` over pip's PyQt5 wheel (pip has to compile it from source there and can be very slow/fail) - the requirements file assumes this.

### 4. Configure Environment Variables

Copy the template and fill in your real credentials:
```bash
cp .env.example .env
nano .env
```

Required: `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`.

### 5. Configure Audio Devices

List available audio devices:
```bash
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"
```
Set `AUDIO_INPUT_DEVICE_INDEX` / `AUDIO_OUTPUT_DEVICE_INDEX` in `.env` if PulseAudio auto-detection doesn't pick the right device.

### 6. Configure the System Prompt

Edit `config/system_prompt.txt` to adjust Jarvis's tutoring persona/tone.

## 🎯 Usage

### Run Jarvis
From the repo root (this is the convention `main.py` itself expects - see its own comments on why `cd src && python main.py` isn't used):
```bash
python3 src/main.py
```

### Interaction Flow
1. Say the wake word ("Hey Jarvis" - openWakeWord's built-in pretrained model)
2. Speak your question or request
3. Pause when finished (VAD detects silence)
4. Jarvis responds with audio, an animated face, and (for teaching moments) an animated whiteboard
5. Continue the conversation, or say the wake word again after a pause

### First Meeting a New Student
Say **"Voice recognition, [name]"** - Jarvis takes a photo (for face-based recognition later), then asks a few onboarding questions (grade, school, favorite subject, hobbies, what's hard). A student it already knows gets a "Welcome back" and a quick warm-up check-in instead.

### Homework Help
Say something containing **"homework"** (e.g. "Can you help me with my homework?") - Jarvis asks to see it, takes a photo, reads the problem via vision, and asks the student to try solving it, with the same hint/re-explain behavior as a normal teaching check.

### Stop Jarvis
Press `Ctrl+C` to gracefully shut down.

## 🏗️ Architecture

```
🔊 Wake Word (openWakeWord)
           ↓
🎤 Audio Capture (VAD, echo cancellation on Pi)
           ↓
☁️ Speech-to-Text (Azure)
           ↓
🛡️ Privacy Protection (PII anonymization)
           ↓
🔀 Turn routing: profile command? homework-help request? math/teaching
   question? onboarding/check-in reply? → otherwise, normal chat
           ↓
🧠 Azure OpenAI (teaching plan / answer evaluation / normal chat,
   with tool calls for volume control and curriculum search)
           ↓                                    ↓
📝 Conversation history &          🎨 Whiteboard draw actions +
   per-student memory (SQLite)        😀 Face expression + lip-sync
                                                 ↓
                                    🎯 Streaming TTS (Azure) → 🔈 Playback
                                                 ↓
                                    🔄 Back to wake-word listening
```

### Project Structure
```
Ai-Agent/
├── src/
│   ├── main.py                      # Orchestrator: wake word, turn routing, GUI wiring
│   ├── session.py                   # Per-conversation state (concepts covered, etc.)
│   ├── audio/
│   │   ├── wake_word.py             # openWakeWord detection
│   │   ├── continuous_vad.py        # VAD, barge-in, echo cancellation
│   │   └── playback.py              # Streaming audio playback + RMS-driven lip-sync
│   ├── azure_services/
│   │   ├── stt_client.py            # Speech-to-Text
│   │   ├── llm_client.py            # Azure OpenAI: chat, teaching plans, answer
│   │   │                            #   evaluation, vision (extract_image_content)
│   │   ├── local_llm_client.py      # llama.cpp fallback if Azure is unreachable
│   │   └── tts_client.py            # Streaming Text-to-Speech
│   ├── profiles/
│   │   ├── profile_manager.py       # SQLite-backed student profiles, mastery, memory
│   │   └── camera_capture.py        # Arducam photo capture (Pi/Linux only)
│   ├── identity/                    # Name + face matching/resolution for "who is this"
│   ├── tutor/                       # Decision engine, review scheduler, question engine
│   ├── curriculum/                  # Curriculum graph (concept prerequisites)
│   ├── expression/                  # Tutoring-moment -> face expression + TTS delivery
│   ├── affect/                      # Signals from how a student answers (hesitation, etc.)
│   ├── guardrails/                  # Fast keyword pre-check + NeMo Guardrails config
│   ├── knowledge/                   # Textbook/curriculum search (RAG)
│   ├── privacy/                     # PII anonymization
│   ├── conversation/                # Conversation history state manager
│   └── visuals/
│       ├── scene_planner.py         # Picks a real-world-object icon for word problems
│       ├── faces/                   # Face art + the legacy cv2-based FaceAnimator
│       └── ui/                      # PyQt5 GUI: main window, face widget, whiteboard,
│                                     #   scene graph/view, volume HUD
├── config/
│   ├── requirements.txt             # The real, current dependency list
│   ├── system_prompt.txt            # LLM system prompt
│   └── guardrails/                  # NeMo Guardrails config
├── tests/unit/                      # pytest suite (see Testing below)
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_SPEECH_KEY` | Azure Speech Services API key | ✅ |
| `AZURE_SPEECH_REGION` | Azure region (e.g., eastus) | ✅ |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | ✅ |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | ✅ |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name (needs vision support) | ✅ |
| `AUDIO_INPUT_DEVICE_INDEX` | Mic device index, if not auto-detected | No |
| `AUDIO_OUTPUT_DEVICE_INDEX` | Speaker device index, if not auto-detected | No |
| `FACE_ENABLED` | `true`/`false` - force the GUI on/off (default: on for Windows; on Linux, on only if `DISPLAY`/`WAYLAND_DISPLAY` is already set) | No |
| `DISPLAY_BACKEND` | `physical` (`DISPLAY=:0`), `vnc` (`DISPLAY=:1`), or `auto` (Linux only) | No |
| `WAKE_WORD_MODEL` | openWakeWord model name (default: its "hey jarvis" pretrained model) | No |
| `WAKE_WORD_THRESHOLD` | Wake word detection sensitivity (0.5) | No |
| `ENABLE_SPEEX_NOISE_SUPPRESSION` | Echo cancellation via SpeexDSP (Linux/Pi only) | No |
| `SAMPLE_RATE` | Audio sample rate (16000) | No |
| `VAD_AGGRESSIVENESS` | VAD sensitivity 0-3 (3) | No |
| `MAX_CONVERSATION_HISTORY` | Max messages kept in the LLM context window (20) | No |
| `TTS_VOICE` | Azure TTS voice name | No |
| `GUARDRAILS_CONFIG_PATH` | Path to the NeMo Guardrails config (`config/guardrails`) | No |
| `LOCAL_LLM_MODEL_PATH` | `.gguf` model path, enables the local-LLM fallback if Azure is down | No |
| `LOG_LEVEL` | Logging level (INFO) | No |

A number of spoken-phrase overrides (`WARMUP_CHECKIN_MESSAGE`, `DISCOVERY_MESSAGE`, `IDENTITY_CHECKIN_MESSAGE`, `GOODBYE_MESSAGE`, `SLEEP_MESSAGE`) and a few more timing/threshold knobs also exist with sane defaults - see `os.getenv(...)` calls in `src/main.py` for the full list.

## 🛡️ Privacy

`PrivacyManager.anonymize()` runs on every piece of user speech before it's logged or sent to Azure - see `src/privacy/privacy_manager.py` for what it currently detects/masks.

## 🧪 Testing

The project has a real pytest suite covering the tutoring logic, memory/profile system, expression mapping, whiteboard/canvas math, and the scene planner - run it from the repo root:
```bash
pytest
```
(`pytest.ini` puts `src/` on the path automatically, matching how `main.py` itself imports things.)

What's **not** covered by the automated suite, and needs a real device to verify:
- Wake word / microphone behavior (never tested with real audio input, only synthetic text driving the bot directly)
- Camera capture (`picamera2` isn't importable off-Pi at all)
- SpeexDSP echo cancellation (Linux/Pi only - Windows silently falls back to a simpler "digital ducking" approach)
- Full-screen rendering on the Pi's actual display resolution

## 🐛 Troubleshooting

### Wake Word Not Detected
- Check the mic is selected correctly (`AUDIO_INPUT_DEVICE_INDEX`)
- Try adjusting `WAKE_WORD_THRESHOLD` (lower = more sensitive)

### No Face / Whiteboard GUI Appears
- On Linux/Pi, confirm `DISPLAY` (or `WAYLAND_DISPLAY`) is actually set, or set `FACE_ENABLED=true` explicitly
- Check the logs for "Failed to init face animation" - it falls back to headless rather than crashing

### No Audio Output
- Verify `AUDIO_OUTPUT_DEVICE_INDEX` matches your speaker
- Check speaker volume and physical connections

### Speech Recognition / LLM Errors
- Verify Azure credentials in `.env`
- Check internet connection

### Homework Help Says It Couldn't Read the Photo
- Make sure the paper is well-lit, in focus, and fills a good portion of the camera's view with one clearly written/printed problem

### PyAudio / PyQt5 Installation Issues (Pi)
Install both via `apt`, not `pip` (see step 2 above) - compiling either from source on a Pi is slow and can fail outright.

## 📊 Performance

- **End-to-End Latency**: a few seconds from end of speech to first audio response (network-dependent) - STT, then LLM streaming (starts speaking before the full response is generated), then TTS per sentence

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

- Code follows the existing module structure (see Project Structure above)
- Run `pytest` before pushing
- Keep documentation (this file, inline comments) in sync with what actually changed

## 📞 Support

- **Azure Services**: [Azure Support](https://azure.microsoft.com/support/)
- **openWakeWord**: [GitHub](https://github.com/dscripka/openWakeWord)
- **Raspberry Pi**: [Raspberry Pi Forums](https://forums.raspberrypi.com/)

---

**Built with ❤️ for education**
