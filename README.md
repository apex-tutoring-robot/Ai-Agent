# Jarvis 🤖 - AI-Powered Raspberry Pi Tutoring Robot

Jarvis is a voice-activated AI tutoring assistant that runs on Raspberry Pi. It uses wake word detection, Azure AI services, and streaming audio for natural, low-latency conversations.

## 🌟 Features

- **Wake Word Detection**: Always listening for "Hey [wakeword]" (low CPU usage ~3%)
- **Voice Activity Detection**: Smart audio capture using WebRTC VAD
- **Azure AI Integration**: 
  - Speech-to-Text for accurate transcription
  - GPT-4 via Azure OpenAI for intelligent tutoring
- **Streaming Architecture**: Minimal latency with LLM → TTS streaming pipeline
- **Privacy Protection**: Built-in PII anonymization framework
- **Continuous Conversations**: Maintains conversation history for multi-turn interactions
- **USB Audio Support**: Configurable USB microphone and speaker devices

## 📋 Prerequisites

### Hardware
- Raspberry Pi (3B+ or newer recommended)
- USB microphone (or compatible audio input device)
- USB speaker or audio output device
- Internet connection

### Software
- Python 3.8 or newer
- Raspberry Pi OS (or compatible Linux distribution)

### Required Accounts & Keys
1. **Azure Cognitive Services** account (for Speech-to-Text and Text-to-Speech)
2. **Azure OpenAI** service access and deployment
3. **Picovoice Console** account (for wake word model)

## 🚀 Installation

### 1. Clone the Repository
```bash
cd ~/Documents/apex-code/Ai-Agent 2.0
cd Jarvis
```

### 2. Install System Dependencies

On Raspberry Pi:
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the template and edit with your credentials:
```bash
cp config/.env.template config/.env
nano config/.env
```

Fill in your Azure credentials:
- `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`
- `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_DEPLOYMENT`
- `PICOVOICE_ACCESS_KEY`

### 5. Configure USB Audio Devices

List available audio devices:
```bash
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"
```

Update `AUDIO_INPUT_DEVICE_INDEX` and `AUDIO_OUTPUT_DEVICE_INDEX` in `.env` with your USB device indices.

### 6. Create Wake Word Model

1. Go to [Picovoice Console](https://console.picovoice.ai/)
2. Create a new wake word: "Hey Jarvis"
3. Select platform: Raspberry Pi
4. Download the `.ppn` file
5. Save it to `config/Hey-Jarvis_en_raspberry-pi_v3_0_0.ppn`

### 7. Configure System Prompt

Edit `config/system_prompt.txt` to define Jarvis's tutoring behavior and personality.

Example:
```
You are Jarvis, a patient and encouraging AI tutor. Your goal is to help students 
learn through guided questions rather than giving direct answers. Be friendly, 
supportive, and adapt your explanations to the student's level.
```

## 🎯 Usage

### Run Jarvis
```bash
cd src
python main.py
```

### Interaction Flow
1. Wait for "Listening for wake word..." message
2. Say "Hey Jarvis"
3. Speak your question or request
4. Pause when finished speaking (VAD will detect silence)
5. Jarvis will respond with audio
6. Continue the conversation by saying "Hey Jarvis" again

### Stop Jarvis
Press `Ctrl+C` to gracefully shutdown.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Jarvis DATA FLOW                      │
└─────────────────────────────────────────────────────────────┘

🔊 Wake Word Detection
           ↓
🎤 Audio Capture (VAD)
           ↓
☁️ Speech-to-Text (Azure)
           ↓
🛡️ Privacy Protection (PII Anonymization)
           ↓
🧠 LLM Processing (Azure OpenAI) ───→ Streaming text chunks
           ↓                                      ↓
📝 Conversation History              🎯 TTS Synthesis (Azure)
                                                  ↓
                                     🔈 Audio Playback (Streaming)
                                                  ↓
                                     🔄 Return to Wake Word Listening
```

### Project Structure
```
Jarvis/
├── src/
│   ├── audio/
│   │   ├── wake_word.py          # Wake word detection
│   │   ├── single_turn_vad.py    # VAD-based audio capture
│   │   └── playback.py           # Streaming audio playback
│   ├── azure_services/
│   │   ├── stt_client.py         # Speech-to-Text
│   │   ├── llm_client.py         # Azure OpenAI streaming
│   │   └── tts_client.py         # Text-to-Speech streaming
│   ├── privacy/
│   │   └── privacy_manager.py    # PII anonymization (placeholder)
│   ├── conversation/
│   │   └── state_manager.py      # Conversation history
│   └── main.py                   # Main orchestrator
├── config/
│   ├── .env.template             # Environment variables template
│   └── system_prompt.txt         # LLM system prompt
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_SPEECH_KEY` | Azure Speech Services API key | ✅ |
| `AZURE_SPEECH_REGION` | Azure region (e.g., eastus) | ✅ |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key | ✅ |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | ✅ |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name | ✅ |
| `PICOVOICE_ACCESS_KEY` | Picovoice access key | ✅ |
| `AUDIO_INPUT_DEVICE_INDEX` | USB mic device index | ✅ |
| `AUDIO_OUTPUT_DEVICE_INDEX` | USB speaker device index | ✅ |
| `SAMPLE_RATE` | Audio sample rate (16000) | No |
| `VAD_AGGRESSIVENESS` | VAD sensitivity 0-3 (3) | No |
| `SILENCE_TIMEOUT_MS` | Silence timeout (1500ms) | No |
| `MAX_CONVERSATION_HISTORY` | Max messages (20) | No |
| `TTS_VOICE` | Azure TTS voice name | No |
| `LOG_LEVEL` | Logging level (INFO) | No |

## 🛡️ Privacy Implementation

The `PrivacyManager` is provided as a placeholder. To implement PII anonymization:

1. Edit `src/privacy/privacy_manager.py`
2. Implement `anonymize()` method to detect and mask:
   - Names
   - Email addresses
   - Phone numbers
   - Addresses
   - Other personally identifiable information
3. Optionally implement `deanonymize()` for response mapping

## 🧪 Testing Components

Each component can be tested individually:

```bash
# Test wake word detection
python src/audio/wake_word.py

# Test VAD audio capture
python src/audio/single_turn_vad.py

# Test audio playback
python src/audio/playback.py

# Test Speech-to-Text
python src/azure_services/stt_client.py

# Test LLM streaming
python src/azure_services/llm_client.py

# Test Text-to-Speech
python src/azure_services/tts_client.py

# Test conversation manager
python src/conversation/state_manager.py
```

## 🐛 Troubleshooting

### Wake Word Not Detected
- Check `PICOVOICE_ACCESS_KEY` is correct
- Verify wake word `.ppn` file exists and path is correct
- Test microphone with: `python src/audio/wake_word.py`
- Try adjusting sensitivity in wake word detector

### No Audio Output
- Run device enumeration: `python src/audio/playback.py`
- Verify `AUDIO_OUTPUT_DEVICE_INDEX` matches your USB speaker
- Check speaker volume and connections

### Speech Recognition Errors
- Verify Azure credentials in `.env`
- Check internet connection
- Ensure `SAMPLE_RATE` is valid for WebRTC VAD (8000, 16000, 32000, or 48000)

### High Latency
- Check internet connection speed
- Reduce `SILENCE_TIMEOUT_MS` for faster cutoff
- Use faster Azure region closer to your location
- Verify USB audio devices support the configured sample rate

### PyAudio Installation Issues
```bash
sudo apt-get install -y portaudio19-dev
pip install --upgrade pip
pip install pyaudio
```

## 📊 Performance

- **Wake Word Detection**: ~3% CPU usage (continuous listening)
- **End-to-End Latency**: ~2-4 seconds (network dependent)
  - STT: ~0.5-1s
  - LLM: ~1-2s (streaming starts immediately)
  - TTS: ~0.5-1s (per sentence, streamed)

## 🔄 Future Enhancements

- [ ] Offline wake word alternatives
- [ ] Local LLM support for privacy
- [ ] Multi-language support
- [ ] Custom voice training
- [ ] Conversation analytics
- [ ] Session persistence and replay
- [ ] Advanced PII detection

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing structure and style
- Components remain modular and testable
- Documentation is updated accordingly

## 📞 Support

For issues related to:
- **Azure Services**: [Azure Support](https://azure.microsoft.com/support/)
- **Picovoice**: [Picovoice Docs](https://picovoice.ai/docs/)
- **Raspberry Pi**: [Raspberry Pi Forums](https://forums.raspberrypi.com/)

---

**Built with ❤️ for education**
