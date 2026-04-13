# Jarvis Hardware Test Scripts

This directory contains test scripts to verify your Raspberry Pi audio hardware setup (ReSpeaker 2-Mic Pi HAT and USB speaker) before running the full Jarvis application.

## Test Scripts

### 1. `test_list_devices.py` - Device Enumeration
Lists all available audio devices and recommends device indices for your setup.

**Usage:**
```bash
cd tests
python test_list_devices.py
```

**What it does:**
- Lists all audio input/output devices
- Detects ReSpeaker 2-Mic Pi HAT
- Detects USB audio devices
- Provides recommended device indices for `.env`

---

### 2. `test_speaker.py` - USB Speaker Test
Tests your USB speaker by playing musical tones.

**Usage:**
```bash
python test_speaker.py                    # Uses device from .env
python test_speaker.py --device 2         # Test specific device
python test_speaker.py --rate 44100       # Custom sample rate
```

**What it does:**
- Plays 4 musical test tones (A, C, E, G)
- Plays ascending frequency sweep
- Verifies speaker is working correctly

**Expected result:** You should hear clear musical tones

---

### 3. `test_microphone.py` - ReSpeaker Microphone Test
Tests your ReSpeaker 2-Mic Pi HAT microphone.

**Usage:**
```bash
python test_microphone.py                 # Uses device from .env
python test_microphone.py --device 1      # Test specific device
python test_microphone.py --duration 10   # Record for 10 seconds
```

**What it does:**
- Records audio for specified duration (default 5 seconds)
- Saves recording to `test_recording.wav`
- Plays back the recording
- Analyzes audio levels

**Expected result:** You should hear your voice played back clearly

---

### 4. `test_audio_pipeline.py` - Full Pipeline Test
Tests the complete audio flow: microphone → VAD → playback.

**Usage:**
```bash
python test_audio_pipeline.py
```

**What it does:**
- Captures speech using VAD (Voice Activity Detection)
- Automatically stops on silence
- Plays back captured audio (both blocking and streaming modes)
- Simulates Jarvis's actual audio pipeline

**Expected result:** Speak when prompted, pause, then hear your voice played back twice (regular and streaming)

---

## Quick Start Testing Guide

### Step 1: Find Your Devices
```bash
python test_list_devices.py
```

Copy the recommended device indices and update your `.env` file:
```
AUDIO_INPUT_DEVICE_INDEX=X   # ReSpeaker index
AUDIO_OUTPUT_DEVICE_INDEX=Y  # USB speaker index
```

### Step 2: Test Speaker
```bash
python test_speaker.py
```

✅ If you hear tones → Speaker is working!  
❌ If no sound → Check USB connection, volume, and device index

### Step 3: Test Microphone
```bash
python test_microphone.py
```

✅ If you hear your voice → Microphone is working!  
❌ If no sound or distorted → Check HAT connection and `alsamixer` settings

### Step 4: Test Full Pipeline
```bash
python test_audio_pipeline.py
```

✅ If VAD captures and plays back correctly → Hardware is ready for Jarvis!

---

## Troubleshooting

### ReSpeaker 2-Mic Pi HAT Not Detected

1. Check HAT is properly seated on GPIO pins
2. Verify HAT drivers are installed:
   ```bash
   sudo apt-get install -y seeed-voicecard
   sudo reboot
   ```
3. Check device appears in ALSA:
   ```bash
   arecord -l
   ```

### USB Speaker Not Working

1. Check USB connection
2. Test with system command:
   ```bash
   speaker-test -D hw:X,0 -c2
   ```
3. Adjust volume:
   ```bash
   alsamixer
   ```

### Audio Quality Issues

**Low volume:**
- Increase volume: `alsamixer` → Select device → Use arrow keys

**Distorted audio:**
- Lower input gain in `alsamixer`
- Try different sample rates (8000, 16000, 44100, 48000)

**Choppy playback:**
- Increase `CHUNK_SIZE` in `.env`
- Check CPU usage: `top`

### Buffer Overflow Errors

If you see buffer overflow warnings:
1. Reduce `CHUNK_SIZE` in `.env`
2. Try lower sample rate (16000 instead of 44100)
3. ReSpeaker works best at 16000 Hz

---

## ReSpeaker 2-Mic Pi HAT Specific Notes

### Supported Sample Rates
- **16000 Hz** - Recommended, native support
- 8000 Hz - Works well
- 44100 Hz - May have buffer issues
- 48000 Hz - Highest quality but requires more processing

### LED Indicators
The ReSpeaker HAT has built-in LEDs that should:
- Light up during recording
- Indicate audio activity
- If LEDs don't work, HAT may not be properly installed

### Volume Control
```bash
# Open mixer
alsamixer

# Select ReSpeaker device
F6 → Select your card

# Adjust levels
Use arrow keys to change volume
```

---

## Common Device Names

**ReSpeaker 2-Mic Pi HAT:**
- `seeed-2mic-voicecard`
- `seeed-voicecard`
- Contains "seeed" or "respeaker"

**USB Speakers (examples):**
- `USB Audio Device`
- `USB PnP Sound Device`
- Contains "USB"

---

## Test Output Files

Tests may create the following files:
- `test_recording.wav` - Microphone test recording
- `*.raw` - Raw audio data

These files are ignored by `.gitignore` and safe to delete.

---

## Next Steps

Once all tests pass:
1. ✅ Devices configured in `.env`
2. ✅ Speaker working
3. ✅ Microphone working
4. ✅ Full pipeline working

→ **Ready to run Jarvis!**
```bash
cd ../src
python main.py
```

Enjoy your AI tutoring robot!
