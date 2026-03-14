# PyAudio/ALSA Crash Issue on Raspberry Pi

## Problem
You're experiencing critical `malloc_consolidate()` crashes with PyAudio + ALSA on Raspberry Pi. This is a **known issue** with memory corruption in the underlying C libraries.

## Why It Happens
- PyAudio uses ALSA directly on Linux
- ALSA on Raspberry Pi is unstable with rapid stream creation/destruction
- Memory corruption happens at C level (below Python's exception handling)
- The error "[Errno -9999] Unanticipated host error" is PyAudio's way of saying ALSA failed catastrophically

## Solutions (in order of recommendation)

### Option 1: Use PulseAudio (Recommended)
Install PulseAudio to add a stable layer between PyAudio and ALSA:

```bash
sudo apt-get update
sudo apt-get install pulseaudio pulseaudio-utils
pulseaudio --start
```

Then restart Chippy. PulseAudio handles ALSA's quirks much better.

### Option 2: Use sounddevice instead of PyAudio
Replace PyAudio with the `sounddevice` library which has better ALSA handling:

```bash
pip install sounddevice
```

This would require refactoring `playback.py` but is more stable on Raspberry Pi.

### Option 3: Lower Buffer Size
Edit `src/audio/playback.py` line ~106:

```python
chunk_size = 512  # Try smaller buffer (currently 1024)
```

Smaller buffers = less memory churn = possibly fewer crashes.

### Option 4: Use External Audio Process
Run audio playback in a separate subprocess so crashes don't kill Chippy:

```python
import subprocess
subprocess.Popen(['aplay', '-'], stdin=subprocess.PIPE)
```

### Option 5: Use USB Audio DAC
Your current USB audio device might have driver issues. Try a different USB audio adapter with better Linux support (e.g., devices with CM108/CM109 chips).

## Current Mitigations in Code
I've added:
- Signal handlers for SIGABRT/SIGSEGV (attempts cleanup before crash)
- PyAudio instance reuse (reduces ALSA churn)
- Comprehensive error handling
- Stream delays to let ALSA settle

But these can't prevent C-level memory corruption.

## Recommended Next Step
**Try PulseAudio first** (Option 1) - it's the simplest fix that often solves these issues.
