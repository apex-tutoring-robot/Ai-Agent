#!/bin/bash
set -e

echo "=== Installing system dependencies ==="
sudo apt-get update && sudo apt-get install -y \
    libcap-dev \
    swig \
    python3-libcamera \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libspeex-dev \
    libspeexdsp-dev \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    ffmpeg

echo "=== Creating virtual environment with system site packages ==="
python3 -m venv --system-site-packages .venv

echo "=== Installing Python dependencies ==="
.venv/bin/pip install -r config/requirements.txt

echo "=== Setup complete. Activate with: source .venv/bin/activate ==="
