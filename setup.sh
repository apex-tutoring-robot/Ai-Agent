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
    ffmpeg \
    libasound2-plugins

echo "=== Creating virtual environment with system site packages ==="
python3 -m venv --system-site-packages .venv

echo "=== Installing Python dependencies ==="
.venv/bin/pip install -r config/requirements.txt

echo "=== Configuring PipeWire WebRTC AEC ==="

# PipeWire echo-cancel module: ReSpeaker mic + USB speaker + WebRTC AEC
mkdir -p ~/.config/pipewire/pipewire.conf.d
cat > ~/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf << 'PIPEWIRE_EOF'
context.modules = [
  {
    name = libpipewire-module-echo-cancel
    args = {
      library.name  = aec/libspa-aec-webrtc

      aec.args = {
        webrtc.gain_control       = true
        webrtc.extended_filter    = true
        webrtc.delay_agnostic     = true
        webrtc.high_pass_filter   = true
        webrtc.noise_suppression  = true
        webrtc.voice_detection    = false
      }

      monitor.mode = false

      # Mic input: ReSpeaker Lite (physically separated from the speaker)
      capture.props = {
        node.name        = "echo-cancel-capture"
        node.description = "Echo Cancellation Capture"
        target.object    = "alsa_input.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo"
      }
      # Virtual sink: apps send audio here; AEC uses it as the echo reference
      sink.props = {
        node.name        = "echo-cancel-sink"
        node.description = "Echo Cancellation Sink"
        media.class      = Audio/Sink
      }
      # AEC output: clean mic audio fed to STT
      source.props = {
        node.name        = "echo-cancel-source"
        node.description = "Echo Cancellation Source"
        media.class      = Audio/Source
      }
      # Relay to real hardware speaker: USB PnP Audio Device
      playback.props = {
        node.name        = "echo-cancel-playback"
        node.description = "Echo Cancellation Playback"
        target.object    = "alsa_output.usb-Solid_State_System_Co._Ltd._USB_PnP_Audio_Device_000000000000-00.analog-stereo"
      }
    }
  }
]
PIPEWIRE_EOF

# WirePlumber routing rules: enforce capture → ReSpeaker, playback → USB speaker
# (WirePlumber 0.5 does not honour target.object from PipeWire module props alone)
mkdir -p ~/.config/wireplumber/wireplumber.conf.d
cat > ~/.config/wireplumber/wireplumber.conf.d/99-echo-cancel-routing.conf << 'WP_EOF'
wireplumber.rules = [
  {
    matches = [ { node.name = "echo-cancel-playback" } ]
    actions = {
      update-props = {
        target.object = "alsa_output.usb-Solid_State_System_Co._Ltd._USB_PnP_Audio_Device_000000000000-00.analog-stereo"
        node.dont-reconnect = false
      }
    }
  }
  {
    matches = [ { node.name = "echo-cancel-capture" } ]
    actions = {
      update-props = {
        target.object = "alsa_input.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo"
        node.dont-reconnect = false
      }
    }
  }
]
WP_EOF

# Reload PipeWire with the new config
systemctl --user restart pipewire wireplumber pipewire-pulse
echo "Waiting for PipeWire AEC nodes..."
sleep 3

# Set echo-cancel-sink and echo-cancel-source as default audio devices so that
# all audio I/O through the 'pulse' ALSA device routes through the AEC chain.
for attempt in $(seq 1 10); do
    AEC_SOURCE=$(wpctl status 2>/dev/null | grep "echo-cancel-source" | grep -oE '[0-9]+' | head -1)
    AEC_SINK=$(wpctl status 2>/dev/null | grep "echo-cancel-sink" | grep -v "capture\|playback\|source" | grep -oE '[0-9]+' | head -1)

    if [ -n "$AEC_SOURCE" ] && [ -n "$AEC_SINK" ]; then
        wpctl set-default "$AEC_SOURCE" && echo "✅ echo-cancel-source ($AEC_SOURCE) set as default source"
        wpctl set-default "$AEC_SINK"   && echo "✅ echo-cancel-sink ($AEC_SINK) set as default sink"
        break
    fi
    echo "  Waiting for AEC nodes (attempt $attempt/10)..."
    sleep 1
done

echo "=== Installing Claude Code (comment out for production) ==="
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

echo "=== Setup complete. Activate with: source .venv/bin/activate ==="
