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

echo "=== Configuring PipeWire WebRTC AEC (ReSpeaker Lite, both directions) ==="

# The ReSpeaker Lite's own onboard "hardware AEC" (XMOS XU316 DSP) is a known
# broken/unreliable firmware feature over USB (confirmed both empirically —
# identical echo leak at 100%/120%/200% volume — and via Seeed's own product
# forum). So software AEC is still required, but now both the capture (mic)
# and playback (speaker) sides point at the ReSpeaker Lite itself, since the
# speaker has been physically moved to its 2-pin terminal.
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

      # Mic input: ReSpeaker Lite
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
      # Relay to real hardware speaker: ReSpeaker Lite's own 2-pin terminal
      playback.props = {
        node.name        = "echo-cancel-playback"
        node.description = "Echo Cancellation Playback"
        target.object    = "alsa_output.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo"
      }
    }
  }
]
PIPEWIRE_EOF

# WirePlumber routing rules: enforce capture → ReSpeaker input, playback → ReSpeaker output
# (WirePlumber 0.5 does not honour target.object from PipeWire module props alone)
mkdir -p ~/.config/wireplumber/wireplumber.conf.d
cat > ~/.config/wireplumber/wireplumber.conf.d/99-echo-cancel-routing.conf << 'WP_EOF'
wireplumber.rules = [
  {
    matches = [ { node.name = "echo-cancel-playback" } ]
    actions = {
      update-props = {
        target.object = "alsa_output.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo"
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

echo "=== Configuring HDMI display (JRP7002 7-inch, micro-HDMI port closest to USB-C) ==="

# The JRP7002 display doesn't assert the HDMI HPD pin, so the Pi firmware
# reports both HDMI connectors as disconnected and the Wayland compositor
# falls back to a virtual headless output. Two fixes are required:
#
# 1. hdmi_force_hotplug=1 in config.txt — tells the firmware to assume a
#    display is present (must appear before dtoverlay=vc4-kms-v3d).
# 2. video=HDMI-A-1:1280x720@60e in cmdline.txt — forces the KMS driver to
#    enable the connector at boot without waiting for a HPD signal ('e' flag).
#    1280x720 is the closest standard VESA mode to the panel's 1024x600 native.
#
# Display also requires a separate micro-USB power cable in addition to HDMI.

CONFIG_TXT=/boot/firmware/config.txt
if ! grep -q "hdmi_force_hotplug" "$CONFIG_TXT"; then
    sudo sed -i 's/# Enable DRM VC4 V3D driver/# Force HDMI hotplug (JRP7002 display does not assert HPD pin)\nhdmi_force_hotplug=1\n\n# Enable DRM VC4 V3D driver/' "$CONFIG_TXT"
    echo "  Added hdmi_force_hotplug=1 to $CONFIG_TXT"
else
    echo "  hdmi_force_hotplug already set in $CONFIG_TXT"
fi

CMDLINE_TXT=/boot/firmware/cmdline.txt
if ! grep -q "video=HDMI-A-1" "$CMDLINE_TXT"; then
    sudo sed -i 's/rootwait/rootwait video=HDMI-A-1:1280x720@60e/' "$CMDLINE_TXT"
    echo "  Added video=HDMI-A-1:1280x720@60e to $CMDLINE_TXT"
else
    echo "  video=HDMI-A-1 already set in $CMDLINE_TXT"
fi

# labwc output config: use only HDMI-A-1 (port closest to USB-C power).
# Without this, labwc enables both forced connectors and splits the desktop.
mkdir -p ~/.config/labwc
if [ ! -f ~/.config/labwc/outputs.xml ]; then
    cat > ~/.config/labwc/outputs.xml << 'OUTPUTS_EOF'
<outputs>
  <output name="HDMI-A-1" enabled="true" />
  <output name="HDMI-A-2" enabled="false" />
</outputs>
OUTPUTS_EOF
    echo "  Wrote ~/.config/labwc/outputs.xml"
else
    echo "  ~/.config/labwc/outputs.xml already exists, skipping"
fi

echo "  NOTE: reboot required for config.txt and cmdline.txt changes to take effect"

echo "=== Installing Claude Code (comment out for production) ==="
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

echo "=== Setup complete. Activate with: source .venv/bin/activate ==="
