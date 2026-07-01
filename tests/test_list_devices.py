"""
Audio Device Enumeration Script
Lists all available audio input and output devices on the Raspberry Pi.
Useful for identifying device indices for ReSpeaker 2-Mic Pi HAT and USB speakers.
"""

import os
import sys

import pyaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from audio.device_finder import resolve_input_device, resolve_output_device, DeviceNotFoundError


def list_audio_devices():
    """List all available audio input and output devices."""
    pa = pyaudio.PyAudio()
    
    print("\n" + "="*70)
    print("AUDIO DEVICE ENUMERATION - Raspberry Pi")
    print("="*70)
    print(f"\nTotal devices found: {pa.get_device_count()}\n")
    
    input_devices = []
    output_devices = []
    
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            
            print(f"Device {i}: {info['name']}")
            print(f"  Host API: {pa.get_host_api_info_by_index(info['hostApi'])['name']}")
            print(f"  Max Input Channels: {info['maxInputChannels']}")
            print(f"  Max Output Channels: {info['maxOutputChannels']}")
            print(f"  Default Sample Rate: {int(info['defaultSampleRate'])} Hz")
            
            # Check if this is ReSpeaker HAT
            if 'seeed' in info['name'].lower() or 'respeaker' in info['name'].lower():
                print(f"  ⭐ ReSpeaker 2-Mic Pi HAT detected!")
            
            # Check if this is USB device
            if 'usb' in info['name'].lower():
                print(f"  🔌 USB Audio Device detected!")
            
            # Categorize devices
            if info['maxInputChannels'] > 0:
                input_devices.append((i, info['name'], info['maxInputChannels']))
            if info['maxOutputChannels'] > 0:
                output_devices.append((i, info['name'], info['maxOutputChannels']))
            
            print()
        
        except Exception as e:
            print(f"  Error reading device {i}: {e}\n")
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n📥 INPUT DEVICES ({len(input_devices)}):")
    if input_devices:
        for idx, name, channels in input_devices:
            print(f"  [{idx}] {name} ({channels} channels)")
    else:
        print("  No input devices found")
    
    print(f"\n📤 OUTPUT DEVICES ({len(output_devices)}):")
    if output_devices:
        for idx, name, channels in output_devices:
            print(f"  [{idx}] {name} ({channels} channels)")
    else:
        print("  No output devices found")
    
    print("\n" + "="*70)
    print("AUTO-RESOLVED DEVICES (what the tests/ scripts will actually use)")
    print("="*70)
    print(
        "\nDevice indices shift across reboots as USB re-enumerates, so nothing "
        "here should be hardcoded to an index. The scripts in tests/ resolve "
        "devices by name every time they run instead, via src/audio/device_finder.py. "
        "This just shows what that resolution picks right now, given the name hints "
        "in .env (AUDIO_INPUT_DEVICE_NAME / AUDIO_OUTPUT_DEVICE_NAME). Note: main.py "
        "has its own separate PipeWire (\"pulse\") auto-discovery and doesn't use "
        "these variables.\n"
    )

    pa2 = pyaudio.PyAudio()
    try:
        idx = resolve_input_device(pa2)
        info = pa2.get_device_info_by_index(idx)
        print(f"  Input  -> [{idx}] {info['name']}")
    except DeviceNotFoundError as e:
        print(f"  Input  -> ❌ {e}")

    try:
        idx = resolve_output_device(pa2)
        info = pa2.get_device_info_by_index(idx)
        print(f"  Output -> [{idx}] {info['name']}")
    except DeviceNotFoundError as e:
        print(f"  Output -> ❌ {e}")
    pa2.terminate()

    print(
        "\nIf either line is wrong or errors out, set AUDIO_INPUT_DEVICE_NAME / "
        "AUDIO_OUTPUT_DEVICE_NAME in .env to a substring of the device name you "
        "want from the lists above (e.g. AUDIO_INPUT_DEVICE_NAME=usb pnp)."
    )
    print("\n" + "="*70 + "\n")
    
    pa.terminate()
    return input_devices, output_devices


if __name__ == "__main__":
    list_audio_devices()
