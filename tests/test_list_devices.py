"""
Audio Device Enumeration Script
Lists all available audio input and output devices on the Raspberry Pi.
Useful for identifying device indices for ReSpeaker 2-Mic Pi HAT and USB speakers.
"""

import pyaudio


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
    print("RECOMMENDATIONS FOR .env FILE")
    print("="*70)
    
    # Find ReSpeaker for input
    respeaker_idx = None
    for idx, name, _ in input_devices:
        if 'seeed' in name.lower() or 'respeaker' in name.lower():
            respeaker_idx = idx
            break
    
    # Find USB or default output
    usb_output_idx = None
    for idx, name, _ in output_devices:
        if 'usb' in name.lower():
            usb_output_idx = idx
            break
    
    if respeaker_idx is not None:
        print(f"\nFor ReSpeaker 2-Mic Pi HAT input:")
        print(f"  AUDIO_INPUT_DEVICE_INDEX={respeaker_idx}")
    else:
        print(f"\n⚠️  ReSpeaker not detected. Using first available input device.")
        if input_devices:
            print(f"  AUDIO_INPUT_DEVICE_INDEX={input_devices[0][0]}")
    
    if usb_output_idx is not None:
        print(f"\nFor USB Speaker output:")
        print(f"  AUDIO_OUTPUT_DEVICE_INDEX={usb_output_idx}")
    else:
        print(f"\n⚠️  USB speaker not detected. Using first available output device.")
        if output_devices:
            print(f"  AUDIO_OUTPUT_DEVICE_INDEX={output_devices[0][0]}")
    
    print("\n" + "="*70 + "\n")
    
    pa.terminate()
    return input_devices, output_devices


if __name__ == "__main__":
    list_audio_devices()
