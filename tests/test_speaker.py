"""
USB Speaker Test Script
Tests audio output through USB speaker by playing test tones at different frequencies.
"""

import sys
import os
import pyaudio
import struct
import math
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()


def generate_tone(frequency, duration, sample_rate=44100, amplitude=0.3):
    """
    Generate a sine wave tone.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)
    
    Returns:
        bytes: Raw audio data
    """
    samples = []
    num_samples = int(sample_rate * duration)
    
    for i in range(num_samples):
        # Generate sine wave
        value = amplitude * math.sin(2 * math.pi * frequency * i / sample_rate)
        # Convert to 16-bit signed integer
        sample = int(32767 * value)
        samples.append(struct.pack('h', sample))
    
    return b''.join(samples)


def test_speaker(device_index=None, sample_rate=44100):
    """
    Test speaker with multiple tones.
    
    Args:
        device_index: Output device index (None for default)
        sample_rate: Sample rate in Hz
    """
    print("\n" + "="*70)
    print("USB SPEAKER TEST")
    print("="*70)
    
    # Get device index from environment if not provided
    if device_index is None:
        device_index = int(os.getenv('AUDIO_OUTPUT_DEVICE_INDEX', 1))
    
    pa = pyaudio.PyAudio()
    
    # Show device info
    try:
        device_info = pa.get_device_info_by_index(device_index)
        print(f"\nTesting output device:")
        print(f"  Index: {device_index}")
        print(f"  Name: {device_info['name']}")
        print(f"  Max Channels: {device_info['maxOutputChannels']}")
        print(f"  Default Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
        print()
    except Exception as e:
        print(f"\n❌ Error accessing device {device_index}: {e}")
        print("\nAvailable output devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
        pa.terminate()
        return
    
    # Test tones
    test_tones = [
        (440, "A4 - Middle A"),
        (523, "C5 - Middle C"),
        (659, "E5"),
        (784, "G5"),
    ]
    
    try:
        # Open audio stream
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=device_index
        )
        
        print("🔊 Playing test tones...\n")
        
        for freq, note in test_tones:
            print(f"  Playing {note} ({freq} Hz)...")
            tone_data = generate_tone(freq, duration=1.0, sample_rate=sample_rate)
            stream.write(tone_data)
            time.sleep(0.3)  # Short pause between tones
        
        # Play ascending tone
        print("\n  Playing ascending tone sequence...")
        for freq in range(400, 800, 50):
            tone_data = generate_tone(freq, duration=0.2, sample_rate=sample_rate, amplitude=0.2)
            stream.write(tone_data)
        
        stream.stop_stream()
        stream.close()
        
        print("\n✅ Speaker test complete!")
        print("\nDid you hear the test tones?")
        print("  - 4 distinct musical notes (A, C, E, G)")
        print("  - An ascending tone sequence")
        print("\nIf you didn't hear anything:")
        print("  1. Check speaker volume")
        print("  2. Verify speaker is connected to correct USB port")
        print("  3. Try a different device index")
        print("  4. Check speaker works with: speaker-test -D hw:X,0 -c2")
    
    except Exception as e:
        print(f"\n❌ Error during playback: {e}")
    
    finally:
        pa.terminate()
    
    print("\n" + "="*70 + "\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test USB speaker output')
    parser.add_argument('--device', type=int, help='Output device index')
    parser.add_argument('--rate', type=int, default=44100, help='Sample rate (default: 44100)')
    
    args = parser.parse_args()
    
    test_speaker(device_index=args.device, sample_rate=args.rate)


if __name__ == "__main__":
    main()
