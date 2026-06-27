"""
ReSpeaker 2-Mic Pi HAT Microphone Test Script
Tests microphone input by recording audio and playing it back.
Specifically optimized for ReSpeaker 2-Mic Pi HAT.
"""

import sys
import os
import pyaudio
import wave
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()


def test_microphone(device_index=None, duration=5, sample_rate=16000):
    """
    Test microphone by recording and playing back audio.
    
    Args:
        device_index: Input device index (None for default from .env)
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz (ReSpeaker supports 16kHz well)
    """
    print("\n" + "="*70)
    print("RESPEAKER 2-MIC PI HAT - MICROPHONE TEST")
    print("="*70)
    
    # Get device index from environment if not provided
    if device_index is None:
        device_index = int(os.getenv('AUDIO_INPUT_DEVICE_INDEX', 1))
    
    pa = pyaudio.PyAudio()
    
    # Show device info
    try:
        device_info = pa.get_device_info_by_index(device_index)
        print(f"\nTesting input device:")
        print(f"  Index: {device_index}")
        print(f"  Name: {device_info['name']}")
        print(f"  Max Input Channels: {device_info['maxInputChannels']}")
        print(f"  Default Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
        
        name_lower = device_info['name'].lower()
        if 'seeed' in name_lower or 'respeaker' in name_lower:
            print(f"  ⭐ ReSpeaker device detected!")
        
        print()
    except Exception as e:
        print(f"\n❌ Error accessing device {device_index}: {e}")
        print("\nAvailable input devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
        pa.terminate()
        return
    
    chunk_size = 1024
    channels = 1  # Mono recording
    format = pyaudio.paInt16
    
    frames = []
    
    try:
        # Open input stream
        print(f"🎤 Recording for {duration} seconds...")
        print("   Speak into the microphone now!\n")
        
        stream = pa.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )
        
        # Record audio
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Show progress
                elapsed = int(time.time() - start_time)
                remaining = duration - elapsed
                print(f"\r  Recording... {elapsed}s / {duration}s (remaining: {remaining}s)", end='', flush=True)
            
            except IOError as e:
                print(f"\n  ⚠️  Buffer overflow (device may not support {sample_rate} Hz)")
                break
        
        print(f"\n\n✅ Recording complete! Captured {len(frames)} chunks")
        
        stream.stop_stream()
        stream.close()
        
        # Save to file
        output_file = "./tests/test_recording.wav"
        print(f"\n💾 Saving recording to {output_file}...")
        
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(pa.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        file_size = os.path.getsize(output_file)
        print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Playback test
        print("\n🔊 Playing back recording...")
        print("   (You should hear what you just said)\n")
        
        # Prefer the env override; otherwise find the pulse output device so
        # PipeWire routes to the speaker without fighting echo-cancel-playback.
        env_out = os.getenv('AUDIO_OUTPUT_DEVICE_INDEX')
        if env_out is not None:
            output_device_index = int(env_out)
        else:
            output_device_index = None
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0 and 'pulse' in info['name'].lower():
                    output_device_index = i
                    break
        
        output_stream = pa.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            output=True,
            output_device_index=output_device_index
        )
        
        for frame in frames:
            output_stream.write(frame)
        
        output_stream.stop_stream()
        output_stream.close()
        
        print("✅ Playback complete!")
        print("\nDid you hear your voice clearly?")
        print(f"Recording saved to: {os.path.abspath(output_file)}")
        
        # Analysis
        print("\n" + "-"*70)
        print("ANALYSIS")
        print("-"*70)
        
        # Check audio level
        import struct
        max_amplitude = 0
        for frame in frames:
            samples = struct.unpack(f'{len(frame)//2}h', frame)
            frame_max = max(abs(s) for s in samples)
            if frame_max > max_amplitude:
                max_amplitude = frame_max
        
        level_percent = (max_amplitude / 32767) * 100
        print(f"Max amplitude: {max_amplitude} / 32767 ({level_percent:.1f}%)")
        
        if level_percent < 5:
            print("⚠️  WARNING: Very low audio level. Check:")
            print("   1. Microphone is properly connected")
            print("   2. Microphone is not muted")
            print("   3. You spoke loudly enough")
        elif level_percent > 95:
            print("⚠️  WARNING: Audio may be clipping (too loud)")
            print("   Consider reducing input gain")
        else:
            print("✅ Audio level looks good!")
        
        print("\nReSpeaker 2-Mic Pi HAT Notes:")
        print("  - Supports 16kHz sample rate natively")
        print("  - Has built-in LEDs that should light up during recording")
        print("  - Check 'alsamixer' to adjust input gain if needed")
    
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pa.terminate()
    
    print("\n" + "="*70 + "\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ReSpeaker 2-Mic Pi HAT microphone')
    parser.add_argument('--device', type=int, help='Input device index')
    parser.add_argument('--duration', type=int, default=5, help='Recording duration in seconds (default: 5)')
    parser.add_argument('--rate', type=int, default=16000, help='Sample rate (default: 16000)')
    
    args = parser.parse_args()
    
    test_microphone(device_index=args.device, duration=args.duration, sample_rate=args.rate)


if __name__ == "__main__":
    main()
