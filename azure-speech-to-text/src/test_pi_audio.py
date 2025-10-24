"""
Quick test script for Raspberry Pi audio setup.
Tests microphone input and speaker output.
"""

import pyaudio
import wave
import tempfile
import os
import subprocess


def test_microphone(duration=3):
    """Test microphone recording."""
    print("🎤 Testing Microphone...")
    print(f"Recording for {duration} seconds...")
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save to file
    fd, temp_file = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    
    wf = wave.open(temp_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"✅ Recording saved to: {temp_file}")
    return temp_file


def test_speaker(audio_file):
    """Test speaker playback."""
    print("\n🔊 Testing Speaker...")
    print("Playing recorded audio...")
    
    try:
        result = subprocess.run(["aplay", audio_file], stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("✅ Speaker test successful!")
            return True
        else:
            print(f"❌ Speaker test failed: {result.stderr.decode()}")
            return False
    except FileNotFoundError:
        print("❌ 'aplay' not found. Install with: sudo apt-get install alsa-utils")
        return False


def main():
    print("=" * 60)
    print("🤖 CHIPPY Pi Audio Test")
    print("=" * 60)
    
    # Test microphone
    audio_file = test_microphone(duration=3)
    
    # Test speaker
    test_speaker(audio_file)
    
    # Cleanup
    os.remove(audio_file)
    
    print("\n=" * 60)
    print("✅ Audio test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()