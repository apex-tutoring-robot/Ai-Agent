"""
Full Audio Pipeline Test
Tests the complete audio pipeline: microphone input → processing → speaker output
Simulates the actual Jarvis data flow without requiring Azure services.
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio.single_turn_vad import VADAudioCapture
from audio.playback import AudioPlayer
from dotenv import load_dotenv

load_dotenv()


def test_audio_pipeline():
    """Test complete audio pipeline with VAD capture and playback."""
    
    print("\n" + "="*70)
    print("FULL AUDIO PIPELINE TEST")
    print("="*70)
    print("\nThis test simulates Jarvis's audio flow:")
    print("  1. VAD-based audio capture from ReSpeaker")
    print("  2. Audio processing (echo back)")
    print("  3. Playback through USB speaker")
    print("\n" + "="*70)
    
    # Test 1: VAD Capture
    print("\n[TEST 1/3] Testing Voice Activity Detection...")
    print("-"*70)
    
    try:
        vad_capture = VADAudioCapture()
        
        print("\n🎤 Speak when ready, then pause for silence detection...")
        print(f"   (VAD will auto-stop after {vad_capture.silence_timeout_ms}ms of silence)\n")
        
        input("Press Enter when ready to start recording...")
        
        audio_data = vad_capture.capture_speech()
        
        if audio_data:
            print(f"✅ Captured {len(audio_data):,} bytes of audio")
            duration = len(audio_data) / (vad_capture.sample_rate * 2)  # 2 bytes per sample
            print(f"   Duration: ~{duration:.1f} seconds")
        else:
            print("❌ No audio captured!")
            return
    
    except Exception as e:
        print(f"❌ Error in VAD capture: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Audio Playback
    print("\n[TEST 2/3] Testing Audio Playback...")
    print("-"*70)
    
    try:
        player = AudioPlayer()
        
        print("\n🔊 Playing back captured audio...")
        print("   (You should hear what you just said)\n")
        
        time.sleep(0.5)  # Brief pause
        
        player.play_audio(audio_data)
        player.cleanup()
        
        print("✅ Playback complete!")
    
    except Exception as e:
        print(f"❌ Error in playback: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Streaming Playback
    print("\n[TEST 3/3] Testing Streaming Playback...")
    print("-"*70)
    
    try:
        player = AudioPlayer()
        
        print("\n🔊 Testing streaming playback (chunked)...")
        print("   Playing the same audio in 4 chunks...\n")
        
        player.start_streaming()
        
        # Split audio into chunks
        chunk_size = len(audio_data) // 4
        for i in range(4):
            start = i * chunk_size
            end = start + chunk_size if i < 3 else len(audio_data)
            chunk = audio_data[start:end]
            player.queue_audio(chunk)
            print(f"  Queued chunk {i+1}/4 ({len(chunk):,} bytes)")
            time.sleep(0.1)
        
        # Wait for playback to complete
        time.sleep(2)
        player.stop_streaming()
        
        print("\n✅ Streaming playback complete!")
    
    except Exception as e:
        print(f"❌ Error in streaming playback: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("\n✅ Audio pipeline test complete!")
    print("\nWhat was tested:")
    print("  ✓ VAD-based speech capture from ReSpeaker")
    print("  ✓ Audio playback through USB speaker")
    print("  ✓ Streaming audio architecture")
    print("\nIf all tests passed, your hardware is ready for Jarvis!")
    print("\nNext steps:")
    print("  1. Configure Azure credentials in .env")
    print("  2. Set up Picovoice wake word")
    print("  3. Run full Jarvis: python src/main.py")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main function."""
    try:
        test_audio_pipeline()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
