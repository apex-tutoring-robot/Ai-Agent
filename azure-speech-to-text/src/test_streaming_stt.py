"""
Test script to compare REST API vs Streaming STT latency.
"""

import time
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from privacy_manager import PrivacyManager
from rest_speech_client import RestSpeechClient
from streaming_speech_client import StreamingSpeechClient

# Load environment
load_dotenv()

def test_latency_comparison():
    """Compare latency between REST and Streaming STT."""
    
    # Validate config
    try:
        Config.validate_config()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Initialize components
    session_id = Config.generate_session_id()
    privacy_manager = PrivacyManager(session_id)
    
    # Test audio file
    test_audio = "../tests/Recording(4).m4a"
    
    # First, convert to WAV if needed
    from utils.audio_converter import AudioConverter
    if not test_audio.endswith('.wav'):
        print("Converting audio to WAV format...")
        test_audio = AudioConverter.convert_to_wav(test_audio)
    
    print("\n" + "="*60)
    print("LATENCY COMPARISON TEST")
    print("="*60)
    
    # Test 1: REST API
    print("\n🔵 TEST 1: REST API (Current Method)")
    print("-" * 60)
    rest_client = RestSpeechClient(Config, privacy_manager, session_id)
    
    start_time = time.time()
    rest_result = rest_client.recognize_from_file(test_audio, anonymize=False)
    rest_latency = time.time() - start_time
    
    if "error" not in rest_result:
        print(f"✅ Recognized: \"{rest_result['recognized_text']}\"")
        print(f"⏱️  REST API Latency: {rest_latency:.3f} seconds")
    else:
        print(f"❌ Error: {rest_result['error']}")
    
    # Test 2: Streaming SDK
    print("\n🟢 TEST 2: Streaming SDK (New Method)")
    print("-" * 60)
    streaming_client = StreamingSpeechClient(Config, privacy_manager, session_id)
    
    start_time = time.time()
    streaming_result = streaming_client.recognize_from_file(test_audio, anonymize=False)
    streaming_latency = time.time() - start_time
    
    if "error" not in streaming_result:
        print(f"✅ Recognized: \"{streaming_result['recognized_text']}\"")
        print(f"⏱️  Streaming SDK Latency: {streaming_latency:.3f} seconds")
    else:
        print(f"❌ Error: {streaming_result['error']}")
    
    # Compare
    if "error" not in rest_result and "error" not in streaming_result:
        print("\n" + "="*60)
        print("📊 RESULTS")
        print("="*60)
        improvement = rest_latency - streaming_latency
        percentage = (improvement / rest_latency) * 100
        
        print(f"REST API:      {rest_latency:.3f}s")
        print(f"Streaming SDK: {streaming_latency:.3f}s")
        print(f"Improvement:   {improvement:.3f}s ({percentage:.1f}% faster)")
        
        if improvement > 0:
            print(f"\n✨ Streaming SDK is {improvement:.3f} seconds faster!")
        else:
            print(f"\n⚠️  REST API was faster this time (network variation)")

if __name__ == "__main__":
    test_latency_comparison()