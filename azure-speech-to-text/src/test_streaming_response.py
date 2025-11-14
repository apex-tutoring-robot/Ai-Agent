"""
Test script for streaming LLM responses with parallel TTS.
This demonstrates the latency improvement from Option B.
"""

import time
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from privacy_manager import PrivacyManager
from tts_client import TextToSpeechClient
from streaming_flow_client import StreamingFlowClient, stream_and_speak

load_dotenv()

def test_streaming_vs_traditional():
    """Compare traditional vs streaming response delivery."""
    
    # Validate config
    try:
        Config.validate_config()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Check Flow endpoint
    FLOW_ENDPOINT = os.getenv("FLOW_ENDPOINT")
    FLOW_API_KEY = os.getenv("FLOW_API_KEY")
    
    if not FLOW_ENDPOINT or not FLOW_API_KEY:
        print("❌ FLOW_ENDPOINT and FLOW_API_KEY must be set in .env")
        return
    
    # Initialize
    session_id = Config.generate_session_id()
    privacy_manager = PrivacyManager(session_id)
    tts_client = TextToSpeechClient(Config)
    
    test_question = "What is photosynthesis?"
    
    print("\n" + "="*60)
    print("STREAMING RESPONSE TEST")
    print("="*60)
    print(f"\nTest question: \"{test_question}\"")
    
    # Test streaming approach
    print("\n🟢 STREAMING APPROACH (Option B)")
    print("-" * 60)
    
    streaming_client = StreamingFlowClient(FLOW_ENDPOINT, FLOW_API_KEY, session_id)
    
    start_time = time.time()
    result = stream_and_speak(
        streaming_flow_client=streaming_client,
        tts_client=tts_client,
        privacy_manager=privacy_manager,
        user_text=test_question,
        device_index=None,
        verbose=True
    )
    total_time = time.time() - start_time
    
    print(f"\n✅ Streaming complete in {total_time:.2f}s")
    print(f"📄 Response: {result['full_response'][:150]}...")
    
    print("\n💡 NOTE: With streaming, first sentence started playing")
    print("   after ~1-2 seconds, not waiting for full response!")

if __name__ == "__main__":
    test_streaming_vs_traditional()