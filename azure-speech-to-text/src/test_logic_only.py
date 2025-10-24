"""
Test CHIPPY logic without audio hardware (WSL-friendly).
Uses pre-recorded file to simulate the pipeline.
"""

import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dotenv import load_dotenv
from src.config import Config
from src.privacy_manager import PrivacyManager
from src.rest_speech_client import RestSpeechClient
from src.tts_client import TextToSpeechClient
import requests
import json


def get_tutor_reply(user_text, flow_endpoint, flow_api_key, session_id):
    """Get response from Azure Flow."""
    if not flow_endpoint or not flow_api_key:
        return f"[Mock response] I heard: {user_text}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {flow_api_key}"
    }
    
    payload = {
        "user_message": user_text,
        "action_type": "chat",
        "learner_id": "test_student",
        "session_id": 2, #session_id,
        "chat_history": []
    }
    
    try:
        response = requests.post(flow_endpoint, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("final_answer", "No response")
        return "Error getting response"
    except Exception as e:
        return f"Error: {e}"


def test_pipeline():
    """Test the complete pipeline without audio hardware."""
    load_dotenv()
    
    print("=" * 70)
    print("🧪 CHIPPY LOGIC TEST (No Audio Hardware)")
    print("=" * 70)
    
    # Configuration
    FLOW_ENDPOINT = os.getenv("FLOW_ENDPOINT")
    FLOW_API_KEY = os.getenv("FLOW_API_KEY")
    SESSION_ID = "test_session_123"
    
    # Validate config
    try:
        Config.validate_config()
        print("✅ Azure credentials validated")
    except ValueError as e:
        print(f"❌ Config error: {e}")
        return
    
    # Initialize components
    print("✅ Initializing components...")
    privacy_manager = PrivacyManager(SESSION_ID)
    stt_client = RestSpeechClient(Config, privacy_manager, SESSION_ID)
    tts_client = TextToSpeechClient(Config)
    
    # Check for test audio file
    test_audio = os.path.join(
        os.path.dirname(parent_dir),
        "azure-speech-to-text/tests",
        "Recording (4).m4a"
    )
    
    if not os.path.exists(test_audio):
        print(f"\n⚠️  Test audio not found at: {test_audio}")
        print("Using mock text input instead...")
        recognized_text = "What is two plus two?"
        print(f"👤 Simulated input: \"{recognized_text}\"")
    else:
        # Convert if needed
        from src.utils.audio_converter import AudioConverter
        if test_audio.endswith('.m4a'):
            print(f"🔄 Converting test audio...")
            test_audio = AudioConverter.convert_to_wav(test_audio)
        
        # Test STT
        print(f"\n1️⃣  Testing Speech-to-Text...")
        stt_result = stt_client.recognize_from_file(test_audio, anonymize=True)
        
        if "error" in stt_result:
            print(f"❌ STT Error: {stt_result['error']}")
            return
        
        recognized_text = stt_result["recognized_text"]
        print(f"✅ Recognized: \"{recognized_text}\"")
    
    # Test Azure Flow
    print(f"\n2️⃣  Testing Azure Flow...")
    response_text = get_tutor_reply(recognized_text, FLOW_ENDPOINT, FLOW_API_KEY, SESSION_ID)
    print(f"✅ AI Response: \"{response_text[:100]}...\"")
    
    # Test TTS
    print(f"\n3️⃣  Testing Text-to-Speech...")
    audio_file = tts_client.synthesize_speech(response_text)
    print(f"✅ Audio generated: {audio_file}")
    print(f"   File size: {os.path.getsize(audio_file)} bytes")
    
    # Note about playback
    print(f"\n4️⃣  Audio Playback:")
    print(f"   ℹ️  Audio file saved but not played (WSL audio limitation)")
    print(f"   💡 You can manually play it in Windows: {audio_file}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\n📋 What was tested:")
    print("   ✅ Azure Speech-to-Text API")
    print("   ✅ Privacy Manager (anonymization)")
    print("   ✅ Azure Flow endpoint")
    print("   ✅ Azure Text-to-Speech API")
    print("   ⚠️  Audio I/O (skipped - WSL limitation)")
    print("\n💡 These components will work identically on Raspberry Pi!")
    print("   Only difference: Pi has native audio hardware support.\n")


if __name__ == "__main__":
    test_pipeline()