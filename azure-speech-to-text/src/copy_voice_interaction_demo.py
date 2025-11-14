"""
CHIPPY Voice Interaction Demo - Complete STT to TTS Loop with Azure Flow
"""

import os
import sys
import time
import traceback
import requests
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required modules
from src.config import Config
from src.privacy_manager import PrivacyManager
from src.utils.audio_converter import AudioConverter

# Import the REST client for STT and TTS client
try:
    # from src.rest_speech_client import RestSpeechClient
    from streaming_speech_client import StreamingSpeechClient

except ImportError:
    # Fall back to dash-named file if that's what exists
    from importlib.machinery import SourceFileLoader
    RestSpeechClient = SourceFileLoader(
        "RestSpeechClient", 
        os.path.join(current_dir, "rest-speech-client.py")
    ).load_module().RestSpeechClient

from src.tts_client import TextToSpeechClient

def get_tutor_reply(user_text: str, flow_endpoint: str, flow_api_key: str, session_id: str) -> str:
    """Call Azure Flow endpoint with recognized text using the correct format."""
    if not flow_endpoint or not flow_api_key:
        print("⚠️ Flow endpoint or API key not configured. Using fallback response.")
        return f"Echo response: {user_text}. (Flow endpoint not configured)"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {flow_api_key}"
    }
    
    # ✅ CORRECT: Single-level data structure
    payload_correct = {
        "user_message": user_text,
        "action_type": "chat", 
        "learner_id": "student_1",
        "session_id": session_id,
        "chat_history": []
    }

    try:
        print(f"📡 Sending correct payload format...")
        print(f"🔍 Payload: {json.dumps(payload_correct, indent=2)}")
        
        resp = requests.post(flow_endpoint, headers=headers, json=payload_correct)
        
        print(f"📊 Response status: {resp.status_code}")
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"📄 Response preview: {json.dumps(data, indent=2)[:200]}...")
                
                # Extract the response
                response_text = data.get("final_answer")
                
                if response_text:
                    print(f"✅ Success! Session ID should be preserved: {session_id}")
                    print(f"🤖 Extracted response: {response_text[:100]}...")
                    return response_text
                else:
                    print(f"⚠️ Got response but couldn't extract text")
                    print(f"📄 Available fields: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    return "Could not extract response"
                    
            except json.JSONDecodeError:
                print(f"⚠️ Response is not valid JSON")
                print(f"📄 Raw response: {resp.text[:200]}...")
                return resp.text
                
        else:
            resp.raise_for_status()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return "I'm having trouble connecting to my brain right now."

def complete_voice_interaction_demo():
    """Test the complete voice interaction loop: STT → Azure Flow → TTS."""
    # Load environment variables
    load_dotenv()
    
    # Get Flow endpoint configuration
    FLOW_ENDPOINT = os.getenv("FLOW_ENDPOINT")
    FLOW_API_KEY = os.getenv("FLOW_API_KEY")
    
    # Get or create consistent session ID
    # Option 1: Use environment variable if set
    CHIPPY_SESSION_ID = os.getenv("CHIPPY_SESSION_ID")
    
    # Option 2: Use hardcoded session ID (what you had)
    if not CHIPPY_SESSION_ID:
        CHIPPY_SESSION_ID = "session_1758939181"
    
    # Option 3: Generate one and reuse it (uncomment if you prefer this)
    # if not CHIPPY_SESSION_ID:
    #     CHIPPY_SESSION_ID = Config.generate_session_id()
    
    print(f"🆔 Using session ID: {CHIPPY_SESSION_ID}")
    
    # Validate configuration
    try:
        Config.validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please make sure you've set up your .env file with Azure Speech credentials.")
        return
    
    print("=" * 60)
    print("CHIPPY Complete Voice Interaction Demo with Azure Flow")
    print("=" * 60)
    print("Testing speech-to-text → Azure Flow → text-to-speech")
    print("=" * 60)
    
    # Show Flow configuration status
    if FLOW_ENDPOINT and FLOW_API_KEY:
        print(f"✅ Flow endpoint configured: {FLOW_ENDPOINT[:50]}...")
    else:
        print("⚠️ Flow endpoint not configured - will use fallback responses")
        print("   Set FLOW_ENDPOINT and FLOW_API_KEY in your .env file")
    
    # Initialize the privacy manager with consistent session ID
    privacy_manager = PrivacyManager(CHIPPY_SESSION_ID)
    
    # Path to the recording in the tests directory
    recording_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", 
        "Recording (4).m4a"
    )
    
    print(f"\nLooking for test recording at: {recording_path}")
    
    # Check if file exists
    if not os.path.exists(recording_path):
        print(f"Error: Recording file not found at {recording_path}")
        return
    
    print("File found! Processing...")
    
    # Initialize speech-to-text client with consistent session ID
    print("\nInitializing Speech-to-Text client...")
    try:
        # stt_client = RestSpeechClient(Config, privacy_manager, CHIPPY_SESSION_ID)
        stt_client = StreamingSpeechClient(Config, privacy_manager, CHIPPY_SESSION_ID)
        print("✓ Speech-to-Text client initialized successfully")
    except Exception as e:
        print(f"Error initializing Speech-to-Text client: {e}")
        return
    
    # Initialize text-to-speech client
    print("\nInitializing Text-to-Speech client...")
    try:
        tts_client = TextToSpeechClient(Config)
        print("✓ Text-to-Speech client initialized successfully")
    except Exception as e:
        print(f"Error initializing Text-to-Speech client: {e}")
        return
    
    # Convert .m4a to WAV if needed
    if recording_path.endswith('.m4a'):
        try:
            print("\nConverting .m4a file to WAV format...")
            recording_path = AudioConverter.convert_to_wav(recording_path)
        except Exception as e:
            print(f"Error converting audio file: {e}")
            return
    
    # Process the input audio
    try:
        print(f"\nProcessing audio input: {recording_path}")
        
        # Step 1: Speech-to-Text
        print("\n1. SPEECH-TO-TEXT PHASE")
        print("-----------------------")
        stt_result = stt_client.recognize_from_file(
            audio_file_path=recording_path,
            anonymize=True,  # Apply privacy protection
            callback=lambda text: print(f"Recognition status: {text}")
        )
        
        if "error" in stt_result and stt_result["error"]:
            print(f"\nError in speech recognition: {stt_result['error']}")
            return
        
        recognized_text = stt_result["recognized_text"]
        print(f"\nRecognized text: \"{recognized_text}\"")
        
        # Step 2: Get AI response (NEW - WITH STREAMING!)
        print("\n2. AZURE FLOW PROCESSING PHASE (STREAMING)")
        print("------------------------------")
        
        # Import streaming components
        from streaming_flow_client import StreamingFlowClient, stream_and_speak
        
        # Create streaming client
        streaming_flow = StreamingFlowClient(FLOW_ENDPOINT, FLOW_API_KEY, CHIPPY_SESSION_ID)
        
        # Stream and speak!
        print("🧠 Getting streaming response and speaking in real-time...")
        result = stream_and_speak(
            streaming_flow_client=streaming_flow,
            tts_client=tts_client,
            privacy_manager=privacy_manager,
            user_text=recognized_text,
            device_index=None,  # Use default
            verbose=True
        )
        
        print(f"\n✅ Streaming complete!")
        print(f"📄 Full response: \"{result['full_response'][:200]}...\"")
        
        if result['interrupted']:
            print("⚠️  Playback was interrupted")
        
        # Step 3: Restore privacy before TTS (in a real system, this would happen after LLM)
        if stt_result.get("anonymized", False):
            print("\nRestoring personal context in response...")
            response_text = privacy_manager.restore_personal_response(response_text)
            print(f"Restored response: \"{response_text[:100]}...\"")
        
        # Step 4: Text-to-Speech
        print("\n3. TEXT-TO-SPEECH PHASE")
        print("----------------------")
        print("Converting response to speech...")
        
        output_file = tts_client.synthesize_speech(response_text)
        print(f"Speech synthesized and saved to: {output_file}")
        
        # Step 5: Play the speech
        print("\n4. AUDIO PLAYBACK PHASE")
        print("---------------------")
        print("Playing response through speakers...")
        print("(If you don't hear anything, check your audio setup or install required audio libraries)")
        
        tts_client.play_speech(output_file)
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE VOICE INTERACTION LOOP SUCCESS!")
        print("=" * 60)
        print("✓ Speech-to-Text: Audio → Text")
        print("✓ Azure Flow: Text → AI Response")
        print("✓ Text-to-Speech: Response → Audio")
        print("✓ Audio Playback: Played through speakers")
        print(f"✓ Session ID maintained: {CHIPPY_SESSION_ID}")
        print("\nIn a production CHIPPY deployment, this would be synchronized")
        print("with face animations on the Raspberry Pi display.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    complete_voice_interaction_demo()