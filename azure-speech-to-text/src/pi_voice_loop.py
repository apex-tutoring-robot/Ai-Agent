"""
CHIPPY Raspberry Pi Voice Loop - Continuous Speech Interaction
Runs continuously on Raspberry Pi, listening and responding to student queries.
"""

import os
import sys
import time
import signal
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required modules
from src.config import Config
from src.privacy_manager import PrivacyManager
from src.rest_speech_client import RestSpeechClient
from src.tts_client import TextToSpeechClient
from src.continuous_listener import ContinuousListener

# Import the flow handler
import requests
import json


class ChippyVoiceLoop:
    """Main voice interaction loop for CHIPPY on Raspberry Pi."""
    
    def __init__(self):
        """Initialize CHIPPY voice loop."""
        # Load environment variables
        load_dotenv()
        
        # Configuration
        self.flow_endpoint = os.getenv("FLOW_ENDPOINT")
        self.flow_api_key = os.getenv("FLOW_API_KEY")
        
        # Generate or load session ID
        self.session_id = os.getenv("CHIPPY_SESSION_ID") or Config.generate_session_id()
        
        # Validate Azure configuration
        try:
            Config.validate_config()
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            print("Please set up your .env file with Azure credentials")
            sys.exit(1)
        
        # Initialize components
        print("🤖 Initializing CHIPPY...")
        print(f"🆔 Session ID: {self.session_id}")
        
        self.privacy_manager = PrivacyManager(self.session_id)
        self.stt_client = RestSpeechClient(Config, self.privacy_manager, self.session_id)
        self.tts_client = TextToSpeechClient(Config)
        
        # Initialize continuous listener with Pi-optimized settings
        self.listener = ContinuousListener(
            rate=16000,
            channels=1,
            chunk_size=1024,
            silence_threshold=float(os.getenv("VAD_SILENCE_THRESHOLD", "0.015")),
            silence_duration=float(os.getenv("VAD_SILENCE_DURATION", "2.0")),
            min_speech_duration=float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.5")),
            pre_speech_buffer=float(os.getenv("VAD_PRE_BUFFER", "0.3"))
        )
        
        # State
        self.running = False
        self.interaction_count = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle interrupt signals gracefully."""
        print("\n\n⏹️  Shutting down CHIPPY gracefully...")
        self.running = False
    
    def get_tutor_reply(self, user_text: str) -> str:
        """
        Get tutoring response from Azure Flow endpoint.
        
        Args:
            user_text: The recognized user text
            
        Returns:
            Tutor response text
        """
        if not self.flow_endpoint or not self.flow_api_key:
            print("⚠️  Flow endpoint not configured, using echo response")
            return f"I heard you say: {user_text}. However, my AI brain is not connected yet."
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.flow_api_key}"
        }
        
        payload = {
            "user_message": user_text,
            "action_type": "chat",
            "learner_id": "pi_student",
            "session_id": self.session_id,
            "chat_history": []
        }
        
        try:
            response = requests.post(
                self.flow_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("final_answer", "I'm not sure how to respond to that.")
                return response_text
            else:
                print(f"⚠️  Flow API error: {response.status_code}")
                return "I'm having trouble thinking right now. Could you try again?"
                
        except requests.Timeout:
            print("⚠️  Flow API timeout")
            return "Sorry, I'm thinking too slowly. Let's try again."
        except Exception as e:
            print(f"⚠️  Flow API error: {e}")
            return "I'm having technical difficulties. Let's continue anyway!"
    
    def process_speech(self, audio_file: str) -> bool:
        """
        Process recorded speech through the complete pipeline.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Step 1: Speech-to-Text
            print("\n📝 Converting speech to text...")
            stt_result = self.stt_client.recognize_from_file(
                audio_file_path=audio_file,
                anonymize=True
            )
            
            if "error" in stt_result and stt_result["error"]:
                print(f"❌ STT Error: {stt_result['error']}")
                return False
            
            recognized_text = stt_result["recognized_text"]
            print(f"👤 You said: \"{recognized_text}\"")
            
            # Step 2: Get AI response
            print("🧠 Thinking...")
            response_text = self.get_tutor_reply(recognized_text)
            
            # Step 3: Restore privacy if needed
            if stt_result.get("anonymized", False):
                response_text = self.privacy_manager.restore_personal_response(response_text)
            
            print(f"🤖 CHIPPY: \"{response_text[:100]}{'...' if len(response_text) > 100 else ''}\"")
            
            # Step 4: Text-to-Speech
            print("🔊 Converting to speech...")
            audio_output = self.tts_client.synthesize_speech(response_text)
            
            # Step 5: Play response
            print("🎵 Playing response...")
            self.tts_client.play_speech(audio_output)
            
            # Cleanup temporary files
            try:
                os.remove(audio_file)
                os.remove(audio_output)
            except:
                pass
            
            self.interaction_count += 1
            return True
            
        except Exception as e:
            print(f"❌ Error processing speech: {e}")
            return False
    
    def run(self, device_index: Optional[int] = None, test_mode: bool = False):
        """
        Run the continuous voice loop.
        
        Args:
            device_index: Audio device index (None for default)
            test_mode: If True, run test mode with microphone check
        """
        print("\n" + "=" * 70)
        print("🎤 CHIPPY CONTINUOUS VOICE INTERACTION - RASPBERRY PI MODE")
        print("=" * 70)
        print(f"Session: {self.session_id}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Show flow configuration
        if self.flow_endpoint:
            print(f"✅ AI Flow: Connected")
        else:
            print(f"⚠️  AI Flow: Not configured (will echo responses)")
        
        print()
        
        # List available devices
        self.listener.list_audio_devices()
        
        # Start audio stream
        if not self.listener.start_stream(device_index):
            print("❌ Failed to start audio stream")
            return
        
        # Test mode
        if test_mode:
            print("\n🔧 Running in TEST MODE")
            print("This will test your microphone without processing speech.")
            print()
            self.listener.test_microphone(duration=5)
            self.listener.cleanup()
            return
        
        # Main loop
        print("\n✅ CHIPPY is ready!")
        print("💡 Speak naturally - I'll respond when you're done talking")
        print("🛑 Press Ctrl+C to exit")
        print("\n" + "=" * 70 + "\n")
        
        self.running = True
        
        while self.running:
            try:
                # Listen for speech
                audio_file = self.listener.listen_for_speech(
                    callback=lambda msg: print(f"  {msg}"),
                    timeout=None  # Wait indefinitely
                )
                
                if audio_file:
                    print(f"\n📊 Interaction #{self.interaction_count + 1}")
                    print("-" * 70)
                    
                    # Process the speech
                    success = self.process_speech(audio_file)
                    
                    if success:
                        print("-" * 70)
                        print("✅ Response complete\n")
                    else:
                        print("-" * 70)
                        print("⚠️  Processing incomplete\n")
                    
                    # Small pause before listening again
                    time.sleep(0.5)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("Continuing...")
                time.sleep(1)
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        self.listener.cleanup()
        print(f"\n👋 CHIPPY shutting down. Total interactions: {self.interaction_count}")
        print("=" * 70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CHIPPY Raspberry Pi Voice Loop")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device index (use --list-devices to see available devices)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (microphone test only)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    args = parser.parse_args()
    
    # List devices only
    if args.list_devices:
        import pyaudio
        pa = pyaudio.PyAudio()
        print("\n🎤 Available Audio Devices:")
        print("-" * 60)
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"Device {i}: {info['name']}")
                print(f"  Channels: {info['maxInputChannels']}")
                print(f"  Sample Rate: {int(info['defaultSampleRate'])} Hz")
                print()
        pa.terminate()
        return
    
    # Run CHIPPY
    chippy = ChippyVoiceLoop()
    chippy.run(device_index=args.device, test_mode=args.test)


if __name__ == "__main__":
    main()