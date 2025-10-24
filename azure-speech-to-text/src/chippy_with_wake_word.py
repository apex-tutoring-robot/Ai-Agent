"""
CHIPPY with Wake Word Detection - Production Version
Waits for "Hello CHIPPY" before processing speech.
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
from src.wake_word_detector import WakeWordDetector

import requests
import json


class ChippyWithWakeWord:
    """CHIPPY voice assistant with wake word detection."""
    
    def __init__(self):
        """Initialize CHIPPY with wake word."""
        # Load environment variables
        load_dotenv()
        
        # Configuration
        self.flow_endpoint = os.getenv("FLOW_ENDPOINT")
        self.flow_api_key = os.getenv("FLOW_API_KEY")
        self.porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        
        # Validate Porcupine access key
        if not self.porcupine_access_key:
            print("❌ PORCUPINE_ACCESS_KEY not found in .env file!")
            print("Get your key from: https://console.picovoice.ai/")
            sys.exit(1)
        
        # Session ID
        self.session_id = os.getenv("CHIPPY_SESSION_ID") or Config.generate_session_id()
        
        # Validate Azure configuration
        try:
            Config.validate_config()
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            sys.exit(1)
        
        # Initialize components
        print("🤖 Initializing CHIPPY with Wake Word Detection...")
        print(f"🆔 Session ID: {self.session_id}")
        
        self.privacy_manager = PrivacyManager(self.session_id)
        self.stt_client = RestSpeechClient(Config, self.privacy_manager, self.session_id)
        self.tts_client = TextToSpeechClient(Config)
        
        # Initialize VAD listener
        self.listener = ContinuousListener(
            rate=16000,
            channels=1,
            chunk_size=1024,
            silence_threshold=float(os.getenv("VAD_SILENCE_THRESHOLD", "0.015")),
            silence_duration=float(os.getenv("VAD_SILENCE_DURATION", "2.0")),
            min_speech_duration=float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.5")),
            pre_speech_buffer=float(os.getenv("VAD_PRE_BUFFER", "0.3"))
        )
        
        # Initialize wake word detector (will be created in run())
        self.wake_word_detector = None
        
        # State
        self.running = False
        self.interaction_count = 0
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle interrupt signals gracefully."""
        print("\n\n⏹️  Shutting down CHIPPY gracefully...")
        self.running = False
    
    def get_tutor_reply(self, user_text: str) -> str:
        """Get tutoring response from Azure Flow endpoint."""
        if not self.flow_endpoint or not self.flow_api_key:
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
                return data.get("final_answer", "I'm not sure how to respond to that.")
            else:
                return "I'm having trouble thinking right now. Could you try again?"
                
        except requests.Timeout:
            return "Sorry, I'm thinking too slowly. Let's try again."
        except Exception as e:
            print(f"⚠️  Flow API error: {e}")
            return "I'm having technical difficulties. Let's continue anyway!"
    
    def process_speech(self, audio_file: str) -> bool:
        """Process recorded speech through the complete pipeline."""
        try:
            # Step 1: Speech-to-Text
            print("📝 Converting speech to text...")
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
            
            # Step 3: Restore privacy
            if stt_result.get("anonymized", False):
                response_text = self.privacy_manager.restore_personal_response(response_text)
            
            print(f"🤖 CHIPPY: \"{response_text[:100]}{'...' if len(response_text) > 100 else ''}\"")
            
            # Step 4: Text-to-Speech
            print("🔊 Converting to speech...")
            audio_output = self.tts_client.synthesize_speech(response_text)
            
            # Step 5: Play response
            print("🎵 Playing response...")
            self.tts_client.play_speech(audio_output)
            
            # Cleanup
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
    
    def run(self, device_index: Optional[int] = None, test_wake_word: bool = False):
        """
        Run CHIPPY with wake word detection.
        
        Args:
            device_index: Audio device index (None for default)
            test_wake_word: If True, only test wake word detection
        """
        print("\n" + "=" * 70)
        print("🎤 CHIPPY WITH WAKE WORD DETECTION")
        print("=" * 70)
        print(f"Session: {self.session_id}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check for custom wake word file
        wake_word_path = os.path.join(
            parent_dir,
            "azure-speech-to-text/wake_words",
            "Hey-Chippy_en_raspberry-pi_v3_0_0.ppn"
        )
        
        # Initialize wake word detector
        if os.path.exists(wake_word_path):
            print(f"✅ Using custom wake word: {wake_word_path}")
            self.wake_word_detector = WakeWordDetector(
                access_key=self.porcupine_access_key,
                keyword_paths=[wake_word_path],
                sensitivities=[0.5],
                device_index=device_index
            )
        else:
            print(f"⚠️  Custom wake word not found at: {wake_word_path}")
            print(f"💡 Using built-in keyword 'porcupine' for testing")
            print(f"   Train your custom 'Hello CHIPPY' at: https://console.picovoice.ai/")
            self.wake_word_detector = WakeWordDetector(
                access_key=self.porcupine_access_key,
                keywords=["porcupine"],  # Built-in test keyword
                sensitivities=[0.5],
                device_index=device_index
            )
        
        # Test mode
        if test_wake_word:
            print("\n🔧 WAKE WORD TEST MODE")
            print("=" * 70)
            self.wake_word_detector.test(duration=15)
            self.wake_word_detector.cleanup()
            return
        
        # Show configuration
        if self.flow_endpoint:
            print(f"✅ AI Flow: Connected")
        else:
            print(f"⚠️  AI Flow: Not configured")
        
        print()
        print("=" * 70)
        print("✅ CHIPPY is ready and waiting for wake word!")
        print("💡 Say 'Hello CHIPPY' (or 'Porcupine' if using test keyword)")
        print("🛑 Press Ctrl+C to exit")
        print("=" * 70)
        print()
        
        self.running = True
        
        # Start wake word detector
        self.wake_word_detector.start()
        
        try:
            while self.running:
                # Wait for wake word
                print("🎧 Listening for wake word...")
                keyword_index = self.wake_word_detector.listen(
                    callback=lambda msg: print(f"  {msg}")
                )
                
                if keyword_index >= 0:
                    print("\n" + "🎉" * 35)
                    print("   WAKE WORD DETECTED - CHIPPY ACTIVATED!")
                    print("🎉" * 35)
                    print(f"\n📊 Interaction #{self.interaction_count + 1}")
                    print("-" * 70)
                    
                    # Stop wake word detector temporarily
                    self.wake_word_detector.stop()
                    
                    # Start VAD listener for speech
                    if not self.listener.start_stream(device_index):
                        print("❌ Failed to start speech listener")
                        self.wake_word_detector.start()
                        continue
                    
                    # Listen for actual speech
                    print("🎤 Listening for your question...")
                    audio_file = self.listener.listen_for_speech(
                        callback=lambda msg: print(f"  {msg}"),
                        timeout=30  # 30 second timeout
                    )
                    
                    # Stop speech listener
                    self.listener.stop_stream()
                    
                    if audio_file:
                        # Process the speech
                        success = self.process_speech(audio_file)
                        
                        if success:
                            print("-" * 70)
                            print("✅ Response complete")
                        else:
                            print("-" * 70)
                            print("⚠️  Processing incomplete")
                    else:
                        print("⚠️  No speech detected, timeout reached")
                    
                    print()
                    
                    # Restart wake word detector
                    time.sleep(0.5)
                    self.wake_word_detector.start()
                    
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            if self.wake_word_detector:
                self.wake_word_detector.cleanup()
            self.listener.cleanup()
            print(f"\n👋 CHIPPY shutting down. Total interactions: {self.interaction_count}")
            print("=" * 70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CHIPPY with Wake Word Detection")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device index"
    )
    parser.add_argument(
        "--test-wake-word",
        action="store_true",
        help="Test wake word detection only"
    )
    
    args = parser.parse_args()
    
    # Run CHIPPY
    chippy = ChippyWithWakeWord()
    chippy.run(device_index=args.device, test_wake_word=args.test_wake_word)


if __name__ == "__main__":
    main()