"""
CHIPPY with Wake Word Detection - Production Version
Waits for "Porcupine" before processing speech.
Features conversation mode and interrupt detection.
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
    """CHIPPY voice assistant with wake word detection and conversation mode."""
    
    def __init__(self):
        """Initialize CHIPPY with wake word."""
        # Load environment variables
        load_dotenv()
        
        # Configuration
        self.flow_endpoint = os.getenv("FLOW_ENDPOINT")
        self.flow_api_key = os.getenv("FLOW_API_KEY")
        self.porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        self.conversation_timeout = float(os.getenv("CONVERSATION_TIMEOUT", "45.0"))
        
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
        self.device_index = None
        
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
    
    def process_speech(self, audio_file: str) -> dict:
        """
        Process recorded speech through the complete pipeline with interrupt detection.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            dict with 'success': bool, 'interrupted': bool
        """
        try:
            # Step 1: Speech-to-Text
            print("📝 Converting speech to text...")
            stt_result = self.stt_client.recognize_from_file(
                audio_file_path=audio_file,
                anonymize=True
            )
            
            if "error" in stt_result and stt_result["error"]:
                print(f"❌ STT Error: {stt_result['error']}")
                return {'success': False, 'interrupted': False}
            
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
            
            # Step 5: Play response with interrupt detection
            print("🎵 Playing response (speak to interrupt)...")
            playback_result = self.tts_client.play_speech_interruptible(
                audio_output,
                device_index=self.device_index
            )
            
            # Cleanup
            try:
                os.remove(audio_file)
                os.remove(audio_output)
            except:
                pass
            
            self.interaction_count += 1
            
            return {
                'success': True,
                'interrupted': playback_result.get('interrupted', False)
            }
            
        except Exception as e:
            print(f"❌ Error processing speech: {e}")
            return {'success': False, 'interrupted': False}
    
    def conversation_mode(self):
        """
        Enter conversation mode - stay active for 30-60 seconds without wake word.
        """
        print("\n" + "🗣️ " * 35)
        print("   CONVERSATION MODE ACTIVE")
        print(f"   I'll stay active for {self.conversation_timeout} seconds")
        print(f"   Just speak - no wake word needed!")
        print("🗣️ " * 35)
        
        conversation_start = time.time()
        last_interaction = time.time()
        
        # Start audio stream for conversation
        if not self.listener.start_stream(self.device_index):
            print("❌ Failed to start speech listener")
            return
        
        try:
            while self.running:
                # Check timeout
                time_since_last = time.time() - last_interaction
                
                if time_since_last > self.conversation_timeout:
                    print(f"\n⏱️  Conversation timeout ({self.conversation_timeout}s reached)")
                    print("💤 Returning to wake word mode...")
                    break
                
                # Show countdown
                time_left = self.conversation_timeout - time_since_last
                print(f"\r🎧 Listening... (timeout in {int(time_left)}s)  ", end="", flush=True)
                
                # Listen for speech with short timeout
                audio_file = self.listener.listen_for_speech(
                    callback=lambda msg: print(f"\n  {msg}"),
                    timeout=5.0  # 5 second timeout per attempt
                )
                
                if audio_file:
                    print(f"\n📊 Interaction #{self.interaction_count + 1}")
                    print("-" * 70)
                    
                    # Reset interaction timer
                    last_interaction = time.time()
                    
                    # Process the speech
                    result = self.process_speech(audio_file)
                    
                    if result['success']:
                        if result['interrupted']:
                            print("-" * 70)
                            print("⚠️  Interrupted - ready for next question\n")
                        else:
                            print("-" * 70)
                            print("✅ Response complete\n")
                    else:
                        print("-" * 70)
                        print("⚠️  Processing incomplete\n")
                    
                    # Small pause
                    time.sleep(0.3)
        
        finally:
            # Stop speech listener
            self.listener.stop_stream()
            print()
    
    def run(self, device_index: Optional[int] = None, test_wake_word: bool = False):
        """
        Run CHIPPY with wake word detection and conversation mode.
        
        Args:
            device_index: Audio device index (None for default)
            test_wake_word: If True, only test wake word detection
        """
        self.device_index = device_index
        
        print("\n" + "=" * 70)
        print("🎤 CHIPPY WITH WAKE WORD DETECTION")
        print("=" * 70)
        print(f"Session: {self.session_id}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Wake Word: 'Porcupine'")
        print(f"Conversation Timeout: {self.conversation_timeout}s")
        print()
        
        # Initialize wake word detector with built-in "porcupine" keyword
        print(f"✅ Using built-in wake word: 'Porcupine'")
        self.wake_word_detector = WakeWordDetector(
            access_key=self.porcupine_access_key,
            keywords=["porcupine"],  # Built-in keyword
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
        print("💡 Say 'Porcupine' to activate")
        print("💬 After activation: I'll stay active for 45 seconds")
        print("🔊 You can interrupt me while I'm speaking!")
        print("🛑 Press Ctrl+C to exit")
        print("=" * 70)
        print()
        
        self.running = True
        
        # Start wake word detector
        self.wake_word_detector.start()
        
        try:
            while self.running:
                # Wait for wake word
                print("🎧 Listening for wake word 'Porcupine'...")
                keyword_index = self.wake_word_detector.listen(
                    callback=lambda msg: print(f"  {msg}")
                )
                
                if keyword_index >= 0:
                    print("\n" + "🎉" * 35)
                    print("   WAKE WORD DETECTED - CHIPPY ACTIVATED!")
                    print("🎉" * 35)
                    
                    # Stop wake word detector
                    self.wake_word_detector.stop()
                    
                    # Enter conversation mode
                    self.conversation_mode()
                    
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
    chippy = ChippyWithWakeWord()
    chippy.run(device_index=args.device, test_wake_word=args.test_wake_word)


if __name__ == "__main__":
    main()