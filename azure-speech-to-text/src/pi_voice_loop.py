"""
CHIPPY Raspberry Pi Voice Loop - Continuous Speech Interaction with Conversation Mode
Runs continuously on Raspberry Pi, listening and responding to student queries.
Features interrupt detection, conversation mode, and Azure OpenAI integration.
"""

import os
import sys
import time
import signal
import threading
import queue
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
from streaming_speech_client import StreamingSpeechClient
from src.tts_client import TextToSpeechClient
from src.continuous_listener import ContinuousListener

# Import Azure OpenAI client and helpers
from azure_openai_client import AzureOpenAIClient
from streaming_flow_client import SentenceStreamer  # Reuse the sentence streamer


class ChippyVoiceLoop:
    """Main voice interaction loop for CHIPPY on Raspberry Pi with conversation mode."""
    
    def __init__(self):
        """Initialize CHIPPY voice loop."""
        # Load environment variables
        load_dotenv()
        
        # Azure OpenAI Configuration (NEW - replaces Prompt Flow)
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "baseModelGPT-4.1")
        self.student_grade = int(os.getenv("STUDENT_GRADE", "5"))
        self.current_topic = os.getenv("CURRENT_TOPIC", "general")
        
        # Other configuration
        self.conversation_timeout = float(os.getenv("CONVERSATION_TIMEOUT", "45.0"))
        
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
        self.stt_client = StreamingSpeechClient(Config, self.privacy_manager, self.session_id)
        self.tts_client = TextToSpeechClient(Config)
        
        # Initialize Azure OpenAI client
        if self.azure_openai_endpoint and self.azure_openai_key:
            try:
                self.azure_openai_client = AzureOpenAIClient(
                    endpoint=self.azure_openai_endpoint,
                    api_key=self.azure_openai_key,
                    deployment=self.azure_openai_deployment,
                    grade=self.student_grade,
                    topic=self.current_topic
                )
            except Exception as e:
                print(f"⚠️  Failed to initialize Azure OpenAI client: {e}")
                self.azure_openai_client = None
        else:
            self.azure_openai_client = None
            print("⚠️  Azure OpenAI not configured")
        
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
        self.device_index = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle interrupt signals gracefully."""
        print("\n\n⏹️  Shutting down CHIPPY gracefully...")
        self.running = False
        
        # Close Azure OpenAI client
        if self.azure_openai_client:
            try:
                self.azure_openai_client.close()
            except:
                pass
    
    def process_speech(self, audio_file: str) -> bool:
        """
        Process recorded speech through the complete pipeline with streaming responses.
        NOW WITH AZURE OPENAI + PARALLEL TTS - speaks while thinking!
        
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
            
            # Step 2 & 3 & 4: Stream AI response + TTS + Play (ALL IN PARALLEL!)
            print("🧠 Thinking and speaking...")
            
            if self.azure_openai_client:
                # Use Azure OpenAI streaming (NEW!)
                result = stream_and_speak_openai(
                    azure_openai_client=self.azure_openai_client,
                    tts_client=self.tts_client,
                    privacy_manager=self.privacy_manager,
                    user_text=recognized_text,
                    device_index=self.device_index,
                    verbose=True
                )
                
                if result['interrupted']:
                    print("⚠️  Response was interrupted by user")
                    return True  # Still successful, just interrupted
                
                print(f"\n✅ Complete response delivered!")
            else:
                # Fallback if Azure OpenAI not configured
                print("⚠️  Azure OpenAI not available, using echo response")
                response_text = f"I heard you say: {recognized_text}. However, my AI brain is not connected yet."
                
                if stt_result.get("anonymized", False):
                    response_text = self.privacy_manager.restore_personal_response(response_text)
                
                print(f"🤖 CHIPPY: \"{response_text}\"")
                
                audio_output = self.tts_client.synthesize_speech(response_text)
                playback_result = self.tts_client.play_speech_interruptible(
                    audio_output,
                    device_index=self.device_index
                )
            
            # Cleanup audio file
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except Exception as e:
                print(f"⚠️  Could not remove audio file: {e}")
            
            self.interaction_count += 1
            return True
            
        except Exception as e:
            print(f"❌ Error processing speech: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, device_index: Optional[int] = None, test_mode: bool = False):
        """
        Run the continuous voice loop with conversation mode.
        
        Args:
            device_index: Audio device index (None for default)
            test_mode: If True, run test mode with microphone check
        """
        self.device_index = device_index
        
        print("\n" + "=" * 70)
        print("🎤 CHIPPY CONTINUOUS VOICE INTERACTION - RASPBERRY PI MODE")
        print("=" * 70)
        print(f"Session: {self.session_id}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Conversation Timeout: {self.conversation_timeout}s")
        print()
        
        # Show Azure OpenAI configuration
        if self.azure_openai_client:
            print(f"✅ Azure OpenAI: Connected ({self.azure_openai_deployment})")
            print(f"   Student: Grade {self.student_grade}, Topic: {self.current_topic}")
        else:
            print(f"⚠️  Azure OpenAI: Not configured (will use fallback responses)")
        
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
        print(f"💬 Conversation mode: I'll stay active for {self.conversation_timeout}s")
        print("🔊 You can interrupt me while I'm speaking!")
        print("🛑 Press Ctrl+C to exit")
        print("\n" + "=" * 70 + "\n")
        
        self.running = True
        conversation_active = True
        last_interaction_time = time.time()
        
        try:
            while self.running:
                try:
                    # Check if conversation timeout reached
                    time_since_last = time.time() - last_interaction_time
                    
                    if conversation_active and time_since_last > self.conversation_timeout:
                        print(f"\n⏱️  Conversation timeout ({self.conversation_timeout}s)")
                        print("💤 Going to sleep... Say something to wake me!")
                        conversation_active = False
                    
                    # Show timeout countdown
                    if conversation_active:
                        time_left = self.conversation_timeout - time_since_last
                        print(f"\r🎧 Listening... (timeout in {int(time_left)}s)  ", end="", flush=True)
                    
                    # Listen for speech with shorter timeout during conversation
                    listen_timeout = 5.0 if conversation_active else None
                    audio_file = self.listener.listen_for_speech(
                        callback=lambda msg: print(f"\n  {msg}"),
                        timeout=listen_timeout
                    )
                    
                    if audio_file:
                        print(f"\n📊 Interaction #{self.interaction_count + 1}")
                        print("-" * 70)
                        
                        # Reset conversation timer
                        last_interaction_time = time.time()
                        conversation_active = True
                        
                        # Process the speech
                        success = self.process_speech(audio_file)
                        
                        if success:
                            print("-" * 70)
                            print("✅ Response complete\n")
                        else:
                            print("-" * 70)
                            print("⚠️  Processing failed\n")
                        
                        # Small pause before listening again
                        time.sleep(0.3)
                    else:
                        # Timeout during listening
                        if conversation_active:
                            # Continue conversation
                            continue
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Unexpected error in main loop: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing...")
                    time.sleep(1)
        
        finally:
            # Cleanup (always runs, even on exception)
            print("\n🧹 Cleaning up...")
            self.listener.cleanup()
            
            # Close Azure OpenAI client
            if self.azure_openai_client:
                try:
                    self.azure_openai_client.close()
                except Exception as e:
                    print(f"⚠️  Error closing Azure OpenAI client: {e}")
            
            print(f"\n👋 CHIPPY shutting down. Total interactions: {self.interaction_count}")
            if self.azure_openai_client:
                print(f"📊 {self.azure_openai_client.get_conversation_summary()}")
            print("=" * 70)


def stream_and_speak_openai(azure_openai_client: AzureOpenAIClient,
                           tts_client,
                           privacy_manager,
                           user_text: str,
                           device_index: Optional[int] = None,
                           verbose: bool = True) -> dict:
    """
    Stream Azure OpenAI response and speak sentences as they arrive.
    This enables low-latency responses by synthesizing and speaking in parallel.
    
    Args:
        azure_openai_client: Azure OpenAI client instance
        tts_client: TTS client for speech synthesis
        privacy_manager: Privacy manager for de-anonymization
        user_text: User's input text (may be anonymized)
        device_index: Audio device index for playback
        verbose: Print status messages
        
    Returns:
        dict with 'success': bool, 'full_response': str, 'interrupted': bool
    """
    # Queue for sentences ready to be spoken
    sentence_queue = queue.Queue()
    
    # Flags and state
    streaming_done = threading.Event()
    speaking_done = threading.Event()
    full_response = []
    interrupted = False
    error_occurred = False
    
    def streaming_thread():
        """Background thread that streams and queues sentences."""
        nonlocal error_occurred
        try:
            sentence_streamer = SentenceStreamer()
            
            # Get streaming response from Azure OpenAI
            for chunk in azure_openai_client.get_streaming_response(user_text):
                full_response.append(chunk)
                
                # Check for complete sentences
                for sentence in sentence_streamer.add_chunk(chunk):
                    if verbose:
                        print(f"📝 Queued: \"{sentence[:50]}...\"")
                    sentence_queue.put(sentence)
            
            # Flush any remaining text
            remaining = sentence_streamer.flush()
            if remaining:
                if verbose:
                    print(f"📝 Queued (final): \"{remaining[:50]}...\"")
                sentence_queue.put(remaining)
            
            # Signal that streaming is done
            streaming_done.set()
            sentence_queue.put(None)  # Sentinel value
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            import traceback
            traceback.print_exc()
            error_occurred = True
            streaming_done.set()
            sentence_queue.put(None)
    
    def speaking_thread():
        """Background thread that synthesizes and plays sentences."""
        nonlocal interrupted
        
        try:
            sentence_count = 0
            
            while True:
                # Get next sentence (blocking)
                sentence = sentence_queue.get()
                
                # Check for sentinel (end of stream)
                if sentence is None:
                    break
                
                sentence_count += 1
                
                # Restore privacy (de-anonymize)
                restored_sentence = privacy_manager.restore_personal_response(sentence)
                
                if verbose:
                    print(f"\n🔊 Speaking sentence {sentence_count}: \"{restored_sentence[:60]}...\"")
                
                # Synthesize speech
                try:
                    audio_file = tts_client.synthesize_speech(restored_sentence)
                    
                    # Play with interrupt detection
                    playback_result = tts_client.play_speech_interruptible(
                        audio_file,
                        device_index=device_index
                    )
                    
                    # Check if interrupted
                    if playback_result.get('interrupted', False):
                        interrupted = True
                        if verbose:
                            print("⚠️  User interrupted, stopping playback...")
                        break
                        
                except Exception as e:
                    print(f"⚠️  TTS/playback error for sentence {sentence_count}: {e}")
                    # Continue with next sentence
                    continue
            
            speaking_done.set()
            
        except Exception as e:
            print(f"❌ Speaking error: {e}")
            import traceback
            traceback.print_exc()
            speaking_done.set()
    
    # Start both threads
    streamer = threading.Thread(target=streaming_thread, daemon=True)
    speaker = threading.Thread(target=speaking_thread, daemon=True)
    
    if verbose:
        print("🧠 Starting Azure OpenAI streaming response...")
    
    streamer.start()
    
    # Wait a tiny bit for first sentence to arrive, then start speaking
    time.sleep(0.1)
    speaker.start()
    
    # Wait for both to complete
    streaming_done.wait()
    speaking_done.wait()
    
    # Combine full response
    full_text = "".join(full_response)
    
    return {
        'success': not error_occurred,
        'full_response': full_text,
        'interrupted': interrupted
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CHIPPY Raspberry Pi Voice Loop with Azure OpenAI")
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
    parser.add_argument(
        "--grade",
        type=int,
        default=None,
        help="Override student grade level from environment"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Override current topic from environment"
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
    
    # Override grade/topic if provided
    if args.grade:
        os.environ["STUDENT_GRADE"] = str(args.grade)
    if args.topic:
        os.environ["CURRENT_TOPIC"] = args.topic
    
    # Run CHIPPY
    try:
        chippy = ChippyVoiceLoop()
        chippy.run(device_index=args.device, test_mode=args.test)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()