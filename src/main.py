"""
Chippy - Main Orchestrator
Coordinates all components for the LLM-powered tutoring robot.
"""

import os
import sys
import logging
import time
import asyncio
import threading
import numpy as np
from typing import Optional
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import components
from audio.wake_word import WakeWordDetector
from audio.vad_capture import VADAudioCapture
from audio.continuous_vad import ContinuousVADCapture
from audio.playback import AudioPlayer
from azure_services.stt_client import SpeechToTextClient
from azure_services.llm_client import LLMClient
from azure_services.tts_client import TextToSpeechClient
from privacy.privacy_manager import PrivacyManager
from conversation.state_manager import ConversationStateManager
from face_animator import FaceAnimator

face_animator = FaceAnimator(face_dir="faces")

# Load environment
load_dotenv("/home/pi/Desktop/Ai-Agent 2.0/chippy/config/.env")
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChippyBot:
    """Main orchestrator for Chippy tutoring robot."""
    
    def __init__(self):
        """Initialize Chippy with all components."""
        logger.info("Initializing Chippy...")
        
        # Initialize components
        self.wake_word_detector = WakeWordDetector()
        self.vad_capture = VADAudioCapture()
        self.audio_player = AudioPlayer()
        self.stt_client = SpeechToTextClient()
        self.llm_client = LLMClient()
        self.tts_client = TextToSpeechClient()
        self.privacy_manager = PrivacyManager()
        self.conversation_manager = ConversationStateManager(
            max_history=int(os.getenv('MAX_CONVERSATION_HISTORY', 20))
        )
        
        self._is_running = False
        self._interaction_lock = threading.Lock()
        
        logger.info("Chippy initialized successfully!")
    
    def _handle_wake_word(self):
        """Handle wake word detection - enter continuous conversation mode."""
        # Prevent concurrent interactions
        if not self._interaction_lock.acquire(blocking=False):
            logger.warning("Already in conversation, ignoring wake word")
            return
        
        try:
            logger.info("\n" + "="*60)
            logger.info("🎤 WAKE WORD DETECTED - CONVERSATION MODE ACTIVATED")
            logger.info("="*60)
            logger.info(f"⏱️  Will end after 10 seconds of silence")
            
            # Stop wake word detection to free microphone
            self.wake_word_detector.stop()
            
            # Enter continuous conversation mode
            idle_timeout = int(os.getenv('CONVERSATION_IDLE_TIMEOUT_SECONDS', 10))
            continuous_vad = ContinuousVADCapture(idle_timeout_seconds=idle_timeout)
            
            turn_count = 0
            for audio_data in continuous_vad.listen_continuous():
                if audio_data is None:
                    # Timeout reached
                    break
                
                turn_count += 1
                logger.info(f"\n💬 Turn {turn_count}")
                
                # Process this turn
                interrupted = self._process_turn(audio_data, continuous_vad)
                
                if interrupted:
                    logger.info("🛑 Conversation interrupted by wake word")
                    break
            
            logger.info(f"\n👋 Conversation ended ({turn_count} turns)")
            logger.info(f"📊 {self.conversation_manager}")
            
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
        
        finally:
            # Explicitly cleanup continuous VAD to free microphone
            try:
                if 'continuous_vad' in locals():
                    continuous_vad.cleanup()
            except:
                pass
            
            # Always restart wake word detection
            logger.info("▶️  Resuming wake word detection...")
            self._restart_wake_word()
            self._interaction_lock.release()
    
    def _process_turn(self, audio_data: bytes, continuous_vad: ContinuousVADCapture) -> bool:
        """
        Process one turn of the conversation.
        
        Args:
            audio_data: Captured audio data
            continuous_vad: Continuous VAD instance (to reset idle timer)
            
        Returns:
            True if interrupted, False otherwise
        """
        try:
            # Step 1: Convert speech to text
            logger.info("☁️  Converting speech to text...")
            user_text = self.stt_client.recognize_from_audio_data(audio_data)
            
            if not user_text:
                logger.warning("No speech recognized")
                return False
            
            logger.info(f"📝 Student: {user_text}")
            
            # Step 2: Anonymize PII
            anonymized_text = self.privacy_manager.anonymize(user_text)
            
            # Step 3: Add to conversation history
            self.conversation_manager.add_user_message(anonymized_text)
            
            # Step 4: Generate LLM response with streaming
            logger.info("🧠 Generating response...")
            messages = self.conversation_manager.get_messages()

            # Thinking face while LLM works
            face_animator.set_emotion("thinking")
            face_animator.render_step()

            response_chunks = []

            try:
                def text_chunk_collector(llm_stream):
                    for chunk in llm_stream:
                        response_chunks.append(chunk)
                        yield chunk

                # Create streams
                llm_stream = self.llm_client.generate_response_stream(messages)
                collected_stream = text_chunk_collector(llm_stream)
                tts_stream = self.tts_client.synthesize_stream(collected_stream)

                # Switch to speaking face BEFORE audio
                face_animator.set_emotion("neutral")
                face_animator.render_step()

                # Start audio playback ONCE
                self.audio_player.start_streaming()

                for audio_chunk in tts_stream:
                    self.audio_player.queue_audio(audio_chunk)

                    audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                    audio_np /= 32768.0
                    rms = np.sqrt(np.mean(audio_np ** 2))

                    face_animator.update_mouth(rms)
                    face_animator.render_step()

                # End of speech
                self.audio_player.stop_streaming()
                face_animator.mouth_open = 0.0
                face_animator.set_emotion("neutral")
                face_animator.render_step()

                response_text = ''.join(response_chunks)
                logger.info(f"🤖 Chippy: {response_text}")
                self.conversation_manager.add_assistant_message(response_text)
                continuous_vad.reset_idle_timer()

            except Exception as e:
                logger.error(f"Error in streaming pipeline: {e}")
                self.audio_player.stop_streaming()
                raise

            
            return False  # No interruption possible in continuous mode
        
        except Exception as e:
            logger.error(f"Error processing turn: {e}")
            return False
    
    
    def _restart_wake_word(self):
        """Restart wake word detection in a non-blocking way."""
        if not self._is_running:
            return
        
        # Make sure previous detector is fully stopped
        if self.wake_word_detector.is_running():
            logger.info("Waiting for previous wake word detector to stop...")
            self.wake_word_detector.stop()
            time.sleep(0.5)  # Give it time to clean up
        
        # Start wake word detector in a new thread
        logger.info("Starting new wake word detection thread...")
        wake_thread = threading.Thread(
            target=self.wake_word_detector.start,
            args=(self._handle_wake_word,),
            daemon=True
        )
        wake_thread.start()
    
    def run(self):
        """Start Chippy and run the main loop."""
        logger.info("\n" + "🤖 "*20)
        logger.info("CHIPPY TUTORING ROBOT STARTED")
        logger.info("🤖 "*20 + "\n")
        logger.info("Listening for wake word: 'Hey CHIPPY'")
        logger.info("Press Ctrl+C to stop\n")
        
        self._is_running = True
        
        try:
            # Start wake word detection (blocking call)
            self.wake_word_detector.start(self._handle_wake_word)
        
        except KeyboardInterrupt:
            logger.info("\nShutdown requested by user")
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop Chippy and cleanup resources."""
        logger.info("Shutting down Chippy...")
        
        self._is_running = False
        
        # Stop wake word detector
        if self.wake_word_detector:
            self.wake_word_detector.stop()
        
        # Cleanup audio player
        if self.audio_player:
            self.audio_player.cleanup()
        
        # Show conversation summary
        logger.info(f"\nFinal conversation state: {self.conversation_manager}")
        
        logger.info("Chippy shutdown complete. Goodbye! 👋\n")


def main():
    """Main entry point."""
    try:
        chippy = ChippyBot()
        chippy.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
