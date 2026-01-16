"""
Jarvis - Main Orchestrator
Coordinates all components for the LLM-powered tutoring robot.
"""

import os
import sys
import logging
import time
import asyncio
import threading
from typing import Optional
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import components
from audio.wake_word import WakeWordDetector
from audio.single_turn_vad import VADAudioCapture
from audio.continuous_vad import ContinuousVADCapture
from audio.playback import AudioPlayer
from azure_services.stt_client import SpeechToTextClient
from azure_services.llm_client import LLMClient
from azure_services.tts_client import TextToSpeechClient
from privacy.privacy_manager import PrivacyManager
from conversation.state_manager import ConversationStateManager
import pyaudio

# Load environment
load_dotenv("/home/pi/Desktop/Ai-Agent 2.0/Jarvis/config/.env")
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JarvisBot:
    """Main orchestrator for Jarvis tutoring robot."""
    
    def __init__(self):
        """Initialize Jarvis with all components."""
        logger.info("Initializing Jarvis...")
        
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
        
        logger.info("Jarvis initialized successfully!")
    
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
            continuous_vad.last_speech_time = time.time()
            while True:
                # Check if idle timeout exceeded
                idle_duration = time.time() - continuous_vad.last_speech_time
                if idle_duration >= idle_timeout:
                    logger.info(f"⏱️  {idle_timeout}s idle timeout reached - ending conversation")
                    break
                
                turn_count += 1
                logger.info(f"\n💬 Turn {turn_count}")
                self.end_time = time.time()
                
                # Process this turn with streaming STT
                success = self._process_turn_streaming(continuous_vad)
                
                if not success:
                    # No speech detected, continue waiting
                    continue
            logger.info(f"\n👋 Conversation ended ({turn_count} turns)")
            # for audio_data in continuous_vad.listen_continuous():
            #     if audio_data is None:
            #         # Timeout reached
            #         break
                
            #     turn_count += 1
            #     logger.info(f"\n💬 Turn {turn_count}")
                
            #     # Process this turn
            #     self.end_time = time.time()
            #     if turn_count == 1:
            #         logger.info(f"⏱️ latency to start STT upon detection of wake word : {(self.end_time - self.start_time):.3f}s")
            #     # interrupted = self._process_turn(audio_data, continuous_vad)
            #     self._process_turn(audio_data, continuous_vad)
            #     # if interrupted:
            #     #     logger.info("🛑 Conversation interrupted by wake word")
            #     #     break
            
            # logger.info(f"\n👋 Conversation ended ({turn_count} turns)")
            # logger.info(f"📊 {self.conversation_manager}")
            
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
            
            # Start audio player streaming
            self.audio_player.start_streaming()
            
            # Collect response text chunks as they stream
            response_chunks = []
            try:
                # Create a wrapper that collects chunks while streaming
                def text_chunk_collector(llm_stream):
                    """Collect text chunks while passing them through."""

                    for chunk in llm_stream:
                        response_chunks.append(chunk)
                        if len(response_chunks) == 1:
                            logger.info(f"⏱️  LLM TTFT: {(time.time() - llm_start):.3f}s")

                        yield chunk
                
                # Stream LLM output through collector to TTS
                llm_start = time.time()
                llm_stream = self.llm_client.generate_response_stream(messages)

                # Now synthesize TTS from collected text
                tts_start = time.perf_counter()
                response_text_stream = text_chunk_collector(llm_stream)  # Iterator over chunks
                tts_stream = self.tts_client.synthesize_stream(response_text_stream)

                # Stream audio to player
                for audio_chunk in tts_stream:
                    self.audio_player.queue_audio(audio_chunk)
                
                # Wait for playback to complete
                self.audio_player.stop_streaming()
                
                # Combine collected chunks into full response
                response_text = ''.join(response_chunks)
                logger.info(f"🤖 Jarvis: {response_text}")
                
                # Add assistant response to conversation
                self.conversation_manager.add_assistant_message(response_text)
                
                # Reset idle timer after our response
                continuous_vad.reset_idle_timer()
                
            except Exception as e:
                logger.error(f"Error in streaming pipeline: {e}")
                self.audio_player.stop_streaming()
                raise
            
            return False  # No interruption possible in continuous mode
        
        except Exception as e:
            logger.error(f"Error processing turn: {e}")
            return False
    
    def _process_turn_streaming(self, continuous_vad: ContinuousVADCapture) -> bool:
        """ Process one turn of the conversation using streaming STT.
        Args:
            continuous_vad: Continuous VAD instance
            
        Returns:
            True if speech was processed, False otherwise
        """
        try:
            # Step 1: Stream audio chunks to STT
            logger.info("☁️  Starting streaming speech recognition...")
            stt_start = time.perf_counter()
            
            # Get streaming audio chunks from VAD
            audio_stream = continuous_vad.stream_audio_chunks()
            
            # Stream to Azure STT
            user_text = ""
            first_text_time = None
            for result_tuple in self.stt_client.recognize_streaming(audio_stream):
                # Unpack tuple: (text, first_recognition_time)
                text_result, first_text_time = result_tuple
                user_text = text_result  # Get the complete text
            
            # Check if we got any speech after all recognition events
            if not user_text.strip():
                    logger.warning("No speech recognized")
                    return False
            
            # LATENCY METRICS
            # 1. STT Latency: User stops speaking (silence detected) → First text recognized
            if continuous_vad.silence_detected_time and first_text_time:
                stt_latency = first_text_time - continuous_vad.silence_detected_time
                logger.info(f"⏱️  STT Latency: {stt_latency:.3f}s")
            
            # 2. End-to-TTFT: Will be calculated when first LLM token arrives

            logger.info(f"📝 Student: {user_text}")
            
            # Step 2: Anonymize PII
            anonymized_text = self.privacy_manager.anonymize(user_text)
            # Step 3: Add to conversation history
            self.conversation_manager.add_user_message(anonymized_text)
            # Step 4: Generate LLM response with streaming
            logger.info("🧠 Generating response...")
            llm_start = time.perf_counter()
            messages = self.conversation_manager.get_messages()
            
            # Collect response text chunks as they stream
            response_chunks = []
            first_token = True
            try:
                # Start audio player BEFORE first audio arrives for lower latency
                self.audio_player.start_streaming()
                
                # Create a wrapper that collects chunks while streaming
                def text_chunk_collector(llm_stream):
                    """Collect text chunks while passing them through."""
                    nonlocal first_token
                    first_token_time = None
                    for chunk in llm_stream:
                        response_chunks.append(chunk)
                        if first_token:
                            first_token_time = time.perf_counter()
                            llm_latency = first_token_time - llm_start
                            
                            # LATENCY: End-to-TTFT (Silence detected → First LLM token)
                            if continuous_vad.silence_detected_time:
                                end_to_ttft = first_token_time - continuous_vad.silence_detected_time
                                logger.info(f"⏱️  End-to-TTFT: {end_to_ttft:.3f}s")
                            
                            logger.info(f"⏱️  LLM TTFT: {llm_latency:.3f}s")
                            
                            # Track for TTFAS calculation
                            self.audio_player.first_token_time = first_token_time
                            first_token = False
                        yield chunk
                
                # Stream LLM output through collector to TTS
                llm_stream = self.llm_client.generate_response_stream(messages)
                collected_stream = text_chunk_collector(llm_stream)
                tts_stream = self.tts_client.synthesize_stream(collected_stream)
                
                # Stream audio to player (player already started, audio plays immediately)
                for audio_chunk in tts_stream:
                    self.audio_player.queue_audio(audio_chunk)
                
                # Wait for playback to complete
                self.audio_player.stop_streaming()
                
                # Combine collected chunks into full response
                response_text = ''.join(response_chunks)
                logger.info(f"🤖 Chippy: {response_text}")
                # Add assistant response to conversation
                self.conversation_manager.add_assistant_message(response_text)
                
                # Reset idle timer AFTER bot finishes speaking
                # This ensures we don't timeout while bot is generating/speaking
                continuous_vad.reset_idle_timer()
                
                return True
                
            except Exception as e:
                logger.error(f"Error in streaming pipeline: {e}")
                self.audio_player.stop_streaming()
                raise
            
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
        """Start Jarvis and run the main loop."""
        logger.info("\n" + "🤖 "*20)
        logger.info("Jarvis TUTORING ROBOT STARTED")
        logger.info("🤖 "*20 + "\n")
        logger.info("Listening for wake word: 'Hey Jarvis'")
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
        """Stop Jarvis and cleanup resources."""
        logger.info("Shutting down Jarvis...")
        
        self._is_running = False
        
        # Stop wake word detector
        if self.wake_word_detector:
            self.wake_word_detector.stop()
        
        # Cleanup audio player
        if self.audio_player:
            self.audio_player.cleanup()
        
        # Show conversation summary
        logger.info(f"\nFinal conversation state: {self.conversation_manager}")
        
        logger.info("Jarvis shutdown complete. Goodbye! 👋\n")


def main():
    """Main entry point."""
    try:
        Jarvis = JarvisBot()
        Jarvis.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
