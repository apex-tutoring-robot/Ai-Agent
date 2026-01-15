"""
Azure Speech-to-Text client with streaming support.
Converts captured audio to text using Azure Cognitive Services.
"""

import os
import logging
from typing import Optional, Iterator, Generator
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import threading

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class SpeechToTextClient:
    """Azure Speech-to-Text client."""
    
    def __init__(
        self,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        language: str = "en-US"
    ):
        """
        Initialize Speech-to-Text client.
        
        Args:
            speech_key: Azure Speech API key
            speech_region: Azure Speech region
            language: Recognition language
        """
        self.speech_key = speech_key or os.getenv('AZURE_SPEECH_KEY')
        self.speech_region = speech_region or os.getenv('AZURE_SPEECH_REGION')
        self.language = language
        
        if not self.speech_key or not self.speech_region:
            raise ValueError("Azure Speech credentials not provided")
        
        # Configure speech
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        self.speech_config.speech_recognition_language = self.language
    
    def recognize_from_audio_data(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        Recognize speech from raw audio data.
        
        Args:
            audio_data: Raw audio bytes (PCM 16-bit)
            sample_rate: Audio sample rate
        
        Returns:
            Transcribed text
        """
        try:
            # Create audio stream from bytes
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=sample_rate,
                bits_per_sample=16,
                channels=1
            )
            
            # Create push stream
            push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            push_stream.write(audio_data)
            push_stream.close()
            
            # Create audio config
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            
            # Create recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Perform recognition
            logger.info("Recognizing speech...")
            result = speech_recognizer.recognize_once()
            
            # Check result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(f"Recognized: {result.text}")
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech could be recognized")
                return ""
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(f"Speech recognition canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {cancellation.error_details}")
                return ""
            
            return ""
        
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            raise
    
    def recognize_from_microphone(self, device_id: Optional[str] = None) -> str:
        """
        Recognize speech directly from microphone (alternative method).
        
        Args:
            device_id: Microphone device ID
        
        Returns:
            Transcribed text
        """
        try:
            # Create audio config from microphone
            if device_id:
                audio_config = speechsdk.audio.AudioConfig(device_name=device_id)
            else:
                audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            
            # Create recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            logger.info("Listening from microphone...")
            result = speech_recognizer.recognize_once()
            
            # Check result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(f"Recognized: {result.text}")
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech could be recognized")
                return ""
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(f"Speech recognition canceled: {cancellation.reason}")
                return ""
            
            return ""
        
        except Exception as e:
            logger.error(f"Error in microphone recognition: {e}")
            raise
    
    def recognize_streaming(self, audio_stream: Iterator[bytes], sample_rate: int = 16000) -> Generator[str, None, None]:
        """
        Recognize speech from streaming audio chunks in real-time.
        Yields partial and final recognition results as they arrive.
        
        Args:
            audio_stream: Iterator yielding audio chunks (bytes)
            sample_rate: Audio sample rate
            
        Yields:
            Transcribed text (final results only)
        """
        try:
            # Create audio stream format
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=sample_rate,
                bits_per_sample=16,
                channels=1
            )
            
            # Create push stream for feeding audio chunks
            push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            
            # Create audio config from push stream
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            
            # Create recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Store results - collect ALL recognition events
            recognized_texts = []
            stream_ended = threading.Event()
            recognition_error = None
            
            # Event handlers
            first_recognition_time = None
            recognition_event_count = 0
            
            def recognized_handler(evt):
                """Handle final recognition results."""
                nonlocal first_recognition_time, recognition_event_count
                recognition_event_count += 1
                logger.info(f"🎯 Recognition event #{recognition_event_count}: reason={evt.result.reason}")
                
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    # LATENCY: Track first text recognition for STT latency
                    if first_recognition_time is None:
                        first_recognition_time = time.perf_counter()
                    
                    # Log what was actually recognized
                    logger.info(f"✓ Recognized: '{evt.result.text}' (length: {len(evt.result.text)})")
                    
                    if evt.result.text:  # Only add non-empty results
                        recognized_texts.append(evt.result.text)
                    else:
                        logger.warning("⚠️  RecognizedSpeech event but text is EMPTY!")
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.warning(f"❌ No speech matched - NoMatchReason: {evt.result.no_match_details}")
            
            def recognizing_handler(evt):
                """Handle interim recognition results."""
                logger.debug(f"🔄 Recognizing (interim): {evt.result.text}")
            
            def canceled_handler(evt):
                """Handle cancellation."""
                nonlocal recognition_error
                logger.error(f"Recognition canceled: {evt.reason}")
                if evt.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {evt.error_details}")
                    recognition_error = evt.error_details
                stream_ended.set()
            
            def session_stopped_handler(evt):
                """Handle session stopped."""
                logger.info("🏁 Recognition session stopped event fired")
                logger.info("🔔 Setting stream_ended event")
                stream_ended.set()
            
            # Subscribe to events
            speech_recognizer.recognized.connect(recognized_handler)
            speech_recognizer.recognizing.connect(recognizing_handler)  # Track interim results
            speech_recognizer.canceled.connect(canceled_handler)
            speech_recognizer.session_stopped.connect(session_stopped_handler)
            
            # Start continuous recognition
            logger.info("Starting streaming recognition...")
            speech_recognizer.start_continuous_recognition()
            
            chunk_count = 0
            try:
                # Feed audio chunks to the push stream
                logger.info("📡 Feeding audio chunks to Azure STT...")
                for audio_chunk in audio_stream:
                    if audio_chunk is None:
                        # End of stream signal
                        logger.info(f"🛑 End of audio stream detected (received {chunk_count} chunks total)")
                        break
                    push_stream.write(audio_chunk)
                    chunk_count += 1
                    if chunk_count % 50 == 0:  # Log every 50 chunks
                        logger.info(f"📊 Processed {chunk_count} audio chunks...")
                
                logger.info(f"✅ Finished feeding {chunk_count} chunks, closing push stream...")
                # Close the push stream to signal end of audio
                push_stream.close()
                logger.info("✅ Push stream closed, waiting for recognition to complete...")
                
                # Wait for recognition session to complete (with timeout)
                if stream_ended.wait(timeout=10.0):
                    logger.info(f"✅ Recognition session ended. Collected {len(recognized_texts)} text segments")
                    if recognition_error:
                        raise Exception(f"Recognition error: {recognition_error}")
                else:
                    logger.warning(f"⚠️  Recognition session didn't end within timeout, stopping manually. Collected {len(recognized_texts)} segments so far")
                
                # Yield all accumulated text as a single result with timing
                combined_text = " ".join(recognized_texts)
                logger.info(f"📝 Final combined text: '{combined_text}' (from {len(recognized_texts)} segments)")
                # Return tuple: (text, first_recognition_time)
                yield (combined_text, first_recognition_time)
                    
            finally:
                # Stop recognition
                logger.info("🛑 Stopping continuous recognition...")
                speech_recognizer.stop_continuous_recognition()
                
        except Exception as e:
            logger.error(f"Error in streaming recognition: {e}")
            raise


if __name__ == "__main__":
    # Test STT client
    print("Testing Azure Speech-to-Text client...")
    print("Speak into your microphone when prompted.\n")
    
    try:
        from src.audio.continuous_vad import ContinuousVADCapture
        client = SpeechToTextClient()
        
        input("Press Enter to start recording from microphone...")
        continuous_vad = ContinuousVADCapture()
        audio_stream = continuous_vad.stream_audio_chunks()
        text = client.recognize_streaming(audio_stream)
        
        if text:
            print(f"\nTranscribed text: {text}")
        else:
            print("\nNo speech recognized")
    
    except Exception as e:
        print(f"Error: {e}")
