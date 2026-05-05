"""
Azure Speech-to-Text client with streaming support.
Converts captured audio to text using Azure Cognitive Services.
"""

import os
import logging
import time
from typing import Optional, Iterator, Generator
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import threading
from queue import Queue

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class SpeechToTextClient:
    """Azure Speech-to-Text client."""
    
    def __init__(
        self,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        language: str = "en-US",
        user_id: Optional[str] = None,
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
        self.user_id = user_id

        if not self.speech_key or not self.speech_region:
            raise ValueError("Azure Speech credentials not provided")
        
        # Configure speech
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region,
            speech_recognition_language=self.language
        )
        
        # Warm-start the STT service
        logger.info("STT client initialized, starting warm-up...")
        self.warm_up()
    
    def warm_up(self):
        """
        Warm-start the Azure STT service to reduce first-call latency.
        Performs a dummy recognition with in-memory silence to pre-load models
        and establish connections, avoiding file I/O.
        """
        try:
            logger.info("Warming up Azure STT service with in-memory silence...")

            # Define audio format for silence
            sample_rate = 16000
            duration = 0.5  # 0.5 seconds of silence
            num_samples = int(sample_rate * duration)
            
            # Generate silent audio data (PCM 16-bit)
            silence_data = b'\x00' * (num_samples * 2)  # 2 bytes per 16-bit sample
            
            # Create an in-memory audio stream
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=sample_rate,
                bits_per_sample=16,
                channels=1
            )
            push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            push_stream.write(silence_data)
            push_stream.close()
            
            # Create audio config from the in-memory stream
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

            # Create a temporary recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )

            # Trigger recognition (this preloads everything)
            result = recognizer.recognize_once_async().get()
            
            logger.info(f"Azure STT warm-up complete. Result: {result.reason}")

        except Exception as e:
            logger.warning(f"STT warm-up failed (non-critical): {e}")

    def _recognize_once_from_config(self, audio_config: speechsdk.audio.AudioConfig) -> str:
        """
        Private helper to perform a single recognition from a given audio config.
        """
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        
        logger.info("Performing single-shot recognition...")
        result = speech_recognizer.recognize_once()
        
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
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=sample_rate,
                bits_per_sample=16,
                channels=1
            )
            push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            push_stream.write(audio_data)
            push_stream.close()
            
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            return self._recognize_once_from_config(audio_config)
        
        except Exception as e:
            logger.error(f"Error in speech recognition from data: {e}")
            raise
    
    def recognize_from_microphone(self, device_id: Optional[str] = None) -> str:
        """
        Recognize speech directly from microphone.
        
        Args:
            device_id: Microphone device ID
        
        Returns:
            Transcribed text
        """
        try:
            if device_id:
                audio_config = speechsdk.audio.AudioConfig(device_name=device_id)
            else:
                audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            
            return self._recognize_once_from_config(audio_config)
        
        except Exception as e:
            logger.error(f"Error in microphone recognition: {e}")
            raise

    def recognize_streaming(self, audio_stream: Iterator[bytes], sample_rate: int = 16000) -> Generator[tuple[str, float], None, None]:
        """
        Recognize speech from streaming audio chunks in real-time.
        This method is a generator, yielding final recognition results as they arrive.

        Args:
            audio_stream: An iterator that yields audio chunks (bytes).
            sample_rate: The sample rate of the audio.

        Yields:
            A tuple of (transcribed_text, first_recognition_time).
        """
        result_queue = Queue()
        recognition_error = None
        
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=sample_rate, bits_per_sample=16, channels=1
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=audio_config
        )

        first_recognition_time = None

        def recognized_handler(evt):
            nonlocal first_recognition_time
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                if first_recognition_time is None:
                    first_recognition_time = time.perf_counter()
                logger.info(f"✓ Recognized: '{evt.result.text}'")
                result_queue.put((evt.result.text, True, first_recognition_time))
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning(f"❌ No speech matched: {evt.result.no_match_details}")

        def canceled_handler(evt):
            nonlocal recognition_error
            logger.error(f"Recognition canceled: {evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Error details: {evt.error_details}")
                recognition_error = Exception(evt.error_details)
            result_queue.put(None)

        def session_stopped_handler(evt):
            logger.info("🏁 Recognition session stopped.")
            result_queue.put(None)

        def recognizing_handler(evt):
            nonlocal first_recognition_time
            if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech and evt.result.text:
                if first_recognition_time is None:
                    first_recognition_time = time.perf_counter()
                logger.info(f"✓ Recognizing: '{evt.result.text}'")
                result_queue.put((evt.result.text, False, first_recognition_time))
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning(f"❌ No speech matched: {evt.result.no_match_details}")


        speech_recognizer.recognized.connect(recognized_handler)
        # speech_recognizer.recognizing.connect(lambda evt: logger.debug(f"🔄 Recognizing: {evt.result.text}"))
        speech_recognizer.recognizing.connect(recognizing_handler)
        speech_recognizer.canceled.connect(canceled_handler)
        speech_recognizer.session_stopped.connect(session_stopped_handler)
        speech_recognizer.session_started.connect(lambda evt: logger.info("🎤 Recognition session started."))

        speech_recognizer.start_continuous_recognition()

        def feed_audio():
            """Feeds audio chunks from the iterator to the push stream in a separate thread."""
            try:
                for chunk in audio_stream:
                    if chunk:
                        push_stream.write(chunk)
                logger.info("✅ Finished feeding audio stream, closing push stream.")
            except Exception as e:
                logger.error(f"Error feeding audio stream: {e}")
            finally:
                push_stream.close()
        
        audio_feeder_thread = threading.Thread(target=feed_audio)
        audio_feeder_thread.start()

        try:
            while True:
                result = result_queue.get()
                if result is None:
                    break
                # result is (text, is_final, timestamp)
                yield result
            
            if recognition_error:
                raise recognition_error

        finally:
            logger.info("🛑 Stopping continuous recognition...")
            speech_recognizer.stop_continuous_recognition()
            audio_feeder_thread.join() # Ensure feeder thread is finished
            logger.info("✓ Streaming recognition finished.")


if __name__ == "__main__":
    # Test STT client
    print("Testing Azure Speech-to-Text client...")
    print("Speak into your microphone when prompted.\n")
    
    try:
        # This import will only work if you run this script from the project root
        from src.audio.continuous_vad import ContinuousVADCapture
        
        client = SpeechToTextClient()
        
        input("Press Enter to start recording from microphone for 10 seconds...")
        
        continuous_vad = ContinuousVADCapture(
            silence_threshold_ms=500,
            recording_duration_ms=10000 # Record for 10 seconds
        )
        audio_stream = continuous_vad.stream_audio_chunks()
        
        print("\n--- Transcribed Text ---")
        full_transcript = []
        for result in client.recognize_streaming(audio_stream):
            text, timestamp = result
            print(f"> {text} (at {timestamp})")
            full_transcript.append(text)
        print("------------------------\n")

        if full_transcript:
            print(f"Final Transcript: {' '.join(full_transcript)}")
        else:
            print("No speech was recognized.")
    
    except ImportError:
        print("Could not import ContinuousVADCapture. Please run this script from the project root directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

