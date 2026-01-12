"""
Azure Speech-to-Text client with streaming support.
Converts captured audio to text using Azure Cognitive Services.
"""

import os
import logging
from typing import Optional
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
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


if __name__ == "__main__":
    # Test STT client
    print("Testing Azure Speech-to-Text client...")
    print("Speak into your microphone when prompted.\n")
    
    try:
        client = SpeechToTextClient()
        
        input("Press Enter to start recording from microphone...")
        text = client.recognize_from_microphone()
        
        if text:
            print(f"\nTranscribed text: {text}")
        else:
            print("\nNo speech recognized")
    
    except Exception as e:
        print(f"Error: {e}")
