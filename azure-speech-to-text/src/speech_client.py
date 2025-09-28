"""
Azure Speech Client for CHIPPY.
Handles speech recognition using Azure Cognitive Services.
"""

import os
import time
import azure.cognitiveservices.speech as speechsdk
from typing import Optional, Callable, Dict, Any

from .config import Config
from .privacy_manager import PrivacyManager

class SpeechClient:
    """Client for Azure Speech-to-Text service."""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the speech client.
        
        Args:
            session_id: Optional session ID. If not provided, one will be generated.
        """
        Config.validate_config()
        
        # Set up session tracking
        self.session_id = session_id or Config.generate_session_id()
        self.privacy_manager = PrivacyManager(self.session_id)
        
        # Initialize the speech config
        self.speech_config = speechsdk.SpeechConfig(
            subscription=Config.SPEECH_KEY, 
            region=Config.SPEECH_REGION
        )
        
        # Set speech recognition language
        self.speech_config.speech_recognition_language = "en-US"
    
    def recognize_from_microphone(self, 
                                  timeout_ms: int = 10000,
                                  anonymize: bool = True,
                                  callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Recognize speech from the microphone.
        
        Args:
            timeout_ms: Recognition timeout in milliseconds
            anonymize: Whether to anonymize the recognized text
            callback: Optional callback function to receive intermediate results
            
        Returns:
            Dictionary containing the recognition results
        """
        # Create the audio configuration for the microphone
        try:
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            
            # Create speech recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            # Set up result container
            result = {
                "recognized_text": "",
                "is_final": False,
                "error": None,
                "anonymized": False,
                "mappings": {},
                "session_id": self.session_id
            }
            
            # Set up recognition callbacks
            def recognized_cb(evt):
                result["recognized_text"] = evt.result.text
                result["is_final"] = True
                
                # Anonymize if requested
                if anonymize and result["recognized_text"]:
                    anonymized_text, mappings = self.privacy_manager.anonymize_for_llm(result["recognized_text"])
                    result["anonymized"] = True
                    result["original_text"] = result["recognized_text"]
                    result["recognized_text"] = anonymized_text
                    result["mappings"] = mappings
                
                # Call user callback if provided
                if callback:
                    callback(result["recognized_text"])
                    
            def canceled_cb(evt):
                result["error"] = f"Recognition canceled: {evt.result.cancellation_details.reason}"
                if evt.result.cancellation_details.reason == speechsdk.CancellationReason.Error:
                    result["error"] += f": {evt.result.cancellation_details.error_details}"
            
            # Connect callbacks
            speech_recognizer.recognized.connect(recognized_cb)
            speech_recognizer.canceled.connect(canceled_cb)
            
            # Start recognition
            speech_recognizer.start_continuous_recognition()
            
            # Wait for recognition to complete or timeout
            start_time = time.time()
            while not result["is_final"] and not result["error"]:
                time.sleep(0.1)
                if (time.time() - start_time) * 1000 > timeout_ms:
                    result["error"] = "Recognition timed out"
                    break
            
            # Stop recognition
            speech_recognizer.stop_continuous_recognition()
            
            return result
        except Exception as e:
            return {
                "recognized_text": "",
                "is_final": True,
                "error": f"Microphone error: {str(e)}",
                "anonymized": False,
                "mappings": {},
                "session_id": self.session_id
            }
    
    def recognize_from_file(self, 
                           audio_file_path: str,
                           anonymize: bool = True,
                           callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Recognize speech from an audio file.
        
        Args:
            audio_file_path: Path to the audio file
            anonymize: Whether to anonymize the recognized text
            callback: Optional callback function to receive intermediate results
            
        Returns:
            Dictionary containing the recognition results
        """
        # Check if file exists
        if not os.path.exists(audio_file_path):
            return {
                "recognized_text": "",
                "is_final": True,
                "error": f"Audio file not found: {audio_file_path}",
                "anonymized": False,
                "mappings": {},
                "session_id": self.session_id
            }
        
        # Create the audio configuration from a file
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        
        # Create speech recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, 
            audio_config=audio_config
        )
        
        # Set up result container
        result = {
            "recognized_text": "",
            "is_final": False,
            "error": None,
            "anonymized": False,
            "mappings": {},
            "session_id": self.session_id,
            "audio_source": audio_file_path
        }
        
        # Set up recognition callbacks
        def recognized_cb(evt):
            result["recognized_text"] = evt.result.text
            result["is_final"] = True
            
            # Anonymize if requested
            if anonymize and result["recognized_text"]:
                anonymized_text, mappings = self.privacy_manager.anonymize_for_llm(result["recognized_text"])
                result["anonymized"] = True
                result["original_text"] = result["recognized_text"]
                result["recognized_text"] = anonymized_text
                result["mappings"] = mappings
            
            # Call user callback if provided
            if callback:
                callback(result["recognized_text"])
                
        def canceled_cb(evt):
            result["error"] = f"Recognition canceled: {evt.result.cancellation_details.reason}"
            if evt.result.cancellation_details.reason == speechsdk.CancellationReason.Error:
                result["error"] += f": {evt.result.cancellation_details.error_details}"
        
        # Start recognition and wait for completion
        speech_result = speech_recognizer.recognize_once()
        
        # Process result
        if speech_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            result["recognized_text"] = speech_result.text
            result["is_final"] = True
            
            # Anonymize if requested
            if anonymize and result["recognized_text"]:
                anonymized_text, mappings = self.privacy_manager.anonymize_for_llm(result["recognized_text"])
                result["anonymized"] = True
                result["original_text"] = result["recognized_text"]
                result["recognized_text"] = anonymized_text
                result["mappings"] = mappings
                
            # Call user callback if provided
            if callback:
                callback(result["recognized_text"])
        elif speech_result.reason == speechsdk.ResultReason.NoMatch:
            result["error"] = "No speech could be recognized"
        elif speech_result.reason == speechsdk.ResultReason.Canceled:
            cancellation = speech_result.cancellation_details
            result["error"] = f"Recognition canceled: {cancellation.reason}"
            if cancellation.reason == speechsdk.CancellationReason.Error:
                result["error"] += f": {cancellation.error_details}"
                
        return result
    
    def restore_personal_context(self, response_text: str) -> str:
        """
        Restore personal context in the response using the privacy manager.
        
        Args:
            response_text: The anonymized response text
            
        Returns:
            Text with personal context restored
        """
        return self.privacy_manager.restore_personal_response(response_text)