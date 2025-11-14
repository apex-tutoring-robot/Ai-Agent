"""
Streaming Speech-to-Text client for CHIPPY using Azure Speech SDK.
Provides real-time speech recognition with lower latency than REST API.
"""

import azure.cognitiveservices.speech as speechsdk
from typing import Optional, Callable, Dict, Any
import time


class StreamingSpeechClient:
    """Client for Azure Speech-to-Text service using streaming SDK."""
    
    def __init__(self, config, privacy_manager, session_id: Optional[str] = None):
        """
        Initialize the streaming speech client.
        
        Args:
            config: Configuration object with Azure credentials
            privacy_manager: Privacy manager for data anonymization
            session_id: Optional session ID for tracking
        """
        # Set up session tracking and privacy
        self.session_id = session_id or config.generate_session_id()
        self.privacy_manager = privacy_manager
        
        # Create speech configuration
        self.speech_config = speechsdk.SpeechConfig(
            subscription=config.SPEECH_KEY,
            region=config.SPEECH_REGION
        )
        
        # Configure recognition settings for best quality
        self.speech_config.speech_recognition_language = "en-US"
        
        # Enable detailed results
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed
        
        # Configure for better accuracy with educational content
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, 
            "15000"  # 15 seconds initial silence timeout
        )
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, 
            "2000"  # 2 seconds end silence timeout
        )
        
        # Profanity filter
        self.speech_config.set_profanity(speechsdk.ProfanityOption.Masked)
    
    def recognize_from_file(self, 
                           audio_file_path: str,
                           anonymize: bool = True,
                           callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Recognize speech from an audio file using streaming SDK.
        This is faster than REST API because it uses WebSocket connection.
        
        Args:
            audio_file_path: Path to the audio file (WAV format)
            anonymize: Whether to anonymize the recognized text
            callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing recognition results
        """
        try:
            if callback:
                callback("Initializing streaming recognition...")
            
            # Create audio configuration from file
            audio_config = speechsdk.AudioConfig(filename=audio_file_path)
            
            # Create speech recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            if callback:
                callback("Processing audio stream...")
            
            # Perform recognition
            # recognize_once_async() is faster than REST API because:
            # 1. Uses WebSocket (persistent connection, no HTTP overhead)
            # 2. Starts processing immediately
            # 3. Returns as soon as speech ends (no waiting for full file upload)
            result = recognizer.recognize_once_async().get()
            
            # Process result based on reason
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = result.text
                
                # Apply privacy anonymization if requested
                if anonymize and text:
                    anonymized_text, mappings = self.privacy_manager.anonymize_for_llm(text)
                    return {
                        "recognized_text": anonymized_text,
                        "original_text": text,
                        "anonymized": True,
                        "mappings": mappings,
                        "session_id": self.session_id,
                        "confidence": self._get_confidence(result)
                    }
                else:
                    return {
                        "recognized_text": text,
                        "anonymized": False,
                        "session_id": self.session_id,
                        "confidence": self._get_confidence(result)
                    }
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return {
                    "error": "No speech could be recognized",
                    "details": result.no_match_details.reason
                }
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                error_msg = f"Recognition canceled: {cancellation.reason}"
                
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    error_msg += f" - Error details: {cancellation.error_details}"
                
                return {"error": error_msg}
            
            else:
                return {"error": f"Unexpected result reason: {result.reason}"}
        
        except Exception as e:
            return {"error": f"Recognition failed: {str(e)}"}
    
    def _get_confidence(self, result) -> Optional[float]:
        """
        Extract confidence score from recognition result.
        
        Args:
            result: SpeechRecognitionResult object
            
        Returns:
            Confidence score (0.0 to 1.0) or None if not available
        """
        try:
            # Try to get confidence from detailed results
            import json
            if hasattr(result, 'json') and result.json:
                result_json = json.loads(result.json)
                if 'NBest' in result_json and len(result_json['NBest']) > 0:
                    return result_json['NBest'][0].get('Confidence', None)
        except:
            pass
        
        return None
    
    def recognize_from_microphone_continuous(self,
                                            device_index: Optional[int] = None,
                                            callback: Optional[Callable[[str], None]] = None,
                                            timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Recognize speech directly from microphone with continuous recognition.
        This is the FASTEST method - no file I/O, processes audio in real-time.
        
        Args:
            device_index: Audio device index (None for default)
            callback: Callback for real-time transcription updates
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary containing recognition results
        """
        try:
            # Create audio configuration for microphone
            if device_index is not None:
                # Note: Azure SDK has limited device selection support
                # For specific devices, you may need to set system default
                audio_config = speechsdk.AudioConfig(use_default_microphone=True)
            else:
                audio_config = speechsdk.AudioConfig(use_default_microphone=True)
            
            # Create speech recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Storage for results
            recognized_texts = []
            done = False
            
            # Event handlers for continuous recognition
            def recognizing_handler(evt):
                """Called for interim results (real-time feedback)."""
                if callback:
                    callback(f"Recognizing: {evt.result.text}")
            
            def recognized_handler(evt):
                """Called when a phrase is fully recognized."""
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    recognized_texts.append(evt.result.text)
                    if callback:
                        callback(f"Recognized: {evt.result.text}")
            
            def canceled_handler(evt):
                """Called if recognition is canceled."""
                nonlocal done
                done = True
                if callback:
                    callback(f"Recognition canceled: {evt.cancellation_details.reason}")
            
            def session_stopped_handler(evt):
                """Called when session stops."""
                nonlocal done
                done = True
            
            # Connect event handlers
            recognizer.recognizing.connect(recognizing_handler)
            recognizer.recognized.connect(recognized_handler)
            recognizer.canceled.connect(canceled_handler)
            recognizer.session_stopped.connect(session_stopped_handler)
            
            # Start continuous recognition
            recognizer.start_continuous_recognition_async().get()
            
            if callback:
                callback("Listening... Speak now!")
            
            # Wait for recognition to complete (or timeout)
            start_time = time.time()
            while not done:
                time.sleep(0.1)
                if timeout and (time.time() - start_time) > timeout:
                    break
            
            # Stop recognition
            recognizer.stop_continuous_recognition_async().get()
            
            # Combine all recognized texts
            full_text = " ".join(recognized_texts)
            
            if full_text:
                # Apply privacy anonymization
                anonymized_text, mappings = self.privacy_manager.anonymize_for_llm(full_text)
                return {
                    "recognized_text": anonymized_text,
                    "original_text": full_text,
                    "anonymized": True,
                    "mappings": mappings,
                    "session_id": self.session_id
                }
            else:
                return {"error": "No speech recognized"}
        
        except Exception as e:
            return {"error": f"Microphone recognition failed: {str(e)}"}


# Convenience function for backward compatibility
def create_speech_client(config, privacy_manager, session_id=None, use_streaming=True):
    """
    Factory function to create the appropriate speech client.
    
    Args:
        config: Configuration object
        privacy_manager: Privacy manager instance
        session_id: Optional session ID
        use_streaming: If True, use streaming client (recommended)
        
    Returns:
        Speech client instance
    """
    if use_streaming:
        return StreamingSpeechClient(config, privacy_manager, session_id)
    else:
        from rest_speech_client import RestSpeechClient
        return RestSpeechClient(config, privacy_manager, session_id)