import os
import json
import time
import requests
from typing import Optional, Dict, Any, Callable

class RestSpeechClient:
    """Client for Azure Speech-to-Text service using REST API."""
    
    def __init__(self, config, privacy_manager, session_id: Optional[str] = None):
        """
        Initialize the REST speech client.
        
        Args:
            config: Configuration object with Azure credentials
            privacy_manager: Privacy manager for data anonymization
            session_id: Optional session ID for tracking
        """
        # Set up session tracking and privacy
        self.session_id = session_id or config.generate_session_id()
        self.privacy_manager = privacy_manager
        
        # API endpoints
        self.region = config.SPEECH_REGION
        self.key = config.SPEECH_KEY
        self.token_url = f"https://{self.region}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
        self.recognition_url = f"https://{self.region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        
        # Get access token
        self.access_token = self._get_token()
        self.token_expiry = time.time() + 540  # Tokens valid for ~10 minutes, refresh after 9
    
    def _get_token(self):
        """Get authentication token for Speech service."""
        response = requests.post(
            self.token_url, 
            headers={'Ocp-Apim-Subscription-Key': self.key}
        )
        
        if response.status_code != 200:
            raise Exception(f"Token request failed: {response.status_code}")
            
        return response.text
    
    def _ensure_valid_token(self):
        """Ensure we have a valid token, refreshing if necessary."""
        if time.time() > self.token_expiry:
            self.access_token = self._get_token()
            self.token_expiry = time.time() + 540
    
    def recognize_from_file(self, 
                           audio_file_path: str,
                           anonymize: bool = True,
                           callback: Optional[Callable[[str], None]] = None):
        """
        Recognize speech from an audio file using REST API.
        
        Args:
            audio_file_path: Path to the audio file
            anonymize: Whether to anonymize the recognized text
            callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing recognition results
        """
        # Ensure token is valid
        self._ensure_valid_token()
        
        # Read audio file
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Set up headers and parameters
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'audio/wav',
            'Accept': 'application/json'
        }
        
        params = {
            'language': 'en-US',
            'format': 'detailed',
            'profanity': 'masked'
        }
        
        # Send request with exponential backoff retry
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if callback:
                    callback("Processing audio...")
                    
                response = requests.post(
                    self.recognition_url, 
                    params=params,
                    headers=headers, 
                    data=audio_data,
                    timeout=30
                )
                
                # Process successful response
                if response.status_code == 200:
                    data = response.json()
                    
                    if data['RecognitionStatus'] == 'Success':
                        text = data['DisplayText']
                        
                        # Apply privacy anonymization if requested
                        if anonymize and text:
                            anonymized_text, mappings = self.privacy_manager.anonymize_for_llm(text)
                            return {
                                "recognized_text": anonymized_text,
                                "original_text": text,
                                "anonymized": True,
                                "mappings": mappings,
                                "session_id": self.session_id
                            }
                        else:
                            return {
                                "recognized_text": text,
                                "anonymized": False,
                                "session_id": self.session_id
                            }
                    else:
                        return {"error": f"Recognition failed: {data['RecognitionStatus']}"}
                
                # Handle authentication errors
                elif response.status_code == 401:
                    # Token expired, get a new one
                    self.access_token = self._get_token()
                    self.token_expiry = time.time() + 540
                    # Retry immediately with new token
                    continue
                    
                # Handle other errors
                else:
                    if attempt < max_retries - 1:
                        if callback:
                            callback(f"Retrying... (attempt {attempt+2}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return {
                            "error": f"Request failed after {max_retries} attempts. Status: {response.status_code}"
                        }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    if callback:
                        callback(f"Network error, retrying... (attempt {attempt+2}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    return {"error": f"Network error after {max_retries} attempts: {str(e)}"}
        
        return {"error": "Recognition failed for unknown reasons"}