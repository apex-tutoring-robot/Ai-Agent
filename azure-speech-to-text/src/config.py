"""
Configuration module for CHIPPY's Speech-to-Text component.
Loads environment variables and provides access to configuration settings.
"""

import os
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the CHIPPY speech-to-text module."""
    
    # Azure Speech Service Configuration
    SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
    SPEECH_REGION = os.getenv('AZURE_SPEECH_REGION')
    
    # Session Configuration
    SESSION_ID_PREFIX = os.getenv('SESSION_ID_PREFIX', 'CHIPPY_')
    
    @staticmethod
    def generate_session_id():
        """Generate a unique session ID for tracking conversations."""
        return f"{Config.SESSION_ID_PREFIX}{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def validate_config():
        """Validate that all required configuration parameters are set."""
        if not Config.SPEECH_KEY:
            raise ValueError("AZURE_SPEECH_KEY environment variable is not set")
        if not Config.SPEECH_REGION:
            raise ValueError("AZURE_SPEECH_REGION environment variable is not set")
        
        return True
        # Voice Activity Detection (VAD) Configuration for Pi
    VAD_SILENCE_THRESHOLD = float(os.getenv('VAD_SILENCE_THRESHOLD', '0.015'))
    VAD_SILENCE_DURATION = float(os.getenv('VAD_SILENCE_DURATION', '2.0'))
    VAD_MIN_SPEECH_DURATION = float(os.getenv('VAD_MIN_SPEECH_DURATION', '0.5'))
    VAD_PRE_BUFFER = float(os.getenv('VAD_PRE_BUFFER', '0.3'))
    
    # Audio Device Configuration
    AUDIO_DEVICE_INDEX = os.getenv('AUDIO_DEVICE_INDEX')  # None for default
    # Porcupine Wake Word Detection
    PORCUPINE_ACCESS_KEY = os.getenv('PORCUPINE_ACCESS_KEY')
    PORCUPINE_KEYWORD_PATH = os.getenv('PORCUPINE_KEYWORD_PATH')  # Path to custom .ppn file
    PORCUPINE_SENSITIVITY = float(os.getenv('PORCUPINE_SENSITIVITY', '0.5'))