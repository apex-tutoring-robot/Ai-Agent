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