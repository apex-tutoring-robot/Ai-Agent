"""
Audio conversion utilities for CHIPPY.
Handles converting various audio formats to compatible WAV files.
"""

import os
import subprocess
import tempfile
from pathlib import Path

# No need to import Config if not used in this module
# If you do need Config, use: from ..config import Config

class AudioConverter:
    """Helper class for audio format conversions."""
    
    @staticmethod
    def convert_to_wav(input_file: str, output_file: str = None) -> str:
        """
        Convert an audio file to WAV format compatible with Azure Speech Services.
        
        Args:
            input_file: Path to the input audio file
            output_file: Path for the output WAV file (optional)
            
        Returns:
            Path to the converted WAV file
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input audio file not found: {input_file}")
            
        # If no output file specified, create one in temp directory
        if output_file is None:
            temp_dir = tempfile.gettempdir()
            input_filename = os.path.basename(input_file)
            base_name = os.path.splitext(input_filename)[0]
            output_file = os.path.join(temp_dir, f"{base_name}_converted.wav")
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert using ffmpeg
        try:
            # Convert to WAV with proper format for Azure Speech SDK:
            # - 16-bit PCM
            # - 16kHz sample rate (standard for speech recognition)
            # - Mono channel
            command = [
                'ffmpeg',
                '-y',  # Overwrite output file if exists
                '-i', input_file,
                '-acodec', 'pcm_s16le',  # 16-bit PCM encoding
                '-ar', '16000',          # 16kHz sample rate
                '-ac', '1',              # Mono channel
                output_file
            ]
            
            # Run ffmpeg command
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Check if conversion was successful
            if process.returncode != 0:
                raise RuntimeError(f"Audio conversion failed: {process.stderr}")
                
            print(f"Successfully converted audio to WAV format: {output_file}")
            return output_file
            
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg to convert audio files.\n"
                "In WSL: sudo apt-get update && sudo apt-get install -y ffmpeg"
            )