"""
Text-to-Speech client for CHIPPY.
Handles converting text responses to speech using Azure Cognitive Services.
"""

import os
import time
import requests
import tempfile
from typing import Optional

class TextToSpeechClient:
    """Client for Azure Text-to-Speech service using REST API for Pi compatibility."""
    
    # def __init__(self, config, voice_name="en-US-AriaNeural"):
    # def __init__(self, config, voice_name="en-US-SaraNeural"):
    # def __init__(self, config, voice_name="en-US-JennyNeural"):
    def __init__(self, config, voice_name="en-US-DavisNeural"):
        """
        Initialize the Text-to-Speech client.
        
        Args:
            config: Configuration object with Azure credentials
            voice_name: Name of the voice to use (default: en-US-AriaNeural)
        """
        # API endpoints
        self.region = config.SPEECH_REGION
        self.key = config.SPEECH_KEY
        self.token_url = f"https://{self.region}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
        self.tts_url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        
        # Voice settings
        self.voice_name = voice_name
        
        # Get access token
        self.access_token = self._get_token()
        self.token_expiry = time.time() + 540  # Tokens valid for ~10 minutes
    
    def _get_token(self) -> str:
        """Get authentication token for Speech service."""
        headers = {
            'Ocp-Apim-Subscription-Key': self.key
        }
        
        response = requests.post(self.token_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Token request failed with status code: {response.status_code}")
            
        return response.text
    
    def _ensure_valid_token(self) -> None:
        """Ensure we have a valid token, refreshing if necessary."""
        if time.time() > self.token_expiry:
            self.access_token = self._get_token()
            self.token_expiry = time.time() + 540
    
    def synthesize_speech(self, text: str, output_file: Optional[str] = None) -> str:
        """
        Synthesize speech from text and save to file.
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file (if None, creates a temp file)
            
        Returns:
            Path to the audio file
        """
        # Ensure token is valid
        self._ensure_valid_token()
        
        # Create temporary file if no output file is specified
        if output_file is None:
            fd, output_file = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
        
        # Set up headers
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'riff-24khz-16bit-mono-pcm',
            'User-Agent': 'CHIPPY-Educational-Bot'
        }
        
        # Create SSML document
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
            xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{self.voice_name}">
                <prosody rate="1.05" pitch="+10%">
                    <mstts:express-as style="cheerful" styledegree="1.5">
                        {text}
                    </mstts:express-as>
                </prosody>
            </voice>
        </speak>
        """
        
        # Make the request with exponential backoff
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.tts_url,
                    headers=headers,
                    data=ssml.encode('utf-8'),
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Write the audio data to file
                    with open(output_file, 'wb') as audio_file:
                        audio_file.write(response.content)
                    return output_file
                    
                elif response.status_code == 401:
                    # Token expired, refresh and retry
                    self.access_token = self._get_token()
                    self.token_expiry = time.time() + 540
                    continue
                    
                else:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise Exception(f"Speech synthesis failed with status code: {response.status_code}, {response.text}")
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e
        
        raise Exception("Speech synthesis failed after multiple attempts")
    
    # def play_speech(self, audio_file: str) -> None:
    #     """
    #     Play synthesized speech from file.
        
    #     Args:
    #         audio_file: Path to audio file
    #     """
    #     # Use platform-independent approach to play audio
    #     try:
    #         import platform
    #         system = platform.system()
            
    #         if system == "Windows":
    #             # Windows
    #             import winsound
    #             winsound.PlaySound(audio_file, winsound.SND_FILENAME)
                
    #         elif system == "Darwin":
    #             # macOS
    #             import subprocess
    #             subprocess.call(["afplay", audio_file])
                
    #         else:
    #             # Linux (including WSL and Raspberry Pi)
    #             import subprocess
                
    #             # First try aplay (ALSA - works on Pi and many Linux systems)
    #             try:
    #                 subprocess.call(["aplay", audio_file])
    #             except:
    #                 # Fall back to other players
    #                 players = ["paplay", "play"]
    #                 for player in players:
    #                     try:
    #                         subprocess.call([player, audio_file])
    #                         break
    #                     except:
    #                         continue
    #     except Exception as e:
    #         print(f"Error playing audio: {e}")
    #         print("Please install audio playback software appropriate for your system.")

    def play_speech(self, audio_file: str) -> None:
        """
        Play synthesized speech from file on Raspberry Pi.
        Uses ALSA (aplay) for direct hardware playback.
        
        Args:
            audio_file: Path to audio file
        """
        import subprocess
        import platform
        
        # Check if running on Windows/WSL (for development/testing)
        is_wsl = "microsoft" in platform.release().lower() or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
        
        if is_wsl:
            # WSL mode - keep existing Windows playback for testing
            try:
                import shutil
                windows_temp = subprocess.check_output(['wslpath', '-w', '/mnt/c/Windows/Temp']).decode('utf-8').strip()
                temp_filename = f"chippy_audio_{os.path.basename(audio_file)}"
                windows_audio_path = os.path.join(windows_temp, temp_filename)
                
                wsl_windows_temp = subprocess.check_output(['wslpath', '-u', windows_temp]).decode('utf-8').strip()
                wsl_temp_path = os.path.join(wsl_windows_temp, temp_filename)
                
                shutil.copy(audio_file, wsl_temp_path)
                cmd_command = f'cmd.exe /c start /wait "CHIPPY Audio" "{windows_audio_path}"'
                os.system(cmd_command)
                return
            except Exception as e:
                print(f"WSL playback failed: {e}, trying Linux methods...")
        
        # Raspberry Pi / Linux mode - Use ALSA (aplay)
        try:
            # Try aplay first (standard on Pi)
            result = subprocess.run(
                ["aplay", "-q", audio_file],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            
            if result.returncode == 0:
                return
            else:
                print(f"aplay failed: {result.stderr.decode()}")
                
                # Try alternative players
                for player in ["paplay", "play"]:
                    try:
                        result = subprocess.run([player, audio_file], stderr=subprocess.PIPE)
                        if result.returncode == 0:
                            return
                    except FileNotFoundError:
                        continue
                
                print("❌ No audio player found. Install alsa-utils: sudo apt-get install alsa-utils")
                
        except Exception as e:
            print(f"❌ Error playing audio: {e}")