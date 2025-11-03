"""
Text-to-Speech client for CHIPPY.
Handles converting text responses to speech using Azure Cognitive Services.
Now with interruptible playback for natural conversation flow.
Includes fixes for ALSA underruns and audio feedback rejection.
"""

import os
import time
import requests
import tempfile
import threading
import numpy as np
import pyaudio
import wave
from typing import Optional, Callable

class TextToSpeechClient:
    """Client for Azure Text-to-Speech service using REST API for Pi compatibility."""
    
    def __init__(self, config, voice_name="en-US-DavisNeural"):
        """
        Initialize the Text-to-Speech client.
        
        Args:
            config: Configuration object with Azure credentials
            voice_name: Name of the voice to use (default: en-US-DavisNeural)
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
        
        # Interrupt detection settings
        self.interrupt_threshold = float(os.getenv("INTERRUPT_SENSITIVITY", "0.020"))
        self.min_playback_time = float(os.getenv("MIN_PLAYBACK_TIME", "1.0"))
    
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
    
    def calculate_rms(self, audio_chunk: bytes) -> float:
        """
        Calculate RMS (Root Mean Square) energy of audio chunk.
        
        Args:
            audio_chunk: Raw audio bytes
            
        Returns:
            Normalized RMS value (0.0 to 1.0)
        """
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        normalized_rms = rms / 32768.0  # Normalize to 0-1 range
        return normalized_rms
    
    def play_speech_interruptible(self, 
                                  audio_file: str, 
                                  interrupt_check: Optional[Callable[[], bool]] = None,
                                  device_index: Optional[int] = None) -> dict:
        """
        Play synthesized speech with interrupt detection.
        Monitors microphone and stops playback if speech is detected.
        
        Args:
            audio_file: Path to audio file
            interrupt_check: Optional callback that returns True if interrupted
            device_index: Input device index for interrupt detection
            
        Returns:
            dict with 'interrupted': bool, 'played_duration': float
        """
        import subprocess
        import platform
        
        # Check if running on Windows/WSL (for development/testing)
        is_wsl = "microsoft" in platform.release().lower() or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
        
        if is_wsl:
            # WSL mode - use Windows playback (no interrupt detection)
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
                return {'interrupted': False, 'played_duration': 0.0}
            except Exception as e:
                print(f"WSL playback failed: {e}")
                return {'interrupted': False, 'played_duration': 0.0}
        
        # Raspberry Pi / Linux mode - Interruptible playback
        return self._play_with_interrupt_detection(audio_file, interrupt_check, device_index)
    
    def _play_with_interrupt_detection(self, 
                                      audio_file: str,
                                      interrupt_check: Optional[Callable[[], bool]] = None,
                                      device_index: Optional[int] = None) -> dict:
        """
        Play audio with real-time interrupt detection via microphone monitoring.
        Fixed for ALSA underruns and audio feedback rejection.
        
        Args:
            audio_file: Path to WAV file to play
            interrupt_check: Optional external interrupt check function
            device_index: Input device for microphone
            
        Returns:
            dict with 'interrupted': bool, 'played_duration': float
        """
        interrupted = False
        played_duration = 0.0
        start_time = time.time()
        
        # Shared flag for interrupt detection
        interrupt_flag = threading.Event()
        
        # Open audio file
        try:
            wf = wave.open(audio_file, 'rb')
        except Exception as e:
            print(f"❌ Error opening audio file: {e}")
            return {'interrupted': False, 'played_duration': 0.0}
        
        # Get audio file parameters
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open output stream for playback with larger buffer to prevent underruns
        try:
            output_stream = p.open(
                format=p.get_format_from_width(sample_width),
                channels=channels,
                rate=sample_rate,
                output=True,
                frames_per_buffer=2048  # Larger buffer to prevent underruns
            )
        except Exception as e:
            print(f"❌ Error opening output stream: {e}")
            wf.close()
            p.terminate()
            return {'interrupted': False, 'played_duration': 0.0}
        
        # Open input stream for interrupt detection
        input_stream = None
        monitor_thread = None
        
        try:
            input_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=2048  # Larger buffer
            )
            
            # Start monitoring thread
            def monitor_microphone():
                """Monitor microphone for speech during playback."""
                # Wait for minimum playback time + extra buffer to avoid audio feedback
                # This prevents the microphone from hearing the robot's own voice
                time.sleep(self.min_playback_time + 0.3)
                
                # Use higher threshold to reject audio feedback from speaker
                # Real human speech from close range will still be detected
                feedback_rejection_multiplier = 2.5
                interrupt_threshold = self.interrupt_threshold * feedback_rejection_multiplier
                
                # Require multiple consecutive chunks of speech to confirm real interrupt
                # This filters out brief noise spikes and echo
                consecutive_speech_chunks = 0
                required_consecutive = 3  # Need 3 consecutive chunks (~150ms) to confirm
                
                while not interrupt_flag.is_set():
                    try:
                        # Read from microphone
                        audio_chunk = input_stream.read(2048, exception_on_overflow=False)
                        rms = self.calculate_rms(audio_chunk)
                        
                        # Check if speech detected (with higher threshold)
                        if rms > interrupt_threshold:
                            consecutive_speech_chunks += 1
                            if consecutive_speech_chunks >= required_consecutive:
                                print(f"\n⚠️  Interrupt detected! (RMS: {rms:.4f}) Stopping playback...")
                                interrupt_flag.set()
                                break
                        else:
                            # Reset counter if silence detected
                            consecutive_speech_chunks = 0
                        
                        # Check external interrupt callback
                        if interrupt_check and interrupt_check():
                            interrupt_flag.set()
                            break
                            
                    except Exception as e:
                        # Handle any audio read errors
                        break
            
            monitor_thread = threading.Thread(target=monitor_microphone, daemon=True)
            monitor_thread.start()
            
        except Exception as e:
            print(f"⚠️  Could not start interrupt detection: {e}")
            print("Playing without interrupt detection...")
        
        # Play audio in chunks with proper buffer handling
        chunk_size = 2048  # Match buffer size to prevent underruns
        data = wf.readframes(chunk_size)
        
        while data and not interrupt_flag.is_set():
            try:
                output_stream.write(data)
                data = wf.readframes(chunk_size)
                played_duration = time.time() - start_time
            except Exception as e:
                # Handle any playback errors gracefully
                print(f"⚠️  Playback error: {e}")
                break
        
        interrupted = interrupt_flag.is_set()
        
        # Cleanup streams
        try:
            output_stream.stop_stream()
            output_stream.close()
        except:
            pass
        
        if input_stream:
            try:
                input_stream.stop_stream()
                input_stream.close()
            except:
                pass
        
        wf.close()
        p.terminate()
        
        if interrupted:
            print(f"🛑 Playback interrupted after {played_duration:.2f}s")
        
        return {'interrupted': interrupted, 'played_duration': played_duration}
    
    def play_speech(self, audio_file: str) -> None:
        """
        Play synthesized speech from file (non-interruptible, legacy method).
        
        Args:
            audio_file: Path to audio file
        """
        result = self.play_speech_interruptible(audio_file, interrupt_check=None)
        # Legacy method doesn't return anything