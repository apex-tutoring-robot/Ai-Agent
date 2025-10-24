"""
Continuous audio listener with Voice Activity Detection (VAD) for Raspberry Pi.
Detects when user starts and stops speaking.
"""

import pyaudio
import wave
import numpy as np
import tempfile
import os
import time
from typing import Optional, Callable
from collections import deque


class ContinuousListener:
    """Continuous audio listener with Voice Activity Detection."""
    
    def __init__(self, 
                 rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 format_type: int = pyaudio.paInt16,
                 silence_threshold: float = 0.015,
                 silence_duration: float = 2.0,
                 min_speech_duration: float = 0.5,
                 pre_speech_buffer: float = 0.3):
        """
        Initialize the continuous listener.
        
        Args:
            rate: Sample rate (16000 Hz is standard for speech)
            channels: Number of audio channels (1 for mono)
            chunk_size: Size of audio chunks to process
            format_type: PyAudio format type
            silence_threshold: RMS threshold below which audio is silent (0.01-0.03 typical)
            silence_duration: Seconds of silence before stopping recording
            min_speech_duration: Minimum speech duration to process (ignore short noises)
            pre_speech_buffer: Seconds of audio to capture before speech detected
        """
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format_type = format_type
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        
        # Calculate buffer sizes
        self.silence_chunks = int(silence_duration * rate / chunk_size)
        self.min_speech_chunks = int(min_speech_duration * rate / chunk_size)
        self.pre_buffer_chunks = int(pre_speech_buffer * rate / chunk_size)
        
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Circular buffer for pre-speech audio
        self.pre_buffer = deque(maxlen=self.pre_buffer_chunks)
        
        # Recording state
        self.is_recording = False
        self.frames = []
        self.silence_counter = 0
        self.speech_chunks = 0
        
    def list_audio_devices(self):
        """List all available audio input devices."""
        print("\n🎤 Available Audio Input Devices:")
        print("-" * 60)
        
        for i in range(self.pyaudio.get_device_count()):
            device_info = self.pyaudio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"Device {i}: {device_info['name']}")
                print(f"  Max Input Channels: {device_info['maxInputChannels']}")
                print(f"  Default Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
                print()
    
    def start_stream(self, device_index: Optional[int] = None):
        """
        Start the audio input stream.
        
        Args:
            device_index: Specific device index to use (None for default)
        """
        if self.stream and self.stream.is_active():
            self.stop_stream()
        
        try:
            self.stream = self.pyaudio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=None
            )
            print("✅ Audio stream started successfully")
            return True
        except Exception as e:
            print(f"❌ Error starting audio stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop the audio input stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
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
    
    def listen_for_speech(self, 
                         callback: Optional[Callable[[str], None]] = None,
                         timeout: Optional[float] = None) -> Optional[str]:
        """
        Listen continuously for speech and return audio file path when speech ends.
        
        Args:
            callback: Optional callback for status updates
            timeout: Optional timeout in seconds (None for infinite)
            
        Returns:
            Path to WAV file containing the speech, or None if timeout/error
        """
        if not self.stream or not self.stream.is_active():
            if callback:
                callback("Error: Audio stream not started")
            return None
        
        self.frames = []
        self.pre_buffer.clear()
        self.is_recording = False
        self.silence_counter = 0
        self.speech_chunks = 0
        
        start_time = time.time()
        
        if callback:
            callback("🎧 Listening for speech...")
        
        try:
            while True:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    if callback:
                        callback("Timeout reached")
                    return None
                
                # Read audio chunk
                try:
                    audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    continue
                
                # Calculate energy level
                rms = self.calculate_rms(audio_chunk)
                
                # Determine if this chunk has speech
                has_speech = rms > self.silence_threshold
                
                if not self.is_recording:
                    # Not recording yet - looking for speech to start
                    self.pre_buffer.append(audio_chunk)
                    
                    if has_speech:
                        # Speech detected! Start recording
                        self.is_recording = True
                        self.speech_chunks = 0
                        self.silence_counter = 0
                        
                        # Add pre-buffer to frames
                        self.frames.extend(list(self.pre_buffer))
                        self.frames.append(audio_chunk)
                        
                        if callback:
                            callback("🎤 Recording... Speak now!")
                        
                else:
                    # Currently recording
                    self.frames.append(audio_chunk)
                    self.speech_chunks += 1
                    
                    if has_speech:
                        # Still speaking - reset silence counter
                        self.silence_counter = 0
                    else:
                        # Silence detected
                        self.silence_counter += 1
                        
                        # Check if silence duration reached
                        if self.silence_counter >= self.silence_chunks:
                            # Check if we have minimum speech duration
                            if self.speech_chunks >= self.min_speech_chunks:
                                # Valid speech detected - save and return
                                if callback:
                                    callback("✅ Speech detected, processing...")
                                
                                return self._save_recording()
                            else:
                                # Too short - probably just noise
                                if callback:
                                    callback("⚠️ Speech too short, continuing to listen...")
                                
                                # Reset and continue listening
                                self.frames = []
                                self.is_recording = False
                                self.silence_counter = 0
                                self.speech_chunks = 0
                
        except KeyboardInterrupt:
            if callback:
                callback("Interrupted by user")
            return None
        except Exception as e:
            if callback:
                callback(f"Error: {e}")
            return None
    
    def _save_recording(self) -> str:
        """
        Save recorded frames to a temporary WAV file.
        
        Returns:
            Path to the saved WAV file
        """
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='chippy_speech_')
        os.close(fd)
        
        # Write WAV file
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format_type))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
        
        return temp_path
    
    def test_microphone(self, duration: int = 3):
        """
        Test microphone by recording for a specific duration and showing levels.
        
        Args:
            duration: Duration to test in seconds
        """
        if not self.stream or not self.stream.is_active():
            print("❌ Audio stream not started")
            return
        
        print(f"\n🎤 Testing microphone for {duration} seconds...")
        print("Speak into the microphone and watch the levels:")
        print("-" * 60)
        
        chunks_to_test = int(duration * self.rate / self.chunk_size)
        
        for i in range(chunks_to_test):
            try:
                audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                rms = self.calculate_rms(audio_chunk)
                
                # Visual representation
                bar_length = int(rms * 50)
                bar = "█" * bar_length
                status = "🔊 SPEECH" if rms > self.silence_threshold else "🔇 SILENCE"
                
                print(f"\r{status} | {bar:<50} | RMS: {rms:.4f}", end="", flush=True)
                
            except Exception as e:
                print(f"\nError: {e}")
                break
        
        print("\n" + "-" * 60)
        print(f"✅ Microphone test complete")
        print(f"Current silence threshold: {self.silence_threshold}")
        print(f"Tip: If levels are too low, speak louder or adjust threshold")
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_stream()
        self.pyaudio.terminate()