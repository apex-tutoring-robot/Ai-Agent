"""
Audio helper utilities for CHIPPY.
Handles audio input/output and processing.
"""

import wave
import pyaudio
import numpy as np
from typing import Optional, Tuple, List

class AudioHelper:
    """Helper class for audio operations."""
    
    def __init__(self):
        """Initialize the audio helper."""
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
    
    def start_recording(self, 
                        rate: int = 16000, 
                        channels: int = 1,
                        chunk_size: int = 1024,
                        format_type: int = pyaudio.paInt16) -> None:
        """
        Start recording audio from the microphone.
        
        Args:
            rate: Sample rate
            channels: Number of audio channels
            chunk_size: Size of audio chunks to process
            format_type: PyAudio format type
        """
        if self.stream and self.stream.is_active():
            self.stop_recording()
            
        self.stream = self.pyaudio.open(
            format=format_type,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        self.audio_format = format_type
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size
        self.frames = []
    
    def record_chunk(self) -> bytes:
        """
        Record a chunk of audio data.
        
        Returns:
            Audio data bytes
        """
        if not self.stream or not self.stream.is_active():
            raise RuntimeError("Recording has not been started")
            
        data = self.stream.read(self.chunk_size)
        self.frames.append(data)
        return data
    
    def stop_recording(self) -> None:
        """Stop recording audio."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def save_recording(self, filename: str) -> None:
        """
        Save the recorded audio to a WAV file.
        
        Args:
            filename: Path to save the WAV file
        """
        if not self.frames:
            raise ValueError("No audio data to save")
        
        # Create WAV file with recorded frames
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            
        # Log success
        print(f"Audio saved to {filename}")
    
    def detect_silence(self, 
                      threshold: float = 0.03, 
                      min_silence_duration: float = 4.0) -> bool:
        """
        Detect if there is silence in the audio stream.
        Useful for Voice Activity Detection (VAD).
        
        Args:
            threshold: RMS threshold below which audio is considered silent
            min_silence_duration: Minimum duration of silence in seconds
            
        Returns:
            True if silence is detected for the specified duration
        """
        if not self.stream:
            raise RuntimeError("Recording has not been started")
        
        # Number of chunks to check for silence
        chunks_to_check = int(min_silence_duration * self.rate / self.chunk_size)
        silence_counter = 0
        
        for _ in range(chunks_to_check):
            data = self.stream.read(self.chunk_size)
            # Convert to numpy array for RMS calculation
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Calculate RMS (loudness)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            # Normalize RMS (0-1 range)
            normalized_rms = rms / 32768.0
            
            if normalized_rms < threshold:
                silence_counter += 1
            else:
                # Reset counter if noise detected
                silence_counter = 0
                
        # If all chunks were silent, return True
        return silence_counter >= chunks_to_check
    
    def cleanup(self) -> None:
        """Clean up PyAudio resources."""
        self.stop_recording()
        self.pyaudio.terminate()
        
    def get_audio_devices(self) -> List[dict]:
        """
        Get a list of available audio input devices.
        
        Returns:
            List of dictionaries with device information
        """
        devices = []
        
        for i in range(self.pyaudio.get_device_count()):
            device_info = self.pyaudio.get_device_info_by_index(i)
            # Only include input devices
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                })
                
        return devices