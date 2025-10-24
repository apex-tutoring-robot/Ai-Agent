"""
Wake word detector using Picovoice Porcupine.
Detects "Hello CHIPPY" wake word locally on Raspberry Pi.
"""

import struct
import pyaudio
import pvporcupine
from typing import Optional, Callable
import os


class WakeWordDetector:
    """Wake word detector using Porcupine."""
    
    def __init__(self, 
                access_key: str,
                keyword_paths: Optional[list] = None,
                keywords: Optional[list] = None,
                sensitivities: Optional[list] = None,
                device_index: Optional[int] = None):
        """
        Initialize wake word detector.
        
        Args:
            access_key: Picovoice access key
            keyword_paths: List of paths to custom .ppn files
            keywords: List of built-in keywords (e.g., ['picovoice', 'porcupine'])
            sensitivities: Sensitivity for each keyword (0.0 to 1.0, default 0.5)
            device_index: Audio device index (None for default)
        """
        self.access_key = access_key
        self.device_index = device_index
        
        # Initialize Porcupine
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=keyword_paths,
                keywords=keywords,
                sensitivities=sensitivities or [0.5] * (len(keyword_paths or keywords or []))
            )
            
            print(f"✅ Wake word detector initialized")
            print(f"   Sample rate: {self.porcupine.sample_rate} Hz")
            print(f"   Frame length: {self.porcupine.frame_length} samples")
            
        except Exception as e:
            print(f"❌ Failed to initialize Porcupine: {e}")
            raise
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def start(self):
        """Start the audio stream for wake word detection."""
        if self.stream and self.stream.is_active():
            return
        
        try:
            self.stream = self.audio.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.porcupine.frame_length
            )
            print("✅ Wake word audio stream started")
        except Exception as e:
            print(f"❌ Failed to start audio stream: {e}")
            raise
    
    def stop(self):
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def listen(self, callback: Optional[Callable[[str], None]] = None) -> int:
        """
        Listen for wake word (blocking call).
        
        Args:
            callback: Optional callback for status updates
            
        Returns:
            Index of detected keyword (-1 if no keyword)
        """
        if not self.stream or not self.stream.is_active():
            raise RuntimeError("Audio stream not started. Call start() first.")
        
        try:
            while True:
                # Read audio frame
                pcm = self.stream.read(
                    self.porcupine.frame_length,
                    exception_on_overflow=False
                )
                
                # Convert to 16-bit integers
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                # Process with Porcupine
                keyword_index = self.porcupine.process(pcm)
                
                # Check if wake word detected
                if keyword_index >= 0:
                    if callback:
                        callback(f"Wake word detected! (index: {keyword_index})")
                    return keyword_index
                    
        except KeyboardInterrupt:
            return -1
        except Exception as e:
            print(f"❌ Error during wake word detection: {e}")
            return -1
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self.porcupine:
            self.porcupine.delete()
        if self.audio:
            self.audio.terminate()
    
    def test(self, duration: int = 10):
        """
        Test wake word detection for a specific duration.
        
        Args:
            duration: Test duration in seconds
        """
        import time
        
        print(f"\n🎤 Testing wake word detection for {duration} seconds...")
        print("Say your wake word!")
        print("-" * 60)
        
        self.start()
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration:
                # Read audio frame
                pcm = self.stream.read(
                    self.porcupine.frame_length,
                    exception_on_overflow=False
                )
                
                # Convert to 16-bit integers
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                # Process with Porcupine
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    print(f"\n🎉 WAKE WORD DETECTED! (index: {keyword_index})")
                    elapsed = time.time() - start_time
                    print(f"   Time: {elapsed:.2f}s")
                    print("\nContinuing to listen...")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Test stopped")
        
        self.stop()
        print("-" * 60)
        print("✅ Test complete!")