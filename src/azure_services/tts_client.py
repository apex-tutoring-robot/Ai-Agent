"""
Azure Text-to-Speech client with streaming support.
Converts text to speech with sentence-based streaming for minimal latency.
"""

import os
import logging
import re
from typing import Iterator, Optional
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class TextToSpeechClient:
    """Azure Text-to-Speech client with streaming support."""
    
    def __init__(
        self,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        voice: Optional[str] = None,
        speech_rate: float = None
    ):
        """
        Initialize Text-to-Speech client.
        
        Args:
            speech_key: Azure Speech API key
            speech_region: Azure Speech region
            voice: Voice name (e.g., 'en-US-JennyNeural')
            speech_rate: Speech rate multiplier (1.0 is normal)
        """
        self.speech_key = speech_key or os.getenv('AZURE_SPEECH_KEY')
        self.speech_region = speech_region or os.getenv('AZURE_SPEECH_REGION')
        self.voice = voice or os.getenv('TTS_VOICE', 'en-US-JennyNeural')
        self.speech_rate = speech_rate or float(os.getenv('TTS_SPEECH_RATE', 1.0))
        
        if not self.speech_key or not self.speech_region:
            raise ValueError("Azure Speech credentials not provided")
        
        # Configure speech
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        self.speech_config.speech_synthesis_voice_name = self.voice
        
        # Set speech rate if not default
        if self.speech_rate != 1.0:
            rate_percent = int((self.speech_rate - 1.0) * 100)
            self.speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
            )
        else:
            self.speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
            )
        
        logger.info(f"TTS client initialized with voice: {self.voice}")
    
    def synthesize_to_audio(self, text: str) -> bytes:
        """
        Synthesize text to audio (non-streaming).
        
        Args:
            text: Text to synthesize
        
        Returns:
            Raw audio bytes (PCM 16-bit, 16kHz, mono)
        """
        try:
            # Create synthesizer with null output (we'll get audio from result)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )
            
            logger.info(f"Synthesizing: {text[:50]}...")
            result = synthesizer.speak_text(text)
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Speech synthesis completed")
                return result.audio_data
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(f"Speech synthesis canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {cancellation.error_details}")
                return b''
            
            return b''
        
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            raise
    
    def synthesize_stream(self, text_stream: Iterator[str]) -> Iterator[bytes]:
        """
        Synthesize streaming text to audio chunks.
        Buffers text until sentence boundaries are detected for natural speech.
        
        Args:
            text_stream: Iterator yielding text chunks
        
        Yields:
            Audio data chunks
        """
        sentence_buffer = ""
        
        # Sentence boundary patterns
        sentence_endings = re.compile(r'[.!?]\s+|[.!?]$')
        
        try:
            for text_chunk in text_stream:
                sentence_buffer += text_chunk
                
                # Check for sentence boundaries
                match = sentence_endings.search(sentence_buffer)
                
                if match:
                    # Extract complete sentence(s)
                    end_pos = match.end()
                    complete_text = sentence_buffer[:end_pos].strip()
                    sentence_buffer = sentence_buffer[end_pos:]
                    
                    if complete_text:
                        # Synthesize the complete sentence(s)
                        audio_data = self.synthesize_to_audio(complete_text)
                        if audio_data:
                            yield audio_data
            
            # Synthesize any remaining text
            if sentence_buffer.strip():
                logger.info(f"📝 Synthesizing remaining text buffer ({len(sentence_buffer)} chars): '{sentence_buffer.strip()[:100]}...'")
                audio_data = self.synthesize_to_audio(sentence_buffer.strip())
                if audio_data:
                    logger.info(f"✓ Final buffer synthesized: {len(audio_data)} bytes")
                    yield audio_data
                else:
                    logger.warning("⚠️  Final buffer synthesis returned no audio!")
            else:
                logger.info("✓ No remaining text in buffer (all sentences complete)")
        
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            if sentence_buffer:
                logger.error(f"Lost text in buffer: '{sentence_buffer}'")
            raise
    
    def synthesize_sentences(self, sentences: list[str]) -> Iterator[bytes]:
        """
        Synthesize a list of sentences to audio chunks.
        
        Args:
            sentences: List of sentences to synthesize
        
        Yields:
            Audio data chunks
        """
        for sentence in sentences:
            if sentence.strip():
                audio_data = self.synthesize_to_audio(sentence.strip())
                if audio_data:
                    yield audio_data
                    
    def cleanup(self):
        """Cleanup TTS resources (safe no-op for Azure SDK)."""
        try:
            if hasattr(self, "audio_player"):
                self.audio_player.stop_streaming()
        except Exception:
            pass



if __name__ == "__main__":
    # Test TTS client
    print("Testing Azure Text-to-Speech client...\n")
    
    try:
        from playback import AudioPlayer
        
        client = TextToSpeechClient()
        player = AudioPlayer()
        
        # Test non-streaming synthesis
        test_text = "Hello! I am Jarvis, your AI tutoring assistant. How can I help you today?"
        print(f"Synthesizing: {test_text}\n")
        
        audio_data = client.synthesize_to_audio(test_text)
        
        if audio_data:
            print(f"Generated {len(audio_data)} bytes of audio")
            print("Playing audio...")
            player.play_audio(audio_data)
            print("Playback complete!")
        
        player.cleanup()
    
    except Exception as e:
        print(f"Error: {e}")
