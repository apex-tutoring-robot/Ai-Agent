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

_SENTENCE_ENDINGS = re.compile(r'[.!?]\s+|[.!?]$')


class SynthesisBlockedError(Exception):
    """Raised by synthesize_stream when pre_synthesis_check rejects a sentence."""
    def __init__(self, refusal_text: str):
        self.refusal_text = refusal_text
        super().__init__(refusal_text)


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
        
        # Configure speech config
        # Using standard REST endpoint - phrase buffering + audio streaming for optimization  
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        logger.info(f"TTS client configured for region: {self.speech_region}")
        self.speech_config.speech_synthesis_voice_name = self.voice
        
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        )
        
        logger.info(f"TTS client initialized with voice: {self.voice}")
        
        # Create persistent synthesizer instance (reused for all synthesis calls)
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None  # We'll get audio from result objects
        )
        logger.info("Persistent synthesizer instance created")
        
        # Warm-start the TTS service using the persistent synthesizer
        self.warm_up_v2()
    
    def warm_up_v2(self):
        """
        Warm-start the Azure TTS service by pre-opening the connection.
        This reduces first-call latency without wasting synthesis quota.
        """
        try:
            logger.info("Opening persistent connection to Azure TTS...")
            
            # Get connection from synthesizer and open it
            self.connection = speechsdk.Connection.from_speech_synthesizer(self.synthesizer)
            self.connection.open(True)  # True = wait for connection to establish
            
            logger.info("Azure TTS connection established")
        except Exception as e:
            logger.warning(f"TTS connection pre-open failed (non-critical): {e}")
        
    def warm_up_v1(self):
        """
        Warm-start the Azure TTS service to reduce first-call latency.
        Performs a dummy synthesis using the persistent synthesizer to pre-load models.
        """
        try:
            logger.info("Warming up Azure TTS service...")
            
            # Use persistent synthesizer for warm-up (no need for temporary instance)
            result = self.synthesizer.speak_text_async("Warm-up request.").get()
            
            logger.info("Azure TTS warm-up complete")
        except Exception as e:
            logger.warning(f"TTS warm-up failed (non-critical): {e}")
    
    def _build_ssml(self, text: str) -> str:
        rate_percent = int((self.speech_rate - 1.0) * 100)
        sign = "+" if rate_percent >= 0 else ""
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            f'<voice name="{self.voice}">'
            f'<prosody rate="{sign}{rate_percent}%">{text}</prosody>'
            f'</voice></speak>'
        )

    @staticmethod
    def _strip_latex(text: str) -> str:
        """Remove LaTeX math notation so Azure TTS doesn't read backslashes aloud."""
        # Strip inline/display math delimiters, keep the content inside
        text = re.sub(r'\\\(|\\\)', '', text)
        text = re.sub(r'\\\[|\\\]', '', text)
        # Convert common math commands to spoken words
        text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1 over \2', text)
        text = re.sub(r'\\times', ' times ', text)
        text = re.sub(r'\\div', ' divided by ', text)
        text = re.sub(r'\\cdot', ' times ', text)
        text = re.sub(r'\\sqrt\{([^}]*)\}', r'square root of \1', text)
        # Strip any remaining LaTeX commands (backslash + word)
        text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
        # Remove stray curly braces left over from LaTeX
        text = re.sub(r'[{}]', '', text)
        # Collapse extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def synthesize_to_audio(self, text: str) -> bytes:
        """
        Synthesize text to audio (non-streaming).
        
        Args:
            text: Text to synthesize
        
        Returns:
            Raw audio bytes (PCM 16-bit, 16kHz, mono)
        """
        try:
            # Reuse persistent synthesizer instance (no initialization overhead)
            text = self._strip_latex(text)
            logger.info(f"Synthesizing: {text[:50]}...")
            if self.speech_rate != 1.0:
                result = self.synthesizer.speak_ssml(self._build_ssml(text))
            else:
                result = self.synthesizer.speak_text(text)
            
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

    def synthesize_stream(self, text_stream: Iterator[str], pre_synthesis_check=None) -> Iterator[bytes]:
        """
        Synthesize streaming text to audio chunks sentence by sentence.

        Args:
            text_stream: Iterator yielding text chunks (e.g. LLM token stream)
            pre_synthesis_check: Optional callable(sentence) -> (bool, str).
                Called before each sentence is synthesized. If it returns
                (False, refusal_text), raises SynthesisBlockedError immediately.

        Yields:
            Audio data chunks (one per sentence)
        """
        sentence_buffer = ""

        try:
            for text_chunk in text_stream:
                sentence_buffer += text_chunk

                match = _SENTENCE_ENDINGS.search(sentence_buffer)

                if match:
                    end_pos = match.end()
                    complete_text = sentence_buffer[:end_pos].strip()
                    sentence_buffer = sentence_buffer[end_pos:]

                    if complete_text:
                        if pre_synthesis_check:
                            safe, refused = pre_synthesis_check(complete_text)
                            if not safe:
                                raise SynthesisBlockedError(refused)
                        audio_data = self.synthesize_to_audio(complete_text)
                        if audio_data:
                            yield audio_data

            if sentence_buffer.strip():
                if pre_synthesis_check:
                    safe, refused = pre_synthesis_check(sentence_buffer.strip())
                    if not safe:
                        raise SynthesisBlockedError(refused)
                audio_data = self.synthesize_to_audio(sentence_buffer.strip())
                if audio_data:
                    yield audio_data

        except SynthesisBlockedError:
            raise
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            if sentence_buffer:
                logger.error(f"Lost text in buffer: '{sentence_buffer}'")
            raise
    
    def _synthesize_with_audio_streaming(self, text: str) -> Iterator[bytes]:
        """
        Synthesize text and stream audio in chunks for lower latency.
        Uses AudioDataStream to get audio chunks as they're generated.
        
        Args:
            text: Text to synthesize
        
        Yields:
            Audio data chunks
        """
        try:
            logger.debug(f"Synthesizing phrase: '{text[:30]}...'")
            
            # Use speak_text_async to get synthesis result
            result = self.synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Speech synthesis completed")
                yield result.audio_data
                    
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(f"Synthesis canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {cancellation.error_details}")
        
        except Exception as e:
            logger.error(f"Error in audio streaming synthesis: {e}")
            raise
    
    def cleanup(self):
        """
        Clean up resources. Call this when done using the TTS client.
        """
        if hasattr(self, 'synthesizer') and self.synthesizer:
            logger.info("Cleaning up TTS synthesizer...")
            self.synthesizer = None


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
