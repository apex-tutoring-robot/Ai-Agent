"""
Azure Text-to-Speech client with streaming support.
Converts text to speech with sentence-based streaming for minimal latency.
"""

import os
import logging
import re
import xml.sax.saxutils as saxutils
from typing import Iterator, Optional
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Defensive cleanup for markdown/LaTeX artifacts that occasionally slip
# into an LLM response despite the "no formatting" instructions in every
# prompt - confirmed live: a generic chat response about circle area
# came back with literal LaTeX ("\[ \text{Area} = \pi r^2 \]"), which
# would read as gibberish through TTS instead of plain spoken math. Not
# a complete LaTeX-to-English converter - just the small set of
# constructs actually seen in practice. Applied once, right before
# synthesis in synthesize_stream (see below), so it catches text from
# ANY caller - generic chat, teaching-plan steps, system messages - not
# just one code path, and only ever sees a complete buffered sentence
# (never a partial streamed token), so a delimiter split across chunks
# is never a concern.
_LATEX_FRAC = re.compile(r'\\frac\{([^{}]+)\}\{([^{}]+)\}')
_LATEX_TEXT_CMD = re.compile(r'\\text\{([^{}]*)\}')
_LATEX_DELIMITERS = re.compile(r'\\[\[\]()]')
_LATEX_SYMBOLS = (
    (r'\times', 'times'), (r'\div', 'divided by'), (r'\cdot', 'times'), (r'\pi', 'pi'),
)
_LATEX_DOUBLE_DOLLAR = re.compile(r'\$\$')
# Exponents: specific cases first (so the generic pattern below doesn't
# also match - and re-mangle - the ^2/^3 it already consumed).
_EXPONENT_2 = re.compile(r'(\w)\^2\b')
_EXPONENT_3 = re.compile(r'(\w)\^3\b')
_EXPONENT_N = re.compile(r'(\w)\^(\d+)\b')
_MARKDOWN_BOLD = re.compile(r'\*\*([^*]+)\*\*')
_MARKDOWN_BULLET = re.compile(r'(?m)^\s*[-*]\s+')


def _strip_math_notation(text: str) -> str:
    text = _LATEX_FRAC.sub(r'\1 over \2', text)
    text = _LATEX_TEXT_CMD.sub(r'\1', text)
    for symbol, word in _LATEX_SYMBOLS:
        text = text.replace(symbol, word)
    text = _LATEX_DELIMITERS.sub('', text)
    text = _LATEX_DOUBLE_DOLLAR.sub('', text)
    text = _EXPONENT_2.sub(r'\1 squared', text)
    text = _EXPONENT_3.sub(r'\1 cubed', text)
    text = _EXPONENT_N.sub(r'\1 to the power of \2', text)
    text = _MARKDOWN_BOLD.sub(r'\1', text)
    text = _MARKDOWN_BULLET.sub('', text)
    # Removing delimiters (e.g. "\[ \pi r^2 \]" -> " pi r squared ") can
    # leave behind extra/leading/trailing whitespace - collapse and trim
    # it so this function always returns clean text on its own, whatever
    # surrounded the parts it stripped.
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
        # Azure neural voices support an mstts:express-as "style" - "chat"
        # is Microsoft's own description for "a user's queries/comments in
        # a casual and relaxed tone", which is exactly the gap found live:
        # most of what Jarvis actually says went through plain speak_text()
        # with no SSML at all (see synthesize_to_audio), so it never got
        # ANY prosody/style shaping - completely flat, default delivery,
        # not just unstyled. An unsupported style name on a given voice
        # doesn't error - Azure's SSML parser just ignores it and
        # synthesizes with default delivery instead (confirmed live), so
        # this is safe to leave on even if TTS_VOICE is swapped to a voice
        # that doesn't support "chat".
        self.style = os.getenv('TTS_STYLE', 'chat')
        
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
        # speech_rate itself is applied per-call via SSML <prosody> in
        # synthesize_to_audio() - speak_text()/speak_text_async() only
        # accept plain text, there's no rate parameter on those calls.
        # (Previously this computed rate_percent here and never used it -
        # TTS_SPEECH_RATE silently had zero effect on anything.)
        
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
    
    def _build_ssml(self, text: str, rate_percent: Optional[int] = None, pitch_percent: Optional[int] = None) -> str:
        """
        Wrap text in SSML with the conversational mstts:express-as style
        (self.style, see __init__) plus <prosody rate="..." pitch="...">
        adjustments. speak_text()/speak_text_async() only accept plain
        text - there's no style/rate/pitch parameter on those calls, so
        every call now goes through this SSML path instead (see
        synthesize_to_audio - previously only delivery/non-default-rate
        calls used SSML, meaning most actual speech got neither a style
        NOR any prosody shaping at all). text is XML-escaped since it's
        LLM-generated and may contain '&', '<', '>', etc.

        rate_percent/pitch_percent, when given, override the instance's
        global self.speech_rate for this one call - see synthesize_to_audio's
        `delivery` parameter.
        """
        if rate_percent is None:
            rate_percent = int(round((self.speech_rate - 1.0) * 100))
        rate_sign = "+" if rate_percent >= 0 else ""
        pitch_percent = pitch_percent or 0
        pitch_sign = "+" if pitch_percent >= 0 else ""
        escaped = saxutils.escape(text)
        prosody = f'<prosody rate="{rate_sign}{rate_percent}%" pitch="{pitch_sign}{pitch_percent}%">{escaped}</prosody>'
        styled = f'<mstts:express-as style="{self.style}">{prosody}</mstts:express-as>' if self.style else prosody
        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">'
            f'<voice name="{self.voice}">{styled}</voice></speak>'
        )

    def synthesize_to_audio(self, text: str, delivery: Optional["DeliveryStyle"] = None) -> bytes:
        """
        Synthesize text to audio (non-streaming).

        Args:
            text: Text to synthesize
            delivery: Optional per-utterance ExpressionController DeliveryStyle
                (rate/pitch nudge) - overrides the instance's global
                speech_rate for this call only. See expression/controller.py.

        Returns:
            Raw audio bytes (PCM 16-bit, 16kHz, mono)
        """
        try:
            # Reuse persistent synthesizer instance (no initialization overhead)
            logger.info(f"Synthesizing: {text[:50]}...")
            # Always through SSML now, not just when delivery/rate differ
            # from default - see _build_ssml's docstring for why
            # (speak_text() carries no style at all, which was the actual
            # gap behind "the voice sounds flat/robotic").
            if delivery is not None:
                ssml = self._build_ssml(text, rate_percent=delivery.rate_percent, pitch_percent=delivery.pitch_percent)
            else:
                ssml = self._build_ssml(text)
            result = self.synthesizer.speak_ssml(ssml)

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

    def synthesize_stream(self, text_stream: Iterator[str], delivery: Optional["DeliveryStyle"] = None) -> Iterator[bytes]:
        """
        Synthesize streaming text to audio chunks.
        Buffers text until sentence boundaries are detected for natural speech.

        Args:
            text_stream: Iterator yielding text chunks
            delivery: Optional per-utterance ExpressionController DeliveryStyle,
                forwarded to every synthesize_to_audio() call below - see that
                method's docstring.

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
                        audio_data = self.synthesize_to_audio(_strip_math_notation(complete_text), delivery=delivery)
                        if audio_data:
                            yield audio_data
            
            # Synthesize any remaining text
            if sentence_buffer.strip():
                logger.info(f"📝 Synthesizing remaining text buffer ({len(sentence_buffer)} chars): '{sentence_buffer.strip()[:100]}...'")
                audio_data = self.synthesize_to_audio(_strip_math_notation(sentence_buffer.strip()), delivery=delivery)
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
    
    def synthesize_stream_old(self, text_stream: Iterator[str]) -> Iterator[bytes]:
        """
        Synthesize streaming text to audio chunks with optimized WebSocket V2 streaming.
        Uses phrase-level buffering (smaller chunks than full sentences) for lower latency.
        Streams audio at byte-level for immediate playback.
        
        Args:
            text_stream: Iterator yielding text chunks from LLM
        
        Yields:
            Audio data chunks (smaller chunks for lower latency)
        """
        phrase_buffer = ""
        
        # Phrase boundary patterns - more frequent than sentences for lower latency
        # Triggers on: sentence end, comma, semicolon, or after ~50 chars
        phrase_endings = re.compile(r'[.!?;:]\s+|[.!?]$')
        
        try:
            for text_chunk in text_stream:
                phrase_buffer += text_chunk
                
                # Check for phrase boundaries OR if buffer is getting long
                match = phrase_endings.search(phrase_buffer)
                should_synthesize = match or len(phrase_buffer) > 50
                
                if should_synthesize:
                    # Extract complete phrase(s)
                    if match:
                        end_pos = match.end()
                        complete_text = phrase_buffer[:end_pos].strip()
                        phrase_buffer = phrase_buffer[end_pos:]
                    else:
                        # Buffer too long without punctuation, split at last space
                        last_space = phrase_buffer.rfind(' ', 0, 50)
                        if last_space > 0:
                            complete_text = phrase_buffer[:last_space].strip()
                            phrase_buffer = phrase_buffer[last_space:]
                        else:
                            # No space found, synthesize everything
                            complete_text = phrase_buffer.strip()
                            phrase_buffer = ""
                    
                    if complete_text:
                        # Use audio streaming for this phrase
                        for audio_chunk in self._synthesize_with_audio_streaming(complete_text):
                            yield audio_chunk
            
            # Synthesize any remaining text
            if phrase_buffer.strip():
                logger.debug(f"📝 Synthesizing remaining buffer: '{phrase_buffer.strip()[:50]}...'")
                for audio_chunk in self._synthesize_with_audio_streaming(phrase_buffer.strip()):
                    yield audio_chunk
        
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            if phrase_buffer:
                logger.error(f"Lost text in buffer: '{phrase_buffer}'")
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
