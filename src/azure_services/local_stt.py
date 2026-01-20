"""
Local Speech-to-Text client using Whisper Tiny (ONNX).
Optimized for low-latency inference on Raspberry Pi.

In professional AI research, "Speech-to-Text" is almost always referred to as Automatic Speech Recognition (ASR).
While "Speech-to-Text" (STT) describes the outcome, ASR describes the process of a machine mapping audio signals to linguistic sequences.
"""

import os
import logging
import numpy as np
from typing import Iterator, Generator, Optional
from pathlib import Path
from dotenv import load_dotenv
import threading
import onnxruntime as ort
from transformers import WhisperTokenizer, WhisperProcessor
import librosa

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class LocalSTTClient:
    """Local Whisper STT client using ONNX Runtime with quantized models."""
    
    # Model configurations (using simple decoders to avoid KV cache complexity)
    MODELS = {
        'fp16': ('encoder_model_fp16.onnx', 'decoder_model_fp16.onnx'),           # ~73 MB - Recommended for Pi
        'fp32': ('encoder_model.onnx', 'decoder_model.onnx'),                    # ~151 MB - Best quality
        'bnb4': ('encoder_model_bnb4.onnx', 'decoder_model_bnb4.onnx'),          # ~95 MB - Alternative
        'q4': ('encoder_model_q4.onnx', 'decoder_model_q4.onnx'),                # ~95 MB - Alternative 
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = 'fp16'
    ):
        """
        Initialize Local STT client with quantized Whisper ONNX model.
        
        Args:
            model_path: Direct path to model directory (overrides model_type)
            model_type: Model type to use ('int8', 'quantized', 'bnb4', 'fp16', 'fp32')
        """
        if ort is None:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        if WhisperProcessor is None:
            raise ImportError("transformers not installed. Install with: pip install transformers")
        if librosa is None:
            raise ImportError("librosa not installed. Install with: pip install librosa")
        
        # Determine model paths
        if model_path:
            model_dir = Path(model_path)
        else:
            # Default to model in repository
            model_dir = Path(__file__).parent.parent.parent / 'whisper-tiny-ONNX' / 'onnx'
        
        encoder_file, decoder_file = self.MODELS.get(model_type, self.MODELS['fp16'])
        self.encoder_path = model_dir / encoder_file
        self.decoder_path = model_dir / decoder_file
        
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {self.encoder_path}")
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Decoder model not found: {self.decoder_path}")
        
        logger.info(f"Loading Whisper Tiny ONNX models ({model_type}):")
        logger.info(f"  Encoder: {self.encoder_path.name} ({self.encoder_path.stat().st_size / 1024 / 1024:.1f} MB)")
        logger.info(f"  Decoder: {self.decoder_path.name} ({self.decoder_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        try:
            # Load tokenizer and processor
            tokenizer_dir = model_dir.parent
            self.tokenizer = WhisperTokenizer.from_pretrained(str(tokenizer_dir))
            self.processor = WhisperProcessor.from_pretrained(str(tokenizer_dir))
            logger.info(f"✓ Tokenizer loaded: {len(self.tokenizer)} tokens")
            
            # Create ONNX Runtime sessions with minimal optimization
            # These models are already optimized, so disable runtime optimization to avoid conflicts
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            
            # Use fewer threads on Pi to avoid overwhelming CPU
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            
            # Load encoder
            self.encoder_session = ort.InferenceSession(
                str(self.encoder_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            
            # Load decoder
            self.decoder_session = ort.InferenceSession(
                str(self.decoder_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            
            logger.info("ONNX models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX models: {e}")
            raise
    
    def _audio_to_mel_spectrogram(self, audio_data: bytes, sample_rate: int = 16000) -> np.ndarray:
        """
        Convert raw audio bytes to mel spectrogram for Whisper.
        
        Args:
            audio_data: Raw audio bytes (PCM 16-bit)
            sample_rate: Audio sample rate
            
        Returns:
            Mel spectrogram as numpy array
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        
        # Use Whisper processor to get mel spectrogram
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np"
        )
        
        return inputs.input_features
    
    def recognize_from_audio_data(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        Recognize speech from raw audio data.
        
        Args:
            audio_data: Raw audio bytes (PCM 16-bit)
            sample_rate: Audio sample rate
        
        Returns:
            Transcribed text
        """
        try:
            if not audio_data:
                logger.warning("Empty audio data provided")
                return ""
            
            logger.info("Converting audio to mel spectrogram...")
            mel_features = self._audio_to_mel_spectrogram(audio_data, sample_rate)
            
            # Run encoder
            logger.info("Running encoder...")
            encoder_outputs = self.encoder_session.run(
                None,
                {"input_features": mel_features}
            )
            encoder_hidden_states = encoder_outputs[0]
            
            # Prepare decoder inputs
            decoder_input_ids = np.array([[self.tokenizer.bos_token_id]], dtype=np.int64)
            
            # Run decoder to generate tokens
            logger.info("Running decoder...")
            generated_tokens = [self.tokenizer.bos_token_id]
            max_length = 448  # Whisper's max length
            
            for _ in range(max_length):
                decoder_outputs = self.decoder_session.run(
                    None,
                    {
                        "input_ids": decoder_input_ids,
                        "encoder_hidden_states": encoder_hidden_states
                    }
                )
                
                # Get logits and find most likely next token
                logits = decoder_outputs[0]
                next_token_id = np.argmax(logits[0, -1, :])
                
                # Check for end of sequence
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(int(next_token_id))
                decoder_input_ids = np.array([generated_tokens], dtype=np.int64)
            
            # Decode tokens to text
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.info(f"✓ Recognized: {text}")
            
            return text
        
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            raise
    
    def recognize_streaming(self, audio_stream: Iterator[bytes], sample_rate: int = 16000) -> Generator[str, None, None]:
        """
        Recognize speech from streaming audio chunks.
        Collects audio chunks and processes as complete utterance.
        
        Args:
            audio_stream: Iterator yielding audio chunks (bytes)
            sample_rate: Audio sample rate
            
        Yields:
            Transcribed text (final result only)
        """
        try:
            # Collect all audio chunks
            audio_chunks = []
            chunk_count = 0
            
            logger.info("Collecting streaming audio chunks...")
            for audio_chunk in audio_stream:
                if audio_chunk is None:
                    # End of stream signal
                    logger.info(f"End of audio stream (received {chunk_count} chunks)")
                    break
                audio_chunks.append(audio_chunk)
                chunk_count += 1
            
            if not audio_chunks:
                logger.warning("No audio chunks received")
                return
            
            # Concatenate all chunks
            logger.info(f"Collected {chunk_count} chunks, concatenating...")
            audio_data = b''.join(audio_chunks)
            
            # Process complete audio
            logger.info(f"Processing {len(audio_data)} bytes of audio...")
            text = self.recognize_from_audio_data(audio_data, sample_rate)
            
            if text:
                yield text
            else:
                logger.warning("No speech recognized")
        
        except Exception as e:
            logger.error(f"Error in streaming recognition: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the loaded models."""
        return {
            'encoder_path': str(self.encoder_path),
            'decoder_path': str(self.decoder_path),
            'encoder_size_mb': self.encoder_path.stat().st_size / 1024 / 1024,
            'decoder_size_mb': self.decoder_path.stat().st_size / 1024 / 1024,
            'total_size_mb': (self.encoder_path.stat().st_size + self.decoder_path.stat().st_size) / 1024 / 1024,
        }


if __name__ == "__main__":
    # Test local STT with ONNX Runtime
    print("Testing Whisper Tiny ONNX STT...")
    print("-" * 60)
    
    try:
        # Initialize client with quantized model (best for 4GB Pi)
        client = LocalSTTClient(model_type='fp16')
        
        # Show model info
        info = client.get_model_info()
        print(f"\nModels loaded:")
        print(f"  Encoder: {info['encoder_path']} ({info['encoder_size_mb']:.1f} MB)")
        print(f"  Decoder: {info['decoder_path']} ({info['decoder_size_mb']:.1f} MB)")
        print(f"  Total: {info['total_size_mb']:.1f} MB")
        
        
        # Test with actual audio file
        print("\nTesting with audio file...")
        from pydub import AudioSegment
        audio = AudioSegment.from_file('/home/pi/Documents/apex-code/Ai-Agent/test_recording.wav')
        audio_bytes = audio.raw_data
        text = client.recognize_from_audio_data(audio_bytes)
        print(f"\nTranscription: {text}")
        
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
