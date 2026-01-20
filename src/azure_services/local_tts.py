"""
Local Text-to-Speech client using Kokoro TTS (ONNX).
Optimized for low-latency inference on Raspberry Pi.
Kokoro-82M is a neural network. Specifically, it is a text-to-speech (TTS) model
Kokoro-82M has a disc size of 325 MB but the ONNX quantized model has a disc size of 86 MB.

The model takes your input text and converts it into numerical representations (embeddings)
Acoustic Model is usually a transformer-based architecture that processes the text representations. 
It generates intermediate acoustic features like mel-spectrograms (visual representations of sound frequencies over time)
Vocoder takes those mel-spectrograms and converts them into actual audio waveforms you can hear. 
This is the final step that creates the actual sound waves

There is a technical and functional distinction between Text-to-Speech (TTS) and Text-to-Audio in the context of modern AI.
Text-to-Speech (TTS): Focuses exclusively on the human voice. Its goal is to take text and synthesize a speaker's voice with natural intonation, rhythm, and pronunciation.
Text-to-Audio: A broader category that includes TTS but can also generate non-speech sounds. A text-to-audio model might take the prompt "birds chirping in a rainy forest" and produce an environmental soundscape rather than a spoken voice.
"""

import os
import logging
import re
from typing import Iterator, Optional
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import onnxruntime as ort
import json
import psutil

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


def log_system_resources(prefix=""):
    """Log current RAM and CPU usage."""
    if psutil is None:
        return
    
    try:
        # Memory info
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        mem_percent = mem.percent
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        logger.info(f"{prefix}RAM: {mem_used_gb:.2f}/{mem_total_gb:.2f} GB ({mem_percent:.1f}%) | CPU: {cpu_percent:.1f}%")
    except Exception as e:
        logger.warning(f"Could not get system resources: {e}")


class LocalTTSClient:
    """Local Kokoro TTS client using ONNX Runtime with quantized models."""
    
    # Model configurations
    MODELS = {
        'q8f16': 'model_q8f16.onnx',      # 86 MB - Best for 4GB Pi (Recommended)
        'quantized': 'model_quantized.onnx',  # 92 MB - Alternative
        'q4f16': 'model_q4f16.onnx',      # 154 MB - Higher quality
        'fp16': 'model_fp16.onnx',        # 163 MB - Full precision FP16
        'full': 'model.onnx',             # 325 MB - Original (not recommended for Pi)
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = 'q8f16',
        speed: float = os.getenv('LOCAL_TTS_SPEED'),
        voice: str = os.getenv('LOCAL_TTS_VOICE')
    ):
        """
        Initialize Local TTS client with quantized Kokoro ONNX model.
        
        Args:
            model_path: Direct path to ONNX model file (overrides model_type)
            model_type: Model type to use ('q8f16', 'quantized', 'q4f16', 'fp16', 'full')
            speed: Speech speed multiplier (1.0 is normal)
        """
               
        self.speed = speed
        
        # Determine model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            # Default to model in repository
            model_dir = Path(__file__).parent.parent.parent / 'Kokoro-82M-v1.0-ONNX' / 'onnx'
            model_file = self.MODELS.get(model_type, self.MODELS['q8f16'])
            self.model_path = model_dir / model_file
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load tokenizer vocab directly from JSON (avoid HuggingFace compatibility issues)
        tokenizer_path = Path(__file__).parent.parent.parent / 'Kokoro-82M-v1.0-ONNX' / 'tokenizer.json'
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        if json is None:
            raise ImportError("json module not available")
        
        # Load tokenizer config
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        
        # Extract vocab (character/phoneme to ID mapping)
        self.vocab = tokenizer_config['model']['vocab']
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        logger.info(f"✓ Tokenizer loaded: {len(self.vocab)} tokens")
        
        # Load voice embeddings (style)
        # Per Kokoro docs: voice file contains multiple 256-dim embeddings (one per token length)
        # Format: reshape(-1, 1, 256) then index by len(tokens)
        
        self.voice = os.getenv("LOCAL_TTS_VOICE")
        
        if self.voice:
            # Load voice embeddings from binary file
            # Shape: (num_token_lengths, 1, 256)
            self.voice_data = np.fromfile(str(self.voice), dtype=np.float32).reshape(-1, 1, 256)
            logger.info(f"✓ Voice loaded:({self.voice_data.shape[0]} embeddings of shape (1, 256))")
        else:
            logger.warning(f"Voice files not found in {voices_dir}, using zero embeddings")
            # Create dummy voice data (512 possible token lengths)
            self.voice_data = np.zeros((512, 1, 256), dtype=np.float32)
        
        logger.info(f"Loading Kokoro TTS model: {self.model_path.name}")
        logger.info(f"Model size: {self.model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Check resources before loading model
        # log_system_resources("[BEFORE MODEL LOAD] ")
        
        try:
            # Create ONNX Runtime session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Use fewer threads on Pi to avoid overwhelming CPU
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            
            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']  # Pi doesn't have GPU
            )
            
            # Get model input/output names
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Inputs: {self.input_names}")
            logger.info(f"  Outputs: {self.output_names}")
            
            # Check resources after loading model
            # log_system_resources("[AFTER MODEL LOAD] ")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _text_to_phonemes(self, text: str) -> np.ndarray:
        """
        Convert text to phoneme/character sequence using Kokoro tokenizer vocab.
        
        The tokenizer uses a vocabulary of:
        - Punctuation: $ ; : , . ! ? etc.
        - Letters: a-z, A, I, O, Q, S, T, W, Y  
        - IPA phonetic symbols for pronunciation
        
        Args:
            text: Input text
            
        Returns:
            Token IDs as numpy array (batch_size=1, sequence_length)
        """
        # Tokenize text using the vocab mapping
        # Add special start/end tokens ($ = token ID 0)
        token_ids = [0]  # Start token
        
        for char in text:
            # Look up character in vocab
            if char in self.vocab:
                token_ids.append(self.vocab[char])
            else:
                # Unknown character - use $ (pad token)
                logger.debug(f"Unknown character '{char}', using pad token")
                token_ids.append(0)
        
        token_ids.append(0)  # End token
        
        # Convert to numpy array with shape (1, seq_len) for ONNX model
        return np.array([token_ids], dtype=np.int64)
    
    def synthesize_to_audio(self, text: str) -> bytes:
        """
        Synthesize text to audio using ONNX Runtime.
        
        Args:
            text: Text to synthesize
        
        Returns:
            Raw audio bytes (PCM 16-bit, 24kHz, mono)
        """
        try:
            if not text.strip():
                logger.warning("Empty text provided for synthesis")
                return b''
            
            logger.info(f"Synthesizing: {text[:50]}...")
            
            # Prepare all three inputs required by Kokoro ONNX model
            input_ids = self._text_to_phonemes(text)
            
            # Select voice embedding based on token count (per Kokoro docs)
            # voices[len(tokens)] - different lengths need different style vectors
            token_count = input_ids.shape[1]  # Get sequence length
            
            # Ensure token count is within bounds
            if token_count >= self.voice_data.shape[0]:
                logger.warning(f"Token count {token_count} exceeds voice data size {self.voice_data.shape[0]}, using last embedding")
                voice_embedding = self.voice_data[-1]
            else:
                voice_embedding = self.voice_data[token_count]
            
            inputs = {
                'input_ids': input_ids,
                'style': voice_embedding,  # Selected based on token count
                'speed': np.array([self.speed], dtype=np.float32)  # Speed as numpy array
            }
            
            # Debug: Log input shapes
            logger.info(f"Input shapes: input_ids={input_ids.shape}, style={voice_embedding.shape}, speed={np.array([self.speed]).shape}")
            # log_system_resources("[BEFORE INFERENCE] ")
            logger.info("Running ONNX inference (may take 10-30s on first run)...")
        
            # Run inference
            outputs = self.session.run(self.output_names, inputs)
            
            # Get audio output waveform
            audio_array = outputs[0]  # Shape: [1, samples] or [samples]
            
            # Flatten if needed
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
            
            # Convert to PCM 16-bit bytes
            # Kokoro outputs float32 in range [-1, 1]
            audio_array = np.clip(audio_array, -1.0, 1.0)  # Ensure in valid range
            audio_array = (audio_array * 32767).astype(np.int16)
            
            audio_bytes = audio_array.tobytes()
            
            log_system_resources("[AFTER INFERENCE] ")
            logger.info(f"✓ Synthesis complete: {len(audio_bytes)} bytes")
            return audio_bytes
        
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            raise
    
    def synthesize_stream(self, text_stream: Iterator[str]) -> Iterator[bytes]:
        """
        Synthesize streaming text to audio chunks.
        Buffers text until sentence boundaries for natural speech and optimal latency.
        
        Args:
            text_stream: Iterator yielding text chunks from LLM
        
        Yields:
            Audio data chunks
        """
        sentence_buffer = ""
        
        # Sentence boundary pattern (period, question mark, exclamation)
        # This pattern detects: "word." or "word!" or "word?" followed by space or end
        sentence_endings = re.compile(r'[.!?](?:\s+|$)')
        
        try:
            for text_chunk in text_stream:
                sentence_buffer += text_chunk

                
                # Check for sentence boundaries
                match = sentence_endings.search(sentence_buffer)
                
                if match:
                    # Extract complete sentence(s)
                    end_pos = match.end()
                    complete_text = sentence_buffer[:end_pos].strip()
                    sentence_buffer = sentence_buffer[end_pos:].lstrip()  # Keep remaining, trim left space
                    
                    if complete_text:
                        # Synthesize the complete sentence
                        logger.info(f"📝 Synthesizing: '{complete_text[:50]}...'")
                        audio_data = self.synthesize_to_audio(complete_text)
                        if audio_data:
                            yield audio_data
            
            # CRITICAL: Synthesize any remaining text after stream ends
            if sentence_buffer.strip():
                logger.info(f"📝 Synthesizing final buffer ({len(sentence_buffer)} chars)")
                audio_data = self.synthesize_to_audio(sentence_buffer.strip())
                if audio_data:
                    logger.info(f"✓ Final buffer synthesized: {len(audio_data)} bytes")
                    yield audio_data
                else:
                    logger.warning("⚠️  Final buffer synthesis returned no audio!")
            else:
                logger.info("✓ All text synthesized")
        
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            # If error occurs and buffer has content, log it
            if sentence_buffer.strip():
                logger.error(f"Lost text in buffer: '{sentence_buffer}'")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_path': str(self.model_path),
            'model_size_mb': self.model_path.stat().st_size / 1024 / 1024,
            'inputs': self.input_names,
            'outputs': self.output_names,
            'speed': self.speed
        }


if __name__ == "__main__":
    # Test local TTS with ONNX Runtime
    print("Testing Kokoro ONNX TTS...")
    print("-" * 60)
    
    try:
        # Initialize client with quantized model (best for 4GB Pi)
        client = LocalTTSClient(model_type='quantized')
        
        # Show model info
        info = client.get_model_info()
        print(f"\n✓ Model loaded:")
        print(f"  Path: {info['model_path']}")
        print(f"  Size: {info['model_size_mb']:.1f} MB")
        print(f"  Inputs: {info['inputs']}")
        print(f"  Outputs: {info['outputs']}")
        
        # Test 1: Simple synthesis
        print("\n" + "=" * 60)
        print("Test 1: Simple synthesis")
        print("=" * 60)
        text = "Hello! I am Chippy, your AI assistant."
        print(f"Input: '{text}'")
        audio = client.synthesize_to_audio(text)
        print(f"✓ Generated {len(audio)} bytes of audio")
        
        # Save to WAV file and play
        import wave
        import subprocess
        output_file = '/tmp/kokoro_test.wav'
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz (Kokoro output rate)
            wav_file.writeframes(audio)
        print(f"✓ Saved to: {output_file}")
        
        # Play audio through ReSpeaker 3.5mm jack (card 3)
        print("▶️  Playing audio through ReSpeaker...")
        try:
            subprocess.run(['aplay', '-D', 'plughw:3,0', output_file], check=True)
            print("✓ Playback complete")
        except subprocess.CalledProcessError:
            print("⚠️  Could not play audio (is aplay installed?)")
        except FileNotFoundError:
            print("⚠️  aplay not found, audio saved but not played")
        
        # Test 2: Streaming synthesis
        print("\n" + "=" * 60)
        print("Test 2: Streaming synthesis")
        print("=" * 60)
        
        def mock_llm_stream():
            """Simulate LLM streaming output"""
            chunks = [
                "Hello", " there", "!", " How", " are", " you", " doing", 
                " today", "?", " I", " can", " help", " you", " learn", "."
            ]
            for chunk in chunks:
                yield chunk
        
        print("Simulating LLM stream:")
        print("  Chunks: Hello there! How are you doing today? I can help you learn.")
        
        # Track TTFAS (Time to First Audio Stream) metrics
        import time
        llm_start_time = None
        first_audio_generated_time = None
        first_audio_played_time = None
        
        total_audio = 0
        chunk_count = 0
        
        # Simulate LLM token timing
        llm_start_time = time.perf_counter()
        print(f"\n⏱️  LLM Start: {llm_start_time:.3f}s (simulated)")
        
        for audio_chunk in client.synthesize_stream(mock_llm_stream()):
            chunk_count += 1
            
            # Mark first audio generation
            if first_audio_generated_time is None:
                first_audio_generated_time = time.perf_counter()
                ttfa_generated = (first_audio_generated_time - llm_start_time) * 1000
                print(f"\n⚡ FIRST AUDIO GENERATED: {ttfa_generated:.0f}ms from LLM start")
                
                # Save first chunk to file and play it
                import wave
                first_chunk_file = '/tmp/first_chunk.wav'
                with wave.open(first_chunk_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(24000)
                    wav_file.writeframes(audio_chunk)
                
                # Play first chunk and measure TTFAS
                print(f"▶️  Playing first audio chunk...")
                try:
                    subprocess.run(['aplay', '-D', 'plughw:3,0', first_chunk_file], 
                                 check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    first_audio_played_time = time.perf_counter()
                    ttfas = (first_audio_played_time - llm_start_time) * 1000
                    print(f"\n🎯 TTFAS (Time to First Audio Stream): {ttfas:.0f}ms")
                    print(f"   Breakdown:")
                    print(f"   - Generation: {ttfa_generated:.0f}ms")
                    print(f"   - Playback start: {(ttfas - ttfa_generated):.0f}ms")
                except Exception as e:
                    print(f"⚠️  Could not play first chunk: {e}")
            
            total_audio += len(audio_chunk)
            print(f"  Chunk {chunk_count}: {len(audio_chunk)} bytes")
        
        print(f"\n✓ Streaming complete:")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Total audio: {total_audio} bytes")
        
        if llm_start_time and first_audio_played_time:
            total_time = (time.perf_counter() - llm_start_time) * 1000
            print(f"\n📊 Latency Summary:")
            print(f"  TTFAS: {ttfas:.0f}ms ⚡")
            print(f"  Total time: {total_time:.0f}ms")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

