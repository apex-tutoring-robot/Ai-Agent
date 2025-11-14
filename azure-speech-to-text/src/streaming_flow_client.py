"""
Streaming client for Azure Prompt Flow with sentence-by-sentence processing.
Enables real-time response streaming for lower perceived latency.
"""

import requests
import json
import re
from typing import Generator, Optional, Dict, Any
import sseclient  # For Server-Sent Events parsing


class StreamingFlowClient:
    """Client for streaming responses from Azure Prompt Flow."""
    
    def __init__(self, flow_endpoint: str, flow_api_key: str, session_id: str):
        """
        Initialize the streaming flow client.
        
        Args:
            flow_endpoint: Azure Prompt Flow endpoint URL
            flow_api_key: API key for authentication
            session_id: Session ID for conversation continuity
        """
        self.flow_endpoint = flow_endpoint
        self.flow_api_key = flow_api_key
        self.session_id = session_id
        
        # Check if endpoint supports streaming
        self.streaming_supported = self._check_streaming_support()
    
    def _check_streaming_support(self) -> bool:
        """
        Check if the Prompt Flow endpoint supports streaming.
        
        Returns:
            True if streaming is supported, False otherwise
        """
        # Most Azure OpenAI-based flows support streaming
        # You can customize this based on your endpoint
        return True
    
    def get_streaming_response(self, 
                               user_text: str,
                               learner_id: str = "pi_student") -> Generator[str, None, None]:
        """
        Get streaming response from Prompt Flow, yielding text chunks as they arrive.
        
        Args:
            user_text: The user's input text
            learner_id: Learner identifier
            
        Yields:
            Text chunks as they arrive from the API
        """
        if not self.streaming_supported:
            # Fallback to non-streaming
            full_response = self.get_complete_response(user_text, learner_id)
            yield full_response
            return
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.flow_api_key}",
            "Accept": "text/event-stream"  # Request streaming response
        }
        
        payload = {
            "user_message": user_text,
            "action_type": "chat",
            "learner_id": learner_id,
            "session_id": self.session_id,
            "chat_history": [],
            "stream": True  # Enable streaming
        }
        
        try:
            # Make streaming request
            response = requests.post(
                self.flow_endpoint,
                headers=headers,
                json=payload,
                stream=True,  # Critical: enable streaming
                timeout=60
            )
            
            if response.status_code == 200:
                # Parse Server-Sent Events
                client = sseclient.SSEClient(response)
                
                for event in client.events():
                    if event.data and event.data != "[DONE]":
                        try:
                            # Parse the JSON data
                            data = json.loads(event.data)
                            
                            # Extract text chunk (adjust based on your API response format)
                            if "choices" in data:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            elif "text" in data:
                                yield data["text"]
                            elif "chunk" in data:
                                yield data["chunk"]
                                
                        except json.JSONDecodeError:
                            continue
            else:
                # Error - yield fallback message
                yield "I'm having trouble connecting right now."
                
        except requests.Timeout:
            yield "Sorry, I'm thinking too slowly. Let's try again."
        except Exception as e:
            print(f"⚠️  Streaming error: {e}")
            yield "I'm having technical difficulties."
    
    def get_complete_response(self, 
                             user_text: str,
                             learner_id: str = "pi_student") -> str:
        """
        Get complete response (non-streaming fallback).
        
        Args:
            user_text: The user's input text
            learner_id: Learner identifier
            
        Returns:
            Complete response text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.flow_api_key}"
        }
        
        payload = {
            "user_message": user_text,
            "action_type": "chat",
            "learner_id": learner_id,
            "session_id": self.session_id,
            "chat_history": []
        }
        
        try:
            response = requests.post(
                self.flow_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("final_answer", "I'm not sure how to respond to that.")
            else:
                return "I'm having trouble thinking right now."
                
        except Exception as e:
            print(f"⚠️  Flow API error: {e}")
            return "I'm having technical difficulties."


class SentenceStreamer:
    """
    Helper class to accumulate streaming text and yield complete sentences.
    This ensures we only synthesize complete thoughts, not word fragments.
    """
    
    def __init__(self, min_sentence_length: int = 10):
        """
        Initialize the sentence streamer.
        
        Args:
            min_sentence_length: Minimum characters before considering a sentence
        """
        self.buffer = ""
        self.min_sentence_length = min_sentence_length
        
        # Sentence ending patterns
        self.sentence_endings = re.compile(r'([.!?]+\s+|[.!?]+$)')
    
    def add_chunk(self, chunk: str) -> Generator[str, None, None]:
        """
        Add a text chunk and yield complete sentences.
        
        Args:
            chunk: Text chunk from streaming response
            
        Yields:
            Complete sentences ready for TTS
        """
        self.buffer += chunk
        
        # Check for sentence endings
        sentences = self.sentence_endings.split(self.buffer)
        
        # Keep the last incomplete part
        if len(sentences) > 1:
            # We have at least one complete sentence
            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                sentence = sentence.strip()
                
                # Only yield if long enough (avoid partial sentences)
                if len(sentence) >= self.min_sentence_length:
                    yield sentence
                    self.buffer = ""
            
            # Keep the remaining text
            self.buffer = sentences[-1] if len(sentences) % 2 == 1 else ""
    
    def flush(self) -> Optional[str]:
        """
        Get any remaining text in the buffer.
        
        Returns:
            Remaining text or None if buffer is empty
        """
        if self.buffer.strip():
            result = self.buffer.strip()
            self.buffer = ""
            return result
        return None


def stream_and_speak(streaming_flow_client: 'StreamingFlowClient',
                    tts_client,
                    privacy_manager,
                    user_text: str,
                    device_index: Optional[int] = None,
                    verbose: bool = True) -> dict:
    """
    Stream LLM response and speak sentences as they arrive.
    This is the main function that orchestrates parallel TTS + streaming.
    
    Args:
        streaming_flow_client: Streaming flow client instance
        tts_client: TTS client for speech synthesis
        privacy_manager: Privacy manager for de-anonymization
        user_text: User's input text (anonymized)
        device_index: Audio device index for playback
        verbose: Print status messages
        
    Returns:
        dict with 'success': bool, 'full_response': str, 'interrupted': bool
    """
    import threading
    import queue
    import time
    
    # Queue for sentences ready to be spoken
    sentence_queue = queue.Queue()
    
    # Flags and state
    streaming_done = threading.Event()
    speaking_done = threading.Event()
    full_response = []
    interrupted = False
    
    def streaming_thread():
        """Background thread that streams and queues sentences."""
        try:
            sentence_streamer = SentenceStreamer()
            
            # Get streaming response
            for chunk in streaming_flow_client.get_streaming_response(user_text):
                full_response.append(chunk)
                
                # Check for complete sentences
                for sentence in sentence_streamer.add_chunk(chunk):
                    if verbose:
                        print(f"📝 Queued: \"{sentence[:50]}...\"")
                    sentence_queue.put(sentence)
            
            # Flush any remaining text
            remaining = sentence_streamer.flush()
            if remaining:
                sentence_queue.put(remaining)
            
            # Signal that streaming is done
            streaming_done.set()
            sentence_queue.put(None)  # Sentinel value
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            streaming_done.set()
            sentence_queue.put(None)
    
    def speaking_thread():
        """Background thread that synthesizes and plays sentences."""
        nonlocal interrupted
        
        try:
            sentence_count = 0
            
            while True:
                # Get next sentence (blocking)
                sentence = sentence_queue.get()
                
                # Check for sentinel (end of stream)
                if sentence is None:
                    break
                
                sentence_count += 1
                
                # Restore privacy (de-anonymize)
                restored_sentence = privacy_manager.restore_personal_response(sentence)
                
                if verbose:
                    print(f"\n🔊 Speaking sentence {sentence_count}: \"{restored_sentence[:60]}...\"")
                
                # Synthesize speech
                audio_file = tts_client.synthesize_speech(restored_sentence)
                
                # Play with interrupt detection
                playback_result = tts_client.play_speech_interruptible(
                    audio_file,
                    device_index=device_index
                )
                
                # Check if interrupted
                if playback_result.get('interrupted', False):
                    interrupted = True
                    if verbose:
                        print("⚠️  User interrupted, stopping playback...")
                    break
            
            speaking_done.set()
            
        except Exception as e:
            print(f"❌ Speaking error: {e}")
            speaking_done.set()
    
    # Start both threads
    streamer = threading.Thread(target=streaming_thread, daemon=True)
    speaker = threading.Thread(target=speaking_thread, daemon=True)
    
    if verbose:
        print("🧠 Starting streaming response...")
    
    streamer.start()
    
    # Wait a tiny bit for first sentence to arrive, then start speaking
    time.sleep(0.1)
    speaker.start()
    
    # Wait for both to complete
    streaming_done.wait()
    speaking_done.wait()
    
    # Combine full response
    full_text = "".join(full_response)
    
    return {
        'success': True,
        'full_response': full_text,
        'interrupted': interrupted
    }