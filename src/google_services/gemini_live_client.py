import os
import asyncio
import logging
import queue
from typing import Optional, Callable
from google import genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiLiveClient:
    """Client for Gemini Multimodal Live API with full-duplex support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp", # Default to flash exp as per typical preview usage
        system_instruction: str = "You are Jarvis, a helpful AI tutor."
    ):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = model
        self.system_instruction = system_instruction
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        
        self.config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": self.system_instruction,
        }
        
        self._session = None
        self._on_audio_received: Optional[Callable[[bytes], None]] = None
        self._on_interrupted: Optional[Callable[[], None]] = None
        self._is_running = False

    def set_callbacks(self, on_audio: Callable[[bytes], None], on_interrupt: Callable[[], None]):
        """Set callbacks for audio and interruption events."""
        self._on_audio_received = on_audio
        self._on_interrupted = on_interrupt

    async def start(self):
        """Starts the Gemini Live session."""
        self._is_running = True
        try:
            async with self.client.aio.live.connect(model=self.model, config=self.config) as session:
                self._session = session
                logger.info("Connected to Gemini Live session")
                
                # Create receiver task
                receive_task = asyncio.create_task(self._receive_loop())
                await receive_task
        except Exception as e:
            logger.error(f"Error in Gemini Live session: {e}")
        finally:
            self._is_running = False
            self._session = None

    async def send_audio(self, audio_data: bytes):
        """Sends audio data to the ongoing session."""
        if self._session and self._is_running:
            await self._session.send(input={"data": audio_data, "mime_type": "audio/pcm"}, end_of_turn=False)

    async def _receive_loop(self):
        """Internal loop to handle responses from Gemini."""
        async for response in self._session.receive():
            if response.server_content:
                if response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if part.inline_data:
                            if self._on_audio_received:
                                self._on_audio_received(part.inline_data.data)
                
                if response.server_content.interrupted:
                    logger.info("Gemini signaled interruption")
                    if self._on_interrupted:
                        self._on_interrupted()

    def stop(self):
        """Signals the session to stop."""
        self._is_running = False
