"""
Text-to-Speech client for CHIPPY.
Handles converting text responses to speech using Azure Cognitive Services.
"""

import os
import time
import requests
import tempfile
from typing import Optional
from visuals.chippy_face.chippy_animator import ChippyAnimator

class TextToSpeechClient:
    """Client for Azure Text-to-Speech service using REST API for Pi compatibility."""
    
    # def __init__(self, config, voice_name="en-US-AriaNeural"):
    # def __init__(self, config, voice_name="en-US-SaraNeural"):
    # def __init__(self, config, voice_name="en-US-JennyNeural"):
    def __init__(self, config, voice_name="en-US-DavisNeural"):
        """
        Initialize the Text-to-Speech client.
        
        Args:
            config: Configuration object with Azure credentials
            voice_name: Name of the voice to use (default: en-US-AriaNeural)
        """
        # API endpoints
        self.region = config.SPEECH_REGION
        self.key = config.SPEECH_KEY
        self.token_url = f"https://{self.region}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
        self.tts_url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        
        # Voice settings
        self.voice_name = voice_name
        
        # Get access token
        self.access_token = self._get_token()
        self.token_expiry = time.time() + 540  # Tokens valid for ~10 minutes
        self.animator = ChippyAnimator()
    
    def _get_token(self) -> str:
        """Get authentication token for Speech service."""
        headers = {
            'Ocp-Apim-Subscription-Key': self.key
        }
        
        response = requests.post(self.token_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Token request failed with status code: {response.status_code}")
            
        return response.text
    
    def _ensure_valid_token(self) -> None:
        """Ensure we have a valid token, refreshing if necessary."""
        if time.time() > self.token_expiry:
            self.access_token = self._get_token()
            self.token_expiry = time.time() + 540
    
    import azure.cognitiveservices.speech as speechsdk
    import numpy as np


    class TextToSpeechClient:
        def __init__(self, config, voice_name="en-US-DavisNeural"):
            self.region = config.SPEECH_REGION
            self.key = config.SPEECH_KEY
            self.voice_name = voice_name
            self.animator = ChippyAnimator("src/visuals/chippy_face/faces")

        def speak_live(self, text: str):
            """
            Stream Azure TTS audio directly to the speaker and animate face in real time.
            Supports both volume-based and viseme-based mouth animation.
            """
            speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
            speech_config.speech_synthesis_voice_name = self.voice_name
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
            )

            # Send audio to system speaker
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config,
                                                  audio_config=audio_config)

            # --- (A) Volume-driven sync ---
            def handle_audio_chunk(evt):
                if not evt.audio_data:
                    return
                # Convert audio bytes to numpy array
                samples = np.frombuffer(evt.audio_data, dtype=np.int16)
                if len(samples) == 0:
                    return

                # Compute loudness
                rms = np.sqrt(np.mean(samples**2)) / 32768.0
                frame_idx = min(int(rms * 8), 4)  # scale 0–1 to 0–4
                frame = self.animator.mouth_frames[frame_idx]
                self.animator._show_image(frame)

            # --- (B) Phoneme / Viseme mapping ---
            def handle_viseme(evt):
                viseme_id = evt.viseme_id
                mapping = {
                    0: 0,  # neutral
                    1: 1,  # A
                    2: 2,  # E
                    3: 3,  # O
                    4: 4   # M/WQ
                }
                idx = mapping.get(viseme_id % 5, 0)
                self.animator._show_image(self.animator.mouth_frames[idx])

            synthesizer.synthesizing.connect(handle_audio_chunk)
            synthesizer.viseme_received.connect(handle_viseme)

            def on_start(evt):
                self.animator.set_state("talking")

            def on_done(evt):
                self.animator.set_state("neutral")

            synthesizer.synthesis_started.connect(on_start)
            synthesizer.synthesis_completed.connect(on_done)

            print("🎙️ Speaking live...")
            synthesizer.speak_text_async(text).get()
            print("✅ Done speaking.")