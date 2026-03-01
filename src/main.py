import os
os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import threading
import time
import logging
from visuals.faces.face_animator import FaceAnimator
import numpy as np

from audio.wake_word import WakeWordDetector
from audio.continuous_vad import ContinuousVADCapture
from audio.playback import AudioPlayer
from azure_services.stt_client import SpeechToTextClient
from azure_services.llm_client import LLMClient
from azure_services.tts_client import TextToSpeechClient
from conversation.state_manager import ConversationStateManager
from privacy.privacy_manager import PrivacyManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JarvisBot:
    def __init__(self, face):
        self.face = face
        self.wake = WakeWordDetector()
        self.audio = AudioPlayer(on_level=self.face.push_mouth_level)
        self.stt = SpeechToTextClient()
        self.llm = LLMClient()
        self.tts = TextToSpeechClient()
        self.conv = ConversationStateManager()
        self.privacy = PrivacyManager()

    def run(self):
        logger.info("Jarvis worker started")
        self.face.start_idle()
        self.wake.start(self._on_wake)

    def _on_wake(self):
        logger.info("Wake word detected")
        self.wake.stop()

        vad = ContinuousVADCapture(idle_timeout_seconds=10)
        vad.last_speech_time = time.time()

        while time.time() - vad.last_speech_time < 10:
            self.face.start_thinking()
            time.sleep(0.5)

            audio_stream = vad.stream_audio_chunks()
            user_text = ""

            for text, _ in self.stt.recognize_streaming(audio_stream):
                user_text = text

            if not user_text.strip():
                continue

            logger.info(f"User: {user_text}")
            self.conv.add_user_message(self.privacy.anonymize(user_text))

            self.audio.start_streaming()
            self.face.start_talking()

            for audio in self.tts.synthesize_stream(
                self.llm.generate_response_stream(self.conv.get_messages())
            ):
                self.audio.queue_audio(audio)

            self.audio.stop_streaming()
            self.face.stop_talking()
            self.face.start_idle()
            vad.reset_idle_timer()

def main():
    face = FaceAnimator("visuals/faces")
    jarvis = JarvisBot(face)

    worker = threading.Thread(target=jarvis.run, daemon=True)
    worker.start()

    # 🔥 THIS MUST BE MAIN THREAD
    face.render_forever()


if __name__ == "__main__":
    main()
