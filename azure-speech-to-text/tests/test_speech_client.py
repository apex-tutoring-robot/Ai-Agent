from unittest import TestCase
from src.speech_client import SpeechClient

class TestSpeechClient(TestCase):
    def setUp(self):
        self.client = SpeechClient()

    def test_initialization(self):
        self.assertIsNotNone(self.client)

    def test_recognize_speech(self):
        # This is a placeholder for the actual audio input
        audio_input = "path/to/audio/file.wav"
        result = self.client.recognize_speech(audio_input)
        self.assertIsInstance(result, str)  # Assuming the result should be a string

    def test_recognize_speech_invalid(self):
        # Test with an invalid audio input
        audio_input = "invalid/path/to/audio/file.wav"
        with self.assertRaises(FileNotFoundError):
            self.client.recognize_speech(audio_input)