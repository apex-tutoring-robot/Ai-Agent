"""
Tests for TextToSpeechClient's speech-rate SSML building.

Regression test for a real bug found via code review: __init__ computed
rate_percent from TTS_SPEECH_RATE but never used it anywhere - both
branches of the if/else that referenced it called the exact same
set_speech_synthesis_output_format(...) regardless, and
synthesize_to_audio() always called speak_text() (plain text, no rate
parameter exists on that call). Setting TTS_SPEECH_RATE to anything other
than 1.0 silently had zero effect on the actual synthesized speech.

Uses TextToSpeechClient.__new__ + manually setting the two attributes
_build_rate_ssml reads (speech_rate, voice) rather than constructing a
real client, since __init__ requires live Azure Speech credentials and
opens a real network connection.
"""

import pytest

from azure_services.tts_client import TextToSpeechClient


def make_client(speech_rate: float, voice: str = "en-US-JennyNeural") -> TextToSpeechClient:
    client = TextToSpeechClient.__new__(TextToSpeechClient)
    client.speech_rate = speech_rate
    client.voice = voice
    return client


class TestBuildRateSsml:
    def test_positive_rate_uses_plus_sign(self):
        client = make_client(speech_rate=1.25)
        ssml = client._build_rate_ssml("hello there")
        assert 'rate="+25%"' in ssml

    def test_negative_rate_uses_minus_sign(self):
        client = make_client(speech_rate=0.75)
        ssml = client._build_rate_ssml("hello there")
        assert 'rate="-25%"' in ssml

    def test_includes_the_configured_voice(self):
        client = make_client(speech_rate=1.5, voice="en-US-GuyNeural")
        ssml = client._build_rate_ssml("hello")
        assert 'name="en-US-GuyNeural"' in ssml

    def test_includes_the_text(self):
        client = make_client(speech_rate=1.5)
        ssml = client._build_rate_ssml("two plus two equals four")
        assert "two plus two equals four" in ssml

    def test_escapes_xml_special_characters(self):
        client = make_client(speech_rate=1.5)
        ssml = client._build_rate_ssml("5 < 10 & 10 > 5")
        assert "&lt;" in ssml
        assert "&amp;" in ssml
        assert "&gt;" in ssml
        # The raw unescaped characters must not appear standalone in the
        # synthesized text position - malformed XML would break synthesis.
        assert "5 < 10 & 10 > 5" not in ssml

    def test_produces_well_formed_speak_root(self):
        client = make_client(speech_rate=1.2)
        ssml = client._build_rate_ssml("hello")
        assert ssml.startswith('<speak version="1.0"')
        assert ssml.endswith("</speak>")
        assert ssml.count("<prosody") == 1
        assert ssml.count("</prosody>") == 1
