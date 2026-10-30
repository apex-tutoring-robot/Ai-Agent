"""
Tests for TextToSpeechClient's SSML building (_build_ssml) - speech-rate/
pitch adjustment plus the conversational mstts:express-as style.

Regression test for a real bug found via code review: __init__ computed
rate_percent from TTS_SPEECH_RATE but never used it anywhere - both
branches of the if/else that referenced it called the exact same
set_speech_synthesis_output_format(...) regardless, and
synthesize_to_audio() always called speak_text() (plain text, no rate
parameter exists on that call). Setting TTS_SPEECH_RATE to anything other
than 1.0 silently had zero effect on the actual synthesized speech.

Also covers a second gap found later, from direct feedback that Jarvis's
speech sounded flat/robotic: _build_ssml (then _build_rate_ssml) was only
used for delivery/non-default-rate calls - most actual speech went
through plain speak_text() instead, with no SSML, no style, no prosody
shaping of any kind. synthesize_to_audio now always builds SSML, and
_build_ssml wraps the text in self.style's mstts:express-as (e.g. "chat"
- Azure's own description is "a casual and relaxed tone"), not just
<prosody>.

Uses TextToSpeechClient.__new__ + manually setting the three attributes
_build_ssml reads (speech_rate, voice, style) rather than constructing a
real client, since __init__ requires live Azure Speech credentials and
opens a real network connection.
"""

import pytest

from azure_services.tts_client import TextToSpeechClient


def make_client(speech_rate: float, voice: str = "en-US-JennyNeural", style: str = "chat") -> TextToSpeechClient:
    client = TextToSpeechClient.__new__(TextToSpeechClient)
    client.speech_rate = speech_rate
    client.voice = voice
    client.style = style
    return client


class TestBuildSsmlRateAndPitch:
    def test_positive_rate_uses_plus_sign(self):
        client = make_client(speech_rate=1.25)
        ssml = client._build_ssml("hello there")
        assert 'rate="+25%"' in ssml

    def test_negative_rate_uses_minus_sign(self):
        client = make_client(speech_rate=0.75)
        ssml = client._build_ssml("hello there")
        assert 'rate="-25%"' in ssml

    def test_includes_the_configured_voice(self):
        client = make_client(speech_rate=1.5, voice="en-US-GuyNeural")
        ssml = client._build_ssml("hello")
        assert 'name="en-US-GuyNeural"' in ssml

    def test_includes_the_text(self):
        client = make_client(speech_rate=1.5)
        ssml = client._build_ssml("two plus two equals four")
        assert "two plus two equals four" in ssml

    def test_escapes_xml_special_characters(self):
        client = make_client(speech_rate=1.5)
        ssml = client._build_ssml("5 < 10 & 10 > 5")
        assert "&lt;" in ssml
        assert "&amp;" in ssml
        assert "&gt;" in ssml
        # The raw unescaped characters must not appear standalone in the
        # synthesized text position - malformed XML would break synthesis.
        assert "5 < 10 & 10 > 5" not in ssml

    def test_produces_well_formed_speak_root(self):
        client = make_client(speech_rate=1.2)
        ssml = client._build_ssml("hello")
        assert ssml.startswith('<speak version="1.0"')
        assert ssml.endswith("</speak>")
        assert ssml.count("<prosody") == 1
        assert ssml.count("</prosody>") == 1


class TestBuildSsmlConversationalStyle:
    def test_wraps_text_in_the_configured_style(self):
        client = make_client(speech_rate=1.0, style="chat")
        ssml = client._build_ssml("hello there")
        assert '<mstts:express-as style="chat">' in ssml
        assert "</mstts:express-as>" in ssml

    def test_declares_the_mstts_namespace(self):
        # mstts:express-as is meaningless to Azure's SSML parser without
        # this namespace declared on the <speak> root.
        client = make_client(speech_rate=1.0, style="chat")
        ssml = client._build_ssml("hello there")
        assert 'xmlns:mstts="http://www.w3.org/2001/mstts"' in ssml

    def test_different_style_is_honored(self):
        client = make_client(speech_rate=1.0, style="cheerful")
        ssml = client._build_ssml("hello there")
        assert 'style="cheerful"' in ssml

    def test_prosody_is_nested_inside_the_style_wrapper(self):
        client = make_client(speech_rate=1.25, style="chat")
        ssml = client._build_ssml("hello there")
        style_open = ssml.index("<mstts:express-as")
        prosody_open = ssml.index("<prosody")
        style_close = ssml.index("</mstts:express-as>")
        assert style_open < prosody_open < style_close

    def test_empty_style_omits_the_wrapper_entirely(self):
        # An escape hatch (TTS_STYLE="") for a voice that doesn't support
        # any style, or a team preference to turn it off - still produces
        # valid SSML with just the <prosody> rate/pitch adjustment.
        client = make_client(speech_rate=1.0, style="")
        ssml = client._build_ssml("hello there")
        assert "mstts:express-as" not in ssml
        assert "<prosody" in ssml


class TestBuildSsmlPerCallStyleOverride:
    """_build_ssml's style/styledegree params - what lets a mood
    (ExpressionController's DeliveryStyle) actually change the Azure
    express-as style per utterance, not just the instance-wide default
    set at TTS client construction (self.style)."""

    def test_per_call_style_overrides_the_instance_default(self):
        client = make_client(speech_rate=1.0, style="chat")
        ssml = client._build_ssml("hello there", style="excited")
        assert 'style="excited"' in ssml
        assert 'style="chat"' not in ssml

    def test_no_style_override_falls_back_to_instance_default(self):
        client = make_client(speech_rate=1.0, style="chat")
        ssml = client._build_ssml("hello there")
        assert 'style="chat"' in ssml

    def test_empty_string_override_omits_the_wrapper_even_with_an_instance_default(self):
        client = make_client(speech_rate=1.0, style="chat")
        ssml = client._build_ssml("hello there", style="")
        assert "mstts:express-as" not in ssml

    def test_styledegree_is_included_when_given(self):
        client = make_client(speech_rate=1.0, style="chat")
        ssml = client._build_ssml("hello there", style="excited", styledegree=1.3)
        assert 'styledegree="1.3"' in ssml

    def test_styledegree_is_omitted_when_not_given(self):
        # Letting Azure fall back to its own default (1.0) rather than
        # always forcing the attribute - matches _build_ssml's existing
        # "omit rather than force a value" pattern for optional SSML bits.
        client = make_client(speech_rate=1.0, style="chat")
        ssml = client._build_ssml("hello there", style="excited")
        assert "styledegree" not in ssml
