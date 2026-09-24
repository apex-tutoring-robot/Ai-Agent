"""
Provider-agnostic interfaces for STT/LLM/TTS. The current implementations
(SpeechToTextClient, LLMClient, TextToSpeechClient) are all Azure-specific;
nothing in main.py should need to change if a different provider is ever
swapped in, as long as the replacement satisfies these interfaces.

Uses typing.Protocol (structural typing), not ABC inheritance - the existing
classes already satisfy these contracts as-is, so this is purely additive
and requires no change to their class declarations. It formalizes contracts
that already exist implicitly in how main.py calls them.
"""

from typing import Callable, Dict, Iterator, List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class SpeechToTextProvider(Protocol):
    def recognize_streaming(
        self, audio_stream: Iterator[bytes], sample_rate: int = 16000
    ) -> Iterator[Tuple[str, bool, float]]:
        """Yields (text, is_final, timestamp) tuples as speech is recognized."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    def generate_response_stream(
        self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500
    ) -> Iterator[str]:
        """Yields text chunks as the LLM generates a response."""
        ...

    def generate_response_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_executor: Callable[[str, dict], str],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Iterator[str]:
        """
        The actual method main.py's live turn-handling path calls (see
        _speaker_loop) - generate_response_stream above is only reachable
        from _process_turn/_process_turn_streaming, both dead code
        superseded by the current listener/speaker thread architecture.
        A provider isn't required to run real tool-calling (an
        implementation may just answer without tools - see
        LocalLLMClient.generate_response_with_tools), but it must accept
        this call shape without erroring.
        """
        ...


@runtime_checkable
class TextToSpeechProvider(Protocol):
    def synthesize_stream(self, text_stream: Iterator[str]) -> Iterator[bytes]:
        """Yields audio chunks as text is synthesized."""
        ...

    def cleanup(self) -> None:
        ...
