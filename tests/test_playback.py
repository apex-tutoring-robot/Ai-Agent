"""
Playback pipeline test.

Mirrors the streaming TTS → AudioPlayer pipeline from src/main.py
(lines ~744-770) to verify that the full text input is spoken regardless
of response length.

Usage:
    python tests/test_playback.py
    python tests/test_playback.py --text "Your custom text here"
    python tests/test_playback.py --long   # uses a long multi-sentence input
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from audio.playback import AudioPlayer
from azure_services.tts_client import TextToSpeechClient

SHORT_TEXT = "Hello! This is a short playback test."

LONG_TEXT = (
    """Come build the next generation of AI agents and high-performance applications - powered by AMD.

This hackathon is your space to explore, experiment, and create with AMD Developer Cloud and ROCm. No hardware, no complex setup - just access to powerful compute and the freedom to build what you actually care about.

Whether you want to prototype an idea, push a system to its limits, or try something completely new, this is the place to do it.

Build an application, agent, or developer tool that feels real, works end-to-end, and shows what AMD’s compute stack can unlock.

Track 1: AI Agents & Agentic Workflows (Best Track for Beginners)

• Objective: Move beyond simple RAG to build sophisticated AI agentic systems and workloads.

• What to Build: Build intelligent AI systems that automate workflows, coordinate agents, or assist users in complex tasks.
• Tech Stack: Utilize frameworks like LangChain, CrewAI, or AutoGen connecting to open-source models (Llama, DeepSeek, Mistral, Qwen).
• Compute Resource: $100 in AMD Developer Cloud credits.

Track 2: Fine-Tuning on AMD GPUs (Advanced / GPU-Intensive)

• Objective: Leverage direct GPU access to fine-tune open-source models for high-impact domain specialization.

• What to Build: Domain-specific LLMs (Healthcare, Finance, Legal, or Code) fine-tuned for accuracy and efficiency on ROCm.
• Tech Stack: ROCm, PyTorch, Hugging Face Optimum-AMD, and vLLM for serving.
• Compute Resource: Access to AMD Instinct MI300X instances via AMD Developer Cloud.

Track 3: Vision & Multimodal AI

• Objective: Build applications that process and understand multiple data types (Images, Video, Audio) using the massive memory bandwidth of AMD GPUs.

• What to Build: High-throughput industrial inspection, medical imaging analysis, or multimodal conversational assistants.
• Tech Stack: Multimodal models (like Llama 3.2 Vision, Qwen-VL) optimized for ROCm™.
• Compute Resource: Access to AMD Instinct MI300X instances via AMD Developer Cloud.

Extra Challenge: Ship It + Build in Public

• Objective: Document your building journey, share insights, and provide feedback on the AMD developer experience.

• Requirements:
1. Share at least 2 technical updates on social media (tag @lablab on X or lablab.ai on LinkedIn, and tag @AIatAMD on X or AMD
"""
)


def test_playback(text: str) -> None:
    """
    Reproduces the exact TTS → AudioPlayer pipeline from main.py.

    Pipeline (mirrors main.py lines ~744-770):
        text (iterator)
        → tts_client.synthesize_stream()   # sentence-buffered Azure TTS
        → audio_player.queue_audio()       # unbounded PCM queue
        → audio_player.stop_streaming()    # drain queue, wait for completion
    """
    print("\n" + "=" * 70)
    print("PLAYBACK PIPELINE TEST")
    print("=" * 70)
    print(f"\nInput text ({len(text)} chars, ~{len(text.split())} words):\n  {text}\n")

    tts_client = TextToSpeechClient()
    audio_player = AudioPlayer()

    # Wrap the static text in an iterator, exactly as main.py feeds LLM stream chunks
    text_iterator = iter([text])

    print("Starting audio stream...")
    audio_player.start_streaming()

    tts_stream = tts_client.synthesize_stream(text_iterator)

    chunks_queued = 0
    for audio_chunk in tts_stream:
        audio_player.queue_audio(audio_chunk)
        chunks_queued += 1

    print(f"Queued {chunks_queued} audio chunk(s). Waiting for playback to finish...")

    # Drain the queue and wait — same as main.py line ~770
    audio_player.stop_streaming()

    print("\nPlayback complete.")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test TTS → AudioPlayer playback pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="Custom text to speak")
    group.add_argument("--long", action="store_true", help="Use long multi-sentence input")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.long:
        text = LONG_TEXT
    else:
        text = SHORT_TEXT

    test_playback(text)


if __name__ == "__main__":
    main()
