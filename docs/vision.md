# Vision Pipeline

## Overview

Jarvis supports a one-shot camera vision flow that lets students say "scan my homework" and have the image content extracted and stored as plain text in the conversation. All subsequent turns are text-only — the image is never re-sent to the LLM.

## Camera Pipeline

1. Wake word detected → user says a camera trigger phrase (e.g. "scan my homework", "take a picture").
2. `Camera.capture_and_save()` captures a frame via `picamera2` and writes it to `captures/` on disk.
3. The saved JPEG is read and encoded as a `data:image/jpeg;base64,...` data URL in memory.
4. `LLMClient.extract_image_content()` makes a single non-streaming GPT-4V call with the data URL and an extraction system prompt.
5. The returned plain-text description is combined with the user's original speech and stored in `ConversationStateManager` as a normal text message.
6. All follow-up turns query the LLM with text-only history — no image tokens on turn 2+.

## Why Text Extraction Instead of Sending the Image Every Turn

Azure OpenAI Vision encodes images as patch tokens before every API call. A typical homework photo costs **765–1105 tokens per turn** regardless of whether the image content is still relevant. Storing a text description costs roughly **100–300 tokens once**, and zero extra tokens on follow-up turns.

| Approach | Turn 1 | Turn 2 | Turn 3 | Turn N |
|---|---|---|---|---|
| Image URL in history (old) | ~900 img tokens | ~900 img tokens | ~900 img tokens | ~900 img tokens |
| Text extraction (new) | ~200 extraction tokens | 0 img tokens | 0 img tokens | 0 img tokens |

For a 5-turn homework session the new approach saves roughly 3,000–4,000 tokens.

## Extraction Prompt

The extraction system prompt instructs the model to:
- Transcribe every question, equation, diagram, number, and piece of text.
- Write all math in plain spoken form (e.g. "2x plus 5 equals 15") so TTS can read it naturally.
- Be exhaustive — no content skipped.

## Fallback Strategy

If `extract_image_content()` raises an exception (network error, quota, etc.):
1. Jarvis speaks: *"Sorry, I had trouble reading the image. Please try again."*
2. The turn is skipped (`continue`) — no LLM call is made.
3. The image is still saved to disk in `captures/` for debugging.
4. The conversation history stays clean (no partial or corrupt entries).

## SAS URL Expiry — No Longer Relevant

The previous implementation uploaded images to Azure Blob Storage and stored a time-limited SAS URL in conversation history. SAS URLs expire (typically after 1 hour), causing silent failures on long sessions. The new approach never uploads to blob storage — the base64 data URL is used only for the single extraction call and is never stored in history.
