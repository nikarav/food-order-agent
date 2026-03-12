"""
ElevenLabs Text-to-Speech with sentence-level streaming.

Key idea: rather than waiting for the full agent response to be synthesised
in one shot, we split the text into sentences and synthesise each one
individually. The first sentence's audio is queued for playback as soon as
it arrives; subsequent sentences are synthesised concurrently while the
previous sentence is still playing.

Result: the user hears the first word of the response ~sentence1-TTS-latency
after the agent returns its text, instead of total-text-TTS-latency.

  Response: "Added Classic Burger to your order. Anything else?"
  ┌─ TTS sentence 1 ─────┐  ← synthesised first, playback starts immediately
                    ┌─ TTS sentence 2 ──┐  ← synthesised while sentence 1 plays
"""

import asyncio
import logging
import re
import time

logger = logging.getLogger(__name__)

# Sentence boundary: period/exclamation/question followed by whitespace or end of string
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")

# Protect common abbreviations and price decimals from being split
_PROTECT = [
    (re.compile(r"\bMr\."), "Mr\x00"),
    (re.compile(r"\bMrs\."), "Mrs\x00"),
    (re.compile(r"\bDr\."), "Dr\x00"),
    (re.compile(r"\bJr\."), "Jr\x00"),
    (re.compile(r"\bSr\."), "Sr\x00"),
    (re.compile(r"\bvs\."), "vs\x00"),
    (re.compile(r"\be\.g\."), "e\x00g\x00"),
    (re.compile(r"\bi\.e\."), "i\x00e\x00"),
    # Prices like $10.50 — replace the decimal point
    (re.compile(r"\$\d+\.(\d)"), lambda m: m.group().replace(".", "\x00")),
]


def split_sentences(text: str) -> list[str]:
    """
    Split *text* into sentences suitable for incremental TTS.

    Handles common abbreviations and price strings to avoid false splits.
    Very long sentences (>120 chars) are further split at comma boundaries
    for more natural pacing.

    :param text: Agent response text
    :return: Non-empty list of sentence strings
    """
    if not text or not text.strip():
        return []

    protected = text
    for pattern, repl in _PROTECT:
        if callable(repl):
            protected = pattern.sub(repl, protected)
        else:
            protected = pattern.sub(repl, protected)

    parts = _SENTENCE_END.split(protected)
    sentences = [s.replace("\x00", ".").strip() for s in parts if s.strip()]

    # Split very long sentences at clause boundaries for better pacing
    result: list[str] = []
    for s in sentences:
        if len(s) > 120:
            clauses = re.split(r"(?<=,)\s+(?=\S{3,})", s)
            result.extend(c.strip() for c in clauses if c.strip())
        else:
            result.append(s)

    return result or [text.strip()]


class ElevenLabsTTS:
    """
    Async streaming TTS using ElevenLabs.

    Splits the response into sentences and synthesises them one by one,
    placing audio bytes into an asyncio.Queue. A background producer task
    runs concurrently with playback so subsequent sentences are ready before
    the current one finishes playing.

    :param api_key: ElevenLabs API key
    :param voice_id: Voice to use
    :param model_id: TTS model — ``eleven_flash_v2_5`` for lowest latency
    :param output_format: ``pcm_16000`` returns raw int16 PCM at 16 kHz
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "pcm_16000",
    ):
        from elevenlabs.client import AsyncElevenLabs

        self._client = AsyncElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format

    async def stream_sentences(
        self,
        text: str,
        cancel_event: asyncio.Event,
    ) -> tuple[asyncio.Queue, float]:
        """
        Synthesise *text* sentence-by-sentence and return an audio queue.

        The queue contains bytes objects (PCM int16 audio per sentence).
        A sentinel ``None`` is placed at the end. A background task drives
        production so the caller can start consuming (playing) immediately.

        :param text: Full response text to synthesise
        :param cancel_event: Set this to abort synthesis mid-stream
        :return: (audio_queue, first_sentence_tts_latency_ms)
        """
        sentences = split_sentences(text)
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=20)

        if not sentences:
            await audio_queue.put(None)
            return audio_queue, 0.0

        # Resolved when the first sentence's bytes land in the queue
        first_byte: asyncio.Future[float] = asyncio.get_event_loop().create_future()

        async def _produce() -> None:
            tts_start = time.perf_counter()
            recorded_first = False

            for i, sentence in enumerate(sentences):
                if cancel_event.is_set():
                    break
                try:
                    audio_bytes = await self._synthesise(sentence)
                except Exception as exc:
                    logger.warning("TTS sentence %d failed: %s", i + 1, exc)
                    if not recorded_first and not first_byte.done():
                        first_byte.set_result(0.0)
                    continue

                if cancel_event.is_set():
                    break

                if not recorded_first:
                    latency = (time.perf_counter() - tts_start) * 1000
                    if not first_byte.done():
                        first_byte.set_result(latency)
                    recorded_first = True

                logger.debug(
                    "TTS sentence %d/%d ready (%.0fms): '%s...'",
                    i + 1,
                    len(sentences),
                    (time.perf_counter() - tts_start) * 1000,
                    sentence[:40],
                )
                await audio_queue.put(audio_bytes)

            await audio_queue.put(None)  # sentinel
            if not first_byte.done():
                first_byte.set_result(0.0)

        asyncio.create_task(_produce())

        # Wait for first sentence (or timeout) before returning so the caller
        # knows audio is ready to play without an empty-queue spin.
        try:
            fb_latency = await asyncio.wait_for(asyncio.shield(first_byte), timeout=15.0)
        except asyncio.TimeoutError:
            fb_latency = 0.0

        return audio_queue, fb_latency

    async def _synthesise(self, text: str) -> bytes:
        """
        Call ElevenLabs TTS for a single sentence and return raw PCM bytes.

        :param text: Sentence to synthesise
        :return: PCM int16 bytes at sample_rate
        """
        # AsyncElevenLabs.text_to_speech.convert() returns an async generator,
        # not a coroutine — collect chunks directly.
        audio_stream = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format=self._output_format,
        )
        chunks: list[bytes] = []
        async for chunk in audio_stream:
            if isinstance(chunk, bytes):
                chunks.append(chunk)
        return b"".join(chunks)

    async def close(self) -> None:
        """Release underlying HTTP connections."""
        client = getattr(self._client, "_client", None)
        if client and hasattr(client, "aclose"):
            await client.aclose()
