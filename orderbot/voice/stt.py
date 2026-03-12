"""
ElevenLabs Speech-to-Text client.

Accepts a list of raw float32 audio frames, encodes them as a WAV file
in-memory, and sends to the ElevenLabs Scribe API for transcription.
Returns the transcript and the round-trip latency in milliseconds.
"""

import io
import logging
import time
import wave

import numpy as np

logger = logging.getLogger(__name__)

class ElevenLabsSTT:
    """
    Async wrapper for ElevenLabs speech-to-text.

    :param api_key: ElevenLabs API key
    :param model_id: Transcription model
    :param language_code: Language for transcription
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "scribe_v1",
        language_code: str = "en",
    ):
        from elevenlabs.client import AsyncElevenLabs

        self._client = AsyncElevenLabs(api_key=api_key)
        self._model_id = model_id
        self._language_code = language_code

    async def transcribe(
        self, audio_frames: list[np.ndarray], sample_rate: int = 16000
    ) -> tuple[str, float]:
        """
        Transcribe a list of float32 audio frames to text.

        :param audio_frames: Frames captured from AudioCapture (float32, values in [-1, 1])
        :param sample_rate: Sample rate of the audio (must be 16000)
        :return: (transcript, latency_ms)
        """
        start = time.perf_counter()

        wav_bytes = _frames_to_wav(audio_frames, sample_rate)

        result = await self._client.speech_to_text.convert(
            file=("audio.wav", wav_bytes, "audio/wav"),
            model_id=self._model_id,
            language_code=self._language_code,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        text = (result.text or "").strip()
        logger.debug("STT: '%s' (%.0fms)", text[:80], latency_ms)
        return text, latency_ms

    async def close(self) -> None:
        """Release underlying HTTP connections."""
        client = getattr(self._client, "_client", None)
        if client and hasattr(client, "aclose"):
            await client.aclose()


def _frames_to_wav(frames: list[np.ndarray], sample_rate: int) -> bytes:
    """Convert float32 PCM frames to a WAV byte string (int16)."""
    audio = np.concatenate(frames)
    pcm_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()
