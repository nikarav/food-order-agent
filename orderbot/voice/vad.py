"""
Voice Activity Detection using Google's webrtcvad.

Processes fixed-size PCM frames and emits speech_start / speech_end events.
webrtcvad requires: 16kHz sample rate, 20ms frames (320 samples = 640 bytes int16).
"""

import logging
from collections import deque

import numpy as np

from orderbot.voice.models import VADConfig

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class WebRTCVAD:
    """
    Thin wrapper around webrtcvad.Vad that adds hysteresis and event emission.

    Hysteresis prevents noisy false positives:
    - speech_start fires only after min_speech_frames consecutive speech frames
    - speech_end fires only after silence_threshold_frames consecutive silence frames
    - A pre-speech ring buffer captures ~200ms before onset so STT never misses the start
    """

    def __init__(self, config: VADConfig | None = None, silence_threshold_ms: int = 700):
        """
        :param config: VAD configuration
        :param silence_threshold_ms: Silence duration before triggering speech_end
        """
        import webrtcvad

        self._config = config or VADConfig()
        self._vad = webrtcvad.Vad(self._config.aggressiveness)

        frame_ms = self._config.frame_duration_ms
        self._samples_per_frame = int(SAMPLE_RATE * frame_ms / 1000)  # 320 for 20ms

        # Hysteresis thresholds (in frames)
        self._min_speech_frames = max(1, self._config.min_speech_ms // frame_ms)
        self._silence_threshold_frames = max(1, silence_threshold_ms // frame_ms)

        # State
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0

        pre_speech_frames = max(1, self._config.pre_speech_buffer_ms // frame_ms)
        self._pre_speech: deque[np.ndarray] = deque(maxlen=pre_speech_frames)

    def process_chunk(self, audio_chunk: np.ndarray) -> dict | None:
        """
        Process one audio frame and return a VAD event if state changes.

        :param audio_chunk: float32 array of exactly samples_per_frame samples, values in [-1, 1]
        :return: {"event": "speech_start", "pre_speech": list[np.ndarray]} |
                 {"event": "speech_end"} | None
        """
        # Convert float32 → int16 bytes for webrtcvad
        pcm_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        buf = pcm_int16.tobytes()

        try:
            is_speech = self._vad.is_speech(buf, SAMPLE_RATE)
        except Exception as e:
            logger.debug("webrtcvad error (likely frame size mismatch): %s", e)
            return None

        if not self._is_speaking:
            self._pre_speech.append(audio_chunk.copy())

            if is_speech:
                self._speech_frames += 1
                if self._speech_frames >= self._min_speech_frames:
                    self._is_speaking = True
                    self._silence_frames = 0
                    pre = list(self._pre_speech)
                    self._pre_speech.clear()
                    logger.debug("VAD: speech_start")
                    return {"event": "speech_start", "pre_speech": pre}
            else:
                self._speech_frames = 0

        else:
            if not is_speech:
                self._silence_frames += 1
                if self._silence_frames >= self._silence_threshold_frames:
                    self._is_speaking = False
                    self._speech_frames = 0
                    self._silence_frames = 0
                    logger.debug("VAD: speech_end")
                    return {"event": "speech_end"}
            else:
                self._silence_frames = 0

        return None

    def reset(self) -> None:
        """Reset state between utterances."""
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._pre_speech.clear()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def samples_per_frame(self) -> int:
        return self._samples_per_frame
