"""
Acoustic Echo Cancellation using LiveKit's WebRTC AudioProcessingModule.

Wraps livekit.rtc.AudioProcessingModule to provide AEC + noise suppression.
The APM processes 10ms int16 frames; this module handles the conversion
from our 20ms float32 frames (splitting each into two 10ms chunks).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLES_10MS = 160  # 10ms at 16kHz
_SAMPLE_RATE = 16000


class EchoCanceller:
    """
    Removes speaker echo from the microphone signal using WebRTC AEC3.

    :param stream_delay_ms: Estimated delay between speaker output and mic capture
    """

    def __init__(self, stream_delay_ms: int = 40):
        self._delay_ms = stream_delay_ms
        self._apm = self._create_apm()

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        Cancel echo from one 20ms mic frame using the speaker reference frame.

        :param mic: float32 array, 320 samples, values in [-1, 1]
        :param ref: float32 array, 320 samples, values in [-1, 1]
        :return: Echo-cancelled float32 frame
        """
        from livekit.rtc import AudioFrame

        mic_i16 = (np.clip(mic, -1.0, 1.0) * 32767).astype(np.int16)
        ref_i16 = (np.clip(ref, -1.0, 1.0) * 32767).astype(np.int16)

        for offset in (0, _SAMPLES_10MS):
            ref_frame = AudioFrame(
                data=bytearray(ref_i16[offset:offset + _SAMPLES_10MS].tobytes()),
                sample_rate=_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=_SAMPLES_10MS,
            )
            self._apm.process_reverse_stream(ref_frame)

            mic_frame = AudioFrame(
                data=bytearray(mic_i16[offset:offset + _SAMPLES_10MS].tobytes()),
                sample_rate=_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=_SAMPLES_10MS,
            )
            self._apm.process_stream(mic_frame)
            mic_i16[offset:offset + _SAMPLES_10MS] = np.frombuffer(
                mic_frame.data, dtype=np.int16
            )

        return mic_i16.astype(np.float32) / 32768.0

    def reset(self) -> None:
        """Re-create the APM to clear internal state."""
        self._apm = self._create_apm()
        logger.debug("EchoCanceller reset")

    def _create_apm(self):
        from livekit.rtc.apm import AudioProcessingModule

        apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
        )
        apm.set_stream_delay_ms(self._delay_ms)
        return apm
