"""
Microphone capture using sounddevice.

Pushes fixed-size float32 frames into an asyncio.Queue for consumption
by the VoiceSession event loop. Frame size matches webrtcvad requirements
(320 samples = 20ms at 16kHz).

The sounddevice callback runs in a dedicated audio thread; frames are
handed off to the async world via loop.call_soon_threadsafe().
"""

import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_CHANNELS = 1
_BLOCK_SIZE = 320   # 20ms at 16kHz — must match webrtcvad frame size
_DTYPE = "float32"


class AudioCapture:
    """
    Streams microphone audio into an asyncio.Queue as numpy float32 frames.

    Usage::

        capture = AudioCapture(loop)
        capture.start()
        async for chunk in capture:   # or read from capture.queue directly
            ...
        capture.stop()
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, sample_rate: int = _SAMPLE_RATE):
        """
        :param loop: The running event loop (needed to bridge the audio thread)
        :param sample_rate: Audio sample rate — must be 16000 for webrtcvad compatibility
        """
        self._loop = loop
        self._sample_rate = sample_rate
        # maxsize ~3s of back-pressure before the audio thread blocks
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=150)
        self._stream = None

    @property
    def queue(self) -> asyncio.Queue:
        return self._queue

    def start(self) -> None:
        """Open and start the microphone input stream."""
        import sounddevice as sd

        if self._stream is not None:
            return

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.warning("AudioCapture status: %s", status)
            # indata shape: (BLOCK_SIZE, 1) — squeeze to 1-D and copy
            chunk = indata[:, 0].copy()
            self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=_CHANNELS,
            dtype=_DTYPE,
            blocksize=_BLOCK_SIZE,
            callback=_callback,
        )
        self._stream.start()
        logger.info("Microphone started (rate=%d, blocksize=%d)", self._sample_rate, _BLOCK_SIZE)

    def stop(self) -> None:
        """Stop and close the microphone stream, sending a sentinel to the queue."""
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as e:
            logger.debug("AudioCapture stop error: %s", e)
        finally:
            self._stream = None
            # Sentinel unblocks any awaiting consumer
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            logger.info("Microphone stopped")

    @staticmethod
    def check_microphone() -> bool:
        """
        Return True if at least one input device is available.

        :return: True if a microphone is found
        """
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            if isinstance(devices, dict):
                devices = [devices]
            return any(d.get("max_input_channels", 0) > 0 for d in devices)
        except Exception as e:
            logger.warning("Microphone check failed: %s", e)
            return False
