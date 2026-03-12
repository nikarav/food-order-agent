"""
Speaker output using sounddevice.

Streams PCM int16 audio chunks to the default output device.
Supports immediate cancellation (barge-in) via a threading.Event.

The sounddevice callback runs in an audio thread; the rest of the API
is async-friendly and drives from the event loop.

When a reference queue is provided (via ``enable_reference``), the callback
also pushes played-back audio as float32 20ms chunks into an asyncio.Queue.
This reference signal is used by the echo canceller to subtract speaker
output from the microphone input.
"""

import asyncio
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024       # Frames per sounddevice callback (~64ms at 16kHz)
_REF_CHUNK_SAMPLES = 320  # 20ms at 16kHz — matches mic capture frame size


class AudioPlayback:
    """
    Plays a stream of PCM int16 byte chunks through the speaker.

    Usage::

        playback = AudioPlayback(sample_rate=16000)
        playback.enable_reference(loop)  # for echo cancellation
        completed = await playback.play_chunks(some_async_iterator)
        # completed is False if the user interrupted (barge-in)
    """

    def __init__(self, sample_rate: int = 16000):
        """
        :param sample_rate: Must match TTS output format (pcm_16000 → 16000)
        """
        self._sample_rate = sample_rate
        self._stream = None
        self._cancelled = threading.Event()
        # Ring buffer — audio thread reads from here, async feed writes to it
        self._buf = bytearray()
        self._buf_lock = threading.Lock()
        self._write_pos = 0
        self._feed_done = False

        # Reference signal for echo cancellation (set via enable_reference)
        self._ref_queue: asyncio.Queue[np.ndarray | None] | None = None
        self._ref_loop: asyncio.AbstractEventLoop | None = None
        self._ref_accum = bytearray()  # accumulator for slicing into 20ms chunks

    def enable_reference(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        """
        Enable reference signal output for echo cancellation.

        Returns a queue that will receive float32 numpy arrays (320 samples each)
        matching the mic capture frame size. A None sentinel is pushed when
        playback ends.

        :param loop: The running event loop (for thread-safe queue writes)
        :return: Queue of float32 reference frames
        """
        self._ref_loop = loop
        self._ref_queue = asyncio.Queue(maxsize=150)
        return self._ref_queue

    async def play_chunks(self, chunk_iter) -> bool:
        """
        Feed an async iterator of bytes into the speaker and play them.

        Starts playback immediately as the first chunk arrives; subsequent
        chunks are buffered and played without gaps.

        :param chunk_iter: Async iterator yielding bytes (PCM int16 at sample_rate)
        :return: True if playback finished normally, False if cancelled (barge-in)
        """
        import sounddevice as sd

        self._cancelled.clear()
        self._buf = bytearray()
        self._write_pos = 0
        self._feed_done = False

        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=_BLOCK_SIZE,
            callback=self._callback,
        )

        try:
            self._stream.start()

            # Feed chunks into the buffer as they arrive from TTS
            async for chunk in chunk_iter:
                if self._cancelled.is_set():
                    return False
                if chunk:
                    with self._buf_lock:
                        self._buf.extend(chunk)

            self._feed_done = True

            # Wait until the buffer drains or we're cancelled
            while not self._cancelled.is_set():
                with self._buf_lock:
                    remaining = len(self._buf) - self._write_pos
                if remaining <= 0 and self._feed_done:
                    break
                await asyncio.sleep(0.02)

            return not self._cancelled.is_set()

        finally:
            self._teardown()

    def cancel(self) -> None:
        """Cancel playback immediately. Safe to call from any thread."""
        self._cancelled.set()
        self._teardown()

    @property
    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """
        Sounddevice audio thread callback — fills outdata from the buffer.

        Also slices output into 20ms reference chunks for echo cancellation
        when a reference queue has been enabled.
        """
        if status:
            logger.debug("AudioPlayback status: %s", status)

        bytes_needed = frames * 2  # int16 = 2 bytes per sample
        with self._buf_lock:
            available = len(self._buf) - self._write_pos
            to_read = min(bytes_needed, available)
            if to_read > 0:
                data = bytes(self._buf[self._write_pos : self._write_pos + to_read])
                self._write_pos += to_read
            else:
                data = b""

        # Pad with silence if the buffer is temporarily empty (prevents underrun noise)
        if len(data) < bytes_needed:
            data = data + b"\x00" * (bytes_needed - len(data))

        outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)

        # Push reference signal for echo cancellation
        if self._ref_queue is not None and self._ref_loop is not None:
            self._push_reference(data)

    def _push_reference(self, pcm_bytes: bytes) -> None:
        """Slice played-back PCM into 20ms float32 chunks and enqueue for AEC."""
        self._ref_accum.extend(pcm_bytes)
        chunk_bytes = _REF_CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample

        while len(self._ref_accum) >= chunk_bytes:
            raw = bytes(self._ref_accum[:chunk_bytes])
            del self._ref_accum[:chunk_bytes]
            # Convert int16 → float32 [-1, 1] to match mic frame format
            chunk_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                self._ref_loop.call_soon_threadsafe(self._ref_queue.put_nowait, chunk_f32)
            except (asyncio.QueueFull, RuntimeError):
                pass  # Drop frames under back-pressure rather than blocking audio thread

    def _teardown(self) -> None:
        # Send reference sentinel so AEC consumer knows playback ended
        if self._ref_queue is not None and self._ref_loop is not None:
            try:
                self._ref_loop.call_soon_threadsafe(self._ref_queue.put_nowait, None)
            except (RuntimeError, asyncio.QueueFull):
                pass
            self._ref_accum.clear()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
