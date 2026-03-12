"""
VoiceSession — top-level orchestrator for the voice pipeline.

State machine::

    IDLE
      │ start()
      ▼
    LISTENING  ──(speech_start)──► RECORDING  ──(speech_end)──► PROCESSING
      ▲                                ▲                              │
      │                                │ (barge-in during SPEAKING)   │ STT + agent.send()
      │                                │                              ▼
      └────────────────────────────────┴──────────────────────── SPEAKING
                                                                      │
                                                           (playback done or cancelled)
                                                                      │
                                                                  LISTENING

Key design decisions:
- The main loop runs continuously, reading mic frames and running VAD. It never
  blocks for more than one frame (32ms) — so barge-in detection is always live.
- _process_utterance() is launched as an asyncio.Task so the main loop keeps
  running (and detecting barge-ins) while STT / agent / TTS execute.
- agent.send() is synchronous and uses its own event loop internally. It is
  run via asyncio.to_thread() to avoid blocking the voice event loop.
- TTS uses sentence-level streaming: the first sentence's audio arrives in
  the queue before the rest are synthesised, cutting perceived latency.
"""

import asyncio
import logging
import re
import time

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from orderbot.voice.aec import EchoCanceller
from orderbot.voice.audio_capture import AudioCapture
from orderbot.voice.audio_playback import AudioPlayback
from orderbot.voice.metrics import VoiceMetrics
from orderbot.voice.models import TurnTimings, VoiceConfig, VoiceState
from orderbot.voice.stt import ElevenLabsSTT
from orderbot.voice.tts import ElevenLabsTTS
from orderbot.voice.vad import WebRTCVAD

logger = logging.getLogger(__name__)

_EXIT_WORDS = {"goodbye", "bye", "exit", "quit"}

# STT transcribes ambient noise as parenthetical descriptions like "(footsteps)"
_NON_SPEECH_RE = re.compile(r"^[\s\(\[]*[\(\[].*[\)\]][\s\)\]]*$")

_STATE_LABELS = {
    VoiceState.LISTENING: "[dim]Listening...[/dim]",
    VoiceState.RECORDING: "[bold red]● Recording[/bold red]",
    VoiceState.PROCESSING: "[bold yellow]⟳ Thinking...[/bold yellow]",
    VoiceState.SPEAKING: "[bold green]▶ Speaking[/bold green]",
}


class VoiceSession:
    """
    Orchestrates mic → VAD → STT → agent → TTS → speaker.

    :param agent: Existing FoodOrderAgent (accessed via agent.send())
    :param config: Voice pipeline configuration
    """

    def __init__(self, agent, config: VoiceConfig) -> None:
        """
        Initialize the VoiceSession.

        :param agent: The agent
        :param config: The configuration
        """
        self._agent = agent
        self._config = config
        self._console = Console()

        self._state = VoiceState.IDLE
        self._turn_count = 0
        self._metrics = VoiceMetrics()

        # Cancellation primitives
        self._tts_cancel = asyncio.Event()
        self._shutdown = asyncio.Event()

        # Components — initialised in start()
        self._capture: AudioCapture | None = None
        self._playback: AudioPlayback | None = None
        self._vad: WebRTCVAD | None = None
        self._stt: ElevenLabsSTT | None = None
        self._tts: ElevenLabsTTS | None = None
        self._aec: EchoCanceller | None = None
        self._ref_queue: asyncio.Queue | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Initialise all components and run the conversation loop.

        Raises RuntimeError if a required component cannot be initialised
        (caller should catch and fall back to text mode).
        """
        self._console.print(Panel.fit("🎙  Food Order Assistant — Voice Mode", border_style="cyan"))

        await self._init_components()

        self._console.print(
            "[green]Voice mode active.[/green] "
            "Speak naturally — say [bold]'goodbye'[/bold] to exit.\n"
        )

        try:
            self._capture.start()
            self._set_state(VoiceState.LISTENING)
            await self._conversation_loop()
        except KeyboardInterrupt:
            pass
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------ #
    # Initialisation & cleanup                                             #
    # ------------------------------------------------------------------ #

    async def _init_components(self) -> None:
        """Initialise all voice components; raise on failure."""
        if not AudioCapture.check_microphone():
            raise RuntimeError("No microphone detected")

        loop = asyncio.get_running_loop()
        self._capture = AudioCapture(loop, self._config.sample_rate)
        self._playback = AudioPlayback(self._config.sample_rate)

        if self._config.enable_barge_in:
            self._aec = EchoCanceller(stream_delay_ms=self._config.aec.stream_delay_ms)
            self._ref_queue = self._playback.enable_reference(loop)

        self._vad = WebRTCVAD(
            config=self._config.vad,
            silence_threshold_ms=self._config.silence_threshold_ms,
        )

        self._stt = ElevenLabsSTT(
            api_key=self._config.elevenlabs_api_key,
            model_id=self._config.stt.model_id,
            language_code=self._config.stt.language_code,
        )
        self._tts = ElevenLabsTTS(
            api_key=self._config.elevenlabs_api_key,
            voice_id=self._config.voice_id,
            model_id=self._config.model_id,
            output_format=self._config.output_format,
        )

    async def _cleanup(self) -> None:
        """Graceful shutdown: stop mic, cancel playback, close API clients."""
        self._set_state(VoiceState.SHUTDOWN)
        if self._capture:
            self._capture.stop()
        if self._playback and self._playback.is_playing:
            self._playback.cancel()
        if self._stt:
            await self._stt.close()
        if self._tts:
            await self._tts.close()
        self._metrics.print_summary()

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    async def _conversation_loop(self) -> None:
        """
        Read mic frames, run VAD, manage state transitions.

        Runs continuously — never blocks for more than one frame so barge-in
        detection stays live even while _process_utterance() is running.

        During SPEAKING state with barge-in enabled, each mic frame is run
        through the echo canceller (using the speaker reference signal) before
        being fed to VAD. This removes the TTS echo so VAD only fires on
        actual user speech.
        """
        audio_buffer: list[np.ndarray] = []

        while not self._shutdown.is_set():
            try:
                chunk = await asyncio.wait_for(self._capture.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if chunk is None:
                break

            vad_chunk = chunk

            if self._state == VoiceState.SPEAKING:
                if not self._config.enable_barge_in:
                    continue

                # Apply AEC: subtract speaker echo from mic signal
                ref = self._get_reference_frame()
                if self._aec is not None and ref is not None:
                    vad_chunk = self._aec.process(chunk, ref)

                vad_event = self._vad.process_chunk(vad_chunk)

                if vad_event and vad_event["event"] == "speech_start":
                    logger.info("Barge-in detected (AEC-cleaned)")
                    self._tts_cancel.set()
                    self._playback.cancel()
                    self._vad.reset()
                    self._set_state(VoiceState.RECORDING)
                    audio_buffer = list(vad_event.get("pre_speech", []))
                    audio_buffer.append(chunk)
                continue

            vad_event = self._vad.process_chunk(vad_chunk)

            if self._state == VoiceState.LISTENING:
                if vad_event and vad_event["event"] == "speech_start":
                    self._set_state(VoiceState.RECORDING)
                    audio_buffer = list(vad_event.get("pre_speech", []))
                    audio_buffer.append(chunk)

            elif self._state == VoiceState.RECORDING:
                audio_buffer.append(chunk)
                if vad_event and vad_event["event"] == "speech_end":
                    self._vad.reset()
                    utterance = list(audio_buffer)
                    audio_buffer = []
                    asyncio.create_task(self._process_utterance(utterance))

    # ------------------------------------------------------------------ #
    # Per-turn pipeline                                                    #
    # ------------------------------------------------------------------ #

    async def _process_utterance(self, audio_frames: list[np.ndarray]) -> None:
        """
        Full pipeline for one utterance: STT → agent → TTS → playback.

        :param audio_frames: Accumulated float32 PCM frames for this utterance
        """
        self._turn_count += 1
        turn_start = time.perf_counter()
        timings = TurnTimings(turn_id=self._turn_count)
        self._set_state(VoiceState.PROCESSING)

        # ── STT ────────────────────────────────────────────────────────
        try:
            transcript, stt_ms = await self._stt.transcribe(audio_frames, self._config.sample_rate)
        except Exception as exc:
            logger.error("STT error: %s", exc)
            self._console.print(f"[red]STT error:[/red] {exc}")
            self._set_state(VoiceState.LISTENING)
            return

        if not transcript:
            logger.debug("Empty transcription — resuming listening")
            self._set_state(VoiceState.LISTENING)
            return

        if _NON_SPEECH_RE.match(transcript):
            logger.debug("Non-speech noise filtered: %s", transcript)
            self._set_state(VoiceState.LISTENING)
            return

        timings.stt_latency_ms = stt_ms
        timings.transcript = transcript
        self._console.print(f"\n[bold blue]You:[/bold blue] {transcript}")

        # Exit command
        if transcript.lower().strip().rstrip(".,!") in _EXIT_WORDS:
            self._console.print(Panel.fit("Goodbye! 👋", border_style="cyan"))
            self._shutdown.set()
            return

        # ── Agent ──────────────────────────────────────────────────────
        agent_start = time.perf_counter()
        try:
            # agent.send() is synchronous (has its own event loop) — run in thread
            response = await asyncio.to_thread(self._agent.send, transcript)
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            self._console.print(f"[red]Agent error:[/red] {exc}")
            self._set_state(VoiceState.LISTENING)
            return

        timings.agent_latency_ms = (time.perf_counter() - agent_start) * 1000
        response_text = response.get("message", "")
        timings.response_chars = len(response_text)

        self._console.print(
            Panel(response_text, title="[green]Assistant[/green]", border_style="green")
        )
        self._console.print(Rule(style="dim"))

        if not response_text.strip():
            self._set_state(VoiceState.LISTENING)
            return

        # ── TTS + Playback ─────────────────────────────────────────────
        self._set_state(VoiceState.SPEAKING)
        self._tts_cancel.clear()

        try:
            audio_queue, tts_fb_ms = await self._tts.stream_sentences(
                response_text, self._tts_cancel
            )
        except Exception as exc:
            logger.error("TTS error: %s", exc)
            self._console.print(f"[yellow]TTS error (text shown above):[/yellow] {exc}")
            self._set_state(VoiceState.LISTENING)
            return

        timings.tts_first_byte_ms = tts_fb_ms
        timings.total_latency_ms = (time.perf_counter() - turn_start) * 1000

        async def _audio_iter():
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                yield chunk

        try:
            completed = await self._playback.play_chunks(_audio_iter())
            timings.was_interrupted = not completed
        except Exception as exc:
            logger.error("Playback error: %s", exc)
            timings.was_interrupted = True

        self._metrics.record(timings)

        # Transition back unless barge-in already moved us to RECORDING
        if self._state == VoiceState.SPEAKING:
            await self._post_playback_cooldown()
            self._set_state(VoiceState.LISTENING)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _get_reference_frame(self) -> np.ndarray | None:
        """Pop the next speaker reference frame for AEC, or None if unavailable."""
        if self._ref_queue is None:
            return None
        try:
            ref = self._ref_queue.get_nowait()
            return ref  # may be None (sentinel)
        except asyncio.QueueEmpty:
            return None

    async def _post_playback_cooldown(self) -> None:
        """Drain mic buffer and reset VAD/AEC after playback to suppress trailing echo."""
        cooldown_s = self._config.post_playback_cooldown_ms / 1000
        await asyncio.sleep(cooldown_s)
        if self._capture is not None:
            while not self._capture.queue.empty():
                try:
                    self._capture.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        if self._ref_queue is not None:
            while not self._ref_queue.empty():
                try:
                    self._ref_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        if self._vad is not None:
            self._vad.reset()
        if self._aec is not None:
            self._aec.reset()

    def _set_state(self, new_state: VoiceState) -> None:
        old = self._state
        self._state = new_state
        if label := _STATE_LABELS.get(new_state):
            self._console.print(label, end="\r")
        logger.debug("State %s → %s", old.value, new_state.value)
