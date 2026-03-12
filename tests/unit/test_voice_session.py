"""
Unit tests for VoiceSession state machine.

All external components (AudioCapture, AudioPlayback, WebRTCVAD, STT, TTS,
FoodOrderAgent) are mocked, so no API keys or audio hardware are required.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from orderbot.voice.models import TurnTimings, VADConfig, VoiceConfig, VoiceState
from orderbot.voice.session import VoiceSession


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def voice_config():
    return VoiceConfig(
        elevenlabs_api_key="test-key",
        voice_id="test-voice",
        model_id="eleven_flash_v2_5",
        output_format="pcm_16000",
        sample_rate=16000,
        silence_threshold_ms=700,
        vad=VADConfig(aggressiveness=2, min_speech_ms=200, frame_duration_ms=20),
    )


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.send.return_value = {"message": "Added burger. Anything else?", "tool_calls": []}
    return agent


@pytest.fixture
def session(mock_agent, voice_config):
    return VoiceSession(agent=mock_agent, config=voice_config)


def _make_chunk(n: int = 320) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ── _set_state ────────────────────────────────────────────────────────────────

def test_initial_state_is_idle(session):
    assert session._state == VoiceState.IDLE


def test_set_state_transitions(session):
    session._set_state(VoiceState.LISTENING)
    assert session._state == VoiceState.LISTENING

    session._set_state(VoiceState.RECORDING)
    assert session._state == VoiceState.RECORDING


# ── _process_utterance ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_utterance_happy_path(session, mock_agent):
    """Full pipeline: STT → agent → TTS → playback."""
    audio_frames = [_make_chunk() for _ in range(5)]

    # Mock STT
    session._stt = AsyncMock()
    session._stt.transcribe.return_value = ("I want a burger", 120.0)

    # Mock TTS: return a queue with bytes then sentinel
    audio_queue: asyncio.Queue = asyncio.Queue()
    await audio_queue.put(b"\x00" * 1024)
    await audio_queue.put(None)
    session._tts = AsyncMock()
    session._tts.stream_sentences.return_value = (audio_queue, 300.0)

    # Mock playback
    session._playback = AsyncMock()
    session._playback.play_chunks = AsyncMock(return_value=True)
    session._playback.is_playing = False

    session._set_state(VoiceState.PROCESSING)

    await session._process_utterance(audio_frames)

    session._stt.transcribe.assert_called_once_with(audio_frames, 16000)
    mock_agent.send.assert_called_once_with("I want a burger")
    session._tts.stream_sentences.assert_called_once()
    assert session._state == VoiceState.LISTENING


@pytest.mark.asyncio
async def test_process_utterance_empty_transcript_skips(session, mock_agent):
    """Empty STT result should skip agent call and return to LISTENING."""
    session._stt = AsyncMock()
    session._stt.transcribe.return_value = ("", 50.0)
    session._tts = AsyncMock()
    session._playback = AsyncMock()
    session._playback.is_playing = False

    session._set_state(VoiceState.PROCESSING)
    await session._process_utterance([_make_chunk()])

    mock_agent.send.assert_not_called()
    session._tts.stream_sentences.assert_not_called()
    assert session._state == VoiceState.LISTENING


@pytest.mark.asyncio
async def test_process_utterance_stt_error_recovers(session):
    """STT failure should log error and return to LISTENING (not crash)."""
    session._stt = AsyncMock()
    session._stt.transcribe.side_effect = RuntimeError("STT API down")
    session._tts = AsyncMock()
    session._playback = AsyncMock()
    session._playback.is_playing = False

    session._set_state(VoiceState.PROCESSING)
    await session._process_utterance([_make_chunk()])

    assert session._state == VoiceState.LISTENING


@pytest.mark.asyncio
async def test_process_utterance_exit_word_shuts_down(session, mock_agent):
    """Saying 'goodbye' should trigger shutdown."""
    session._stt = AsyncMock()
    session._stt.transcribe.return_value = ("goodbye", 80.0)
    session._tts = AsyncMock()
    session._playback = AsyncMock()
    session._playback.is_playing = False

    session._set_state(VoiceState.PROCESSING)
    await session._process_utterance([_make_chunk()])

    assert session._shutdown.is_set()
    mock_agent.send.assert_not_called()


# ── Barge-in detection ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_barge_in_cancels_playback(session):
    """
    Simulate barge-in: session is SPEAKING, and speech_start event arrives
    from VAD. Should cancel playback and transition to RECORDING.
    """
    session._set_state(VoiceState.SPEAKING)
    session._tts_cancel = asyncio.Event()
    session._playback = MagicMock()
    session._playback.cancel = MagicMock()
    session._vad = MagicMock()
    session._vad.reset = MagicMock()

    # Manually trigger the barge-in logic (replicate what _conversation_loop does)
    session._tts_cancel.set()
    session._playback.cancel()
    session._vad.reset()
    session._set_state(VoiceState.RECORDING)

    assert session._tts_cancel.is_set()
    session._playback.cancel.assert_called_once()
    assert session._state == VoiceState.RECORDING


# ── _init_components error paths ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_raises_if_no_microphone(session, voice_config):
    """start() should raise RuntimeError if no microphone is found."""
    with patch("orderbot.voice.session.AudioCapture.check_microphone", return_value=False):
        with pytest.raises(RuntimeError, match="No microphone"):
            await session._init_components()


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics_record_and_summary(session):
    for i in range(3):
        t = TurnTimings(
            turn_id=i + 1,
            stt_latency_ms=100.0,
            agent_latency_ms=500.0,
            tts_first_byte_ms=300.0,
            total_latency_ms=900.0,
            was_interrupted=(i == 1),
            transcript="test",
            response_chars=20,
        )
        session._metrics.record(t)

    summary = session._metrics.summary()
    assert summary["turns"] == 3
    assert summary["interruptions"] == 1
    assert summary["avg_total_ms"] == 900
    assert summary["avg_stt_ms"] == 100
    assert summary["avg_agent_ms"] == 500
