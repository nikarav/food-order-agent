"""
Unit tests for WebRTCVAD.

webrtcvad is mocked so no real audio hardware is required.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from orderbot.voice.models import VADConfig


@pytest.fixture
def vad():
    """WebRTCVAD with a mocked webrtcvad module — no native lib required."""
    mock_vad_instance = MagicMock()
    mock_webrtcvad = MagicMock()
    mock_webrtcvad.Vad.return_value = mock_vad_instance

    # Inject the fake module before WebRTCVAD's lazy `import webrtcvad` runs
    original = sys.modules.get("webrtcvad")
    sys.modules["webrtcvad"] = mock_webrtcvad
    try:
        from orderbot.voice.vad import WebRTCVAD

        config = VADConfig(aggressiveness=2, min_speech_ms=40, frame_duration_ms=20)
        v = WebRTCVAD(config=config, silence_threshold_ms=60)
        v._vad = mock_vad_instance  # ensure the exact mock instance is used
        yield v, mock_vad_instance
    finally:
        if original is None:
            sys.modules.pop("webrtcvad", None)
        else:
            sys.modules["webrtcvad"] = original


def _chunk() -> np.ndarray:
    """Create a zero-filled 320-sample float32 chunk."""
    return np.zeros(320, dtype=np.float32)


# ── Silence → no event ───────────────────────────────────────────────────────

def test_silence_returns_none(vad):
    v, mock = vad
    mock.is_speech.return_value = False
    assert v.process_chunk(_chunk()) is None


# ── Speech start fires after min_speech_frames ───────────────────────────────

def test_speech_start_requires_min_frames(vad):
    """speech_start must not fire on the very first speech frame."""
    v, mock = vad
    mock.is_speech.return_value = True

    # min_speech_ms=40, frame_duration_ms=20 → need 2 consecutive frames
    result_first = v.process_chunk(_chunk())
    assert result_first is None  # 1st frame: not enough yet

    result_second = v.process_chunk(_chunk())
    assert result_second is not None
    assert result_second["event"] == "speech_start"
    assert v.is_speaking is True


def test_speech_start_includes_pre_speech(vad):
    """speech_start event carries pre-speech buffer frames."""
    v, mock = vad
    # Prime the pre-speech buffer with silence frames
    mock.is_speech.return_value = False
    for _ in range(3):
        v.process_chunk(_chunk())

    # Now speech starts
    mock.is_speech.return_value = True
    v.process_chunk(_chunk())
    event = v.process_chunk(_chunk())

    assert event is not None
    assert event["event"] == "speech_start"
    assert isinstance(event["pre_speech"], list)
    assert len(event["pre_speech"]) > 0


# ── Speech end fires after silence_threshold_frames ──────────────────────────

def test_speech_end_after_silence_threshold(vad):
    v, mock = vad
    # Get into speaking state
    mock.is_speech.return_value = True
    v.process_chunk(_chunk())
    v.process_chunk(_chunk())  # speech_start fires here

    # silence_threshold_ms=60, frame_duration_ms=20 → need 3 silence frames
    mock.is_speech.return_value = False
    v.process_chunk(_chunk())  # silence 1
    v.process_chunk(_chunk())  # silence 2
    event = v.process_chunk(_chunk())  # silence 3 → speech_end

    assert event is not None
    assert event["event"] == "speech_end"
    assert v.is_speaking is False


def test_brief_silence_does_not_end_speech(vad):
    """Two silence frames when threshold is 3 should not fire speech_end."""
    v, mock = vad
    mock.is_speech.return_value = True
    v.process_chunk(_chunk())
    v.process_chunk(_chunk())  # speech_start

    mock.is_speech.return_value = False
    v.process_chunk(_chunk())  # silence 1
    event = v.process_chunk(_chunk())  # silence 2 (not enough)
    assert event is None


def test_speech_resumes_resets_silence_counter(vad):
    """If speech resumes during silence, the silence counter resets."""
    v, mock = vad
    mock.is_speech.return_value = True
    v.process_chunk(_chunk())
    v.process_chunk(_chunk())  # speech_start

    # Two silence frames
    mock.is_speech.return_value = False
    v.process_chunk(_chunk())
    v.process_chunk(_chunk())

    # Speech resumes — should NOT trigger speech_end after next silence
    mock.is_speech.return_value = True
    v.process_chunk(_chunk())

    mock.is_speech.return_value = False
    v.process_chunk(_chunk())  # 1 silence after resume — not enough
    event = v.process_chunk(_chunk())  # 2 silences — still not enough
    assert event is None


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_reset_clears_state(vad):
    v, mock = vad
    mock.is_speech.return_value = True
    v.process_chunk(_chunk())
    v.process_chunk(_chunk())  # speech_start → is_speaking = True

    v.reset()
    assert v.is_speaking is False


def test_no_speech_after_single_cough(vad):
    """A single speech frame below min_speech_frames should not trigger speech_start."""
    v, mock = vad
    mock.is_speech.return_value = True
    event = v.process_chunk(_chunk())  # only 1 frame (need 2)
    assert event is None

    mock.is_speech.return_value = False
    event = v.process_chunk(_chunk())  # silence resets counter
    assert event is None
    assert not v.is_speaking
