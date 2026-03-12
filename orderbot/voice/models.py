"""Pydantic models for voice pipeline configuration and state."""

from enum import Enum

from pydantic import BaseModel, Field


class VoiceState(str, Enum):
    """State machine states for the voice session."""

    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    SHUTDOWN = "shutdown"


class VADConfig(BaseModel):
    """webrtcvad configuration."""

    aggressiveness: int = 2
    min_speech_ms: int = 200
    frame_duration_ms: int = 20
    pre_speech_buffer_ms: int = 200


class STTConfig(BaseModel):
    """ElevenLabs STT configuration."""

    model_id: str = "scribe_v1"
    language_code: str = "en"


class TTSConfig(BaseModel):
    """ElevenLabs TTS configuration."""

    sentence_pause_ms: int = 0


class AECConfig(BaseModel):
    """Acoustic Echo Cancellation configuration."""

    stream_delay_ms: int = 40


class VoiceConfig(BaseModel):
    """Full voice pipeline configuration loaded from agent.yaml."""

    elevenlabs_api_key: str
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    model_id: str = "eleven_flash_v2_5"
    output_format: str = "pcm_16000"
    sample_rate: int = 16000
    silence_threshold_ms: int = 700
    enable_barge_in: bool = True
    post_playback_cooldown_ms: int = 300
    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    aec: AECConfig = Field(default_factory=AECConfig)


class TurnTimings(BaseModel):
    """Per-turn latency breakdown for diagnostics."""

    turn_id: int = 0
    stt_latency_ms: float = 0.0       # Audio end → transcription received
    agent_latency_ms: float = 0.0     # agent.send() call duration
    tts_first_byte_ms: float = 0.0    # Text sent → first audio chunk from TTS
    total_latency_ms: float = 0.0     # User stops speaking → first audio playback
    was_interrupted: bool = False
    transcript: str = ""
    response_chars: int = 0
