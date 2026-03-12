"""Per-turn latency metrics for the voice pipeline."""

import logging

import structlog

from orderbot.voice.models import TurnTimings

logger = logging.getLogger(__name__)


class VoiceMetrics:
    """
    Records per-turn latency breakdowns and prints a session summary on exit.

    Logged fields per turn (via structlog):
      - stt_ms     : audio end → transcription received
      - agent_ms   : agent.send() call duration
      - tts_first_byte_ms : text sent → first audio chunk from TTS
      - total_ms   : user stops speaking → first audio playback begins
      - interrupted: whether the user cut off the response
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger("voice.metrics")
        self._turns: list[TurnTimings] = []

    def record(self, timings: TurnTimings) -> None:
        """
        Record and log metrics for a completed turn.

        :param timings: Populated TurnTimings from the session
        """
        self._turns.append(timings)
        self._log.info(
            "voice_turn",
            turn=timings.turn_id,
            stt_ms=round(timings.stt_latency_ms),
            agent_ms=round(timings.agent_latency_ms),
            tts_first_byte_ms=round(timings.tts_first_byte_ms),
            total_ms=round(timings.total_latency_ms),
            interrupted=timings.was_interrupted,
            transcript=timings.transcript[:60],
            response_chars=timings.response_chars,
        )

    def summary(self) -> dict:
        """Return aggregate statistics for the session."""
        if not self._turns:
            return {"turns": 0}

        def _p(values: list[float], pct: int) -> float:
            s = sorted(values)
            return s[min(int(len(s) * pct / 100), len(s) - 1)]

        totals = [t.total_latency_ms for t in self._turns]
        stt = [t.stt_latency_ms for t in self._turns]
        agent = [t.agent_latency_ms for t in self._turns]
        tts_fb = [t.tts_first_byte_ms for t in self._turns]
        n = len(self._turns)

        return {
            "turns": n,
            "interruptions": sum(1 for t in self._turns if t.was_interrupted),
            "avg_total_ms": round(sum(totals) / n),
            "p95_total_ms": round(_p(totals, 95)),
            "avg_stt_ms": round(sum(stt) / n),
            "avg_agent_ms": round(sum(agent) / n),
            "avg_tts_first_byte_ms": round(sum(tts_fb) / n),
        }

    def print_summary(self) -> None:
        """Print a formatted latency report to the console."""
        s = self.summary()
        if s["turns"] == 0:
            return
        lines = [
            "",
            "─── Voice Session Metrics ───",
            f"  Turns          : {s['turns']}",
            f"  Interruptions  : {s['interruptions']}",
            f"  Avg total      : {s['avg_total_ms']} ms",
            f"  P95 total      : {s['p95_total_ms']} ms",
            f"  Avg STT        : {s['avg_stt_ms']} ms",
            f"  Avg agent      : {s['avg_agent_ms']} ms",
            f"  Avg TTS (1st)  : {s['avg_tts_first_byte_ms']} ms",
            "─────────────────────────────",
        ]
        logger.info("\n".join(lines))
