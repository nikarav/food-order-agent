"""
Langfuse observability integration with graceful degradation.

If langfuse is installed AND LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set,
real Langfuse tracing is enabled. Otherwise, all exports are no-ops so the rest
of the codebase works without any changes.

Exports:
    LANGFUSE_ENABLED        — bool, True when Langfuse is active
    propagate_attributes    — context manager to set session_id, user_id, etc.
    get_langfuse_client()   — real Langfuse instance or _NoopLangfuse
    flush_langfuse()        — flush pending spans (no-op when disabled)
    shutdown_langfuse()     — flush + shutdown (no-op when disabled)
"""

import contextlib
import logging
import os

logger = logging.getLogger(__name__)

LANGFUSE_ENABLED = False
_client = None


def _init() -> None:
    global LANGFUSE_ENABLED, _client

    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        logger.debug("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — observability disabled")
        return

    try:
        from langfuse import Langfuse

        _client = Langfuse()
        LANGFUSE_ENABLED = True
        logger.info("Langfuse observability enabled")
    except ImportError:
        logger.debug("langfuse not installed — observability disabled")
    except Exception as exc:
        logger.warning("Failed to initialize Langfuse: %s", exc)


_init()

# ---------------------------------------------------------------------------
# propagate_attributes — sets session_id / user_id / tags on traces
# ---------------------------------------------------------------------------

if LANGFUSE_ENABLED:
    from langfuse import propagate_attributes  # type: ignore[assignment]
else:

    @contextlib.contextmanager
    def propagate_attributes(**kwargs):  # type: ignore[misc]
        """No-op propagate_attributes context manager."""
        yield


# ---------------------------------------------------------------------------
# No-op span / observation helpers
# ---------------------------------------------------------------------------


class _NoopObservation:
    def update(self, **kwargs) -> None:
        pass


@contextlib.contextmanager
def _noop_ctx(*args, **kwargs):
    yield _NoopObservation()


class _NoopLangfuse:
    def start_as_current_observation(self, *args, **kwargs):
        return _noop_ctx()

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_langfuse_client():
    """Return the real Langfuse client, or a no-op object if disabled."""
    return _client if LANGFUSE_ENABLED else _NoopLangfuse()


def flush_langfuse() -> None:
    """Flush pending Langfuse spans. No-op when disabled."""
    if LANGFUSE_ENABLED and _client:
        _client.flush()


def shutdown_langfuse() -> None:
    """Flush + shut down Langfuse. No-op when disabled."""
    if LANGFUSE_ENABLED and _client:
        _client.shutdown()
