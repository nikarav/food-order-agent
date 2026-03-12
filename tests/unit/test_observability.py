"""
Unit tests for the observability no-op path.

All tests run without Langfuse installed or configured, verifying that:
- LANGFUSE_ENABLED is False
- propagate_attributes is a no-op context manager
- Context managers from _NoopLangfuse don't raise
- flush/shutdown don't crash
"""

import sys
from unittest.mock import patch


def _reload_observability(**env_overrides):
    """Reload the observability module with the given env overrides."""
    sys.modules.pop("orderbot.utils.observability", None)

    with patch.dict("os.environ", env_overrides, clear=False):
        import orderbot.utils.observability as obs

    return obs


class TestNoOpPath:
    """Tests for the fallback path (no Langfuse keys set)."""

    def test_langfuse_enabled_is_false_without_keys(self):
        obs = _reload_observability(
            LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY=""
        )
        assert obs.LANGFUSE_ENABLED is False

    def test_propagate_attributes_noop_does_not_raise(self):
        obs = _reload_observability(
            LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY=""
        )
        with obs.propagate_attributes(session_id="test-session"):
            pass  # should not raise

    def test_noop_client_start_as_current_observation_does_not_raise(self):
        obs = _reload_observability(
            LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY=""
        )
        client = obs.get_langfuse_client()
        with client.start_as_current_observation(
            name="gemini_generate", as_type="generation", model="test-model"
        ) as gen:
            gen.update(
                usage_details={"input": 10, "output": 5, "total": 15}
            )

    def test_noop_client_nested_observations_do_not_raise(self):
        obs = _reload_observability(
            LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY=""
        )
        client = obs.get_langfuse_client()
        with client.start_as_current_observation(name="parent", as_type="span"):
            with client.start_as_current_observation(name="child", as_type="generation") as gen:
                gen.update(output="hello")

    def test_flush_langfuse_does_not_raise_when_disabled(self):
        obs = _reload_observability(
            LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY=""
        )
        obs.flush_langfuse()

    def test_shutdown_langfuse_does_not_raise_when_disabled(self):
        obs = _reload_observability(
            LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY=""
        )
        obs.shutdown_langfuse()

    def test_get_langfuse_client_returns_noop_when_disabled(self):
        obs = _reload_observability(
            LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY=""
        )
        client = obs.get_langfuse_client()
        assert hasattr(client, "start_as_current_observation")
        assert hasattr(client, "flush")
        assert hasattr(client, "shutdown")

    def test_noop_enabled_flag_is_false_when_langfuse_not_importable(self):
        with patch.dict("sys.modules", {"langfuse": None}):
            obs = _reload_observability(
                LANGFUSE_PUBLIC_KEY="pk-test", LANGFUSE_SECRET_KEY="sk-test"
            )
        assert obs.LANGFUSE_ENABLED is False
