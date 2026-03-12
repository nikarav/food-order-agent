"""
Tests for GeminiClient._fallback_response.

This method is the safety net when the LLM returns empty text after a
tool-calling loop.  It must produce a context-aware response based on
the last tool call made, so the user never sees a generic "How can I
help you?" when they asked for the menu, their order, etc.
"""

import pytest
from unittest.mock import patch, MagicMock

from orderbot.llm.gemini import GeminiClient


@pytest.fixture
def gemini_client():
    """Create a GeminiClient with mocked genai client and file read."""
    config = MagicMock()
    config.gemini_api_key = "fake-key"
    config.model_name = "gemini-test"
    config.prompts.system = "prompts/v2/system.txt"

    with patch.object(GeminiClient, "__init__", lambda self, cfg: None):
        client = GeminiClient.__new__(GeminiClient)
    return client


# ---------- no tool calls ----------

def test_no_tool_calls_returns_greeting(gemini_client):
    result = gemini_client._fallback_response([])
    assert result == "How can I help you?"


# ---------- add_item ----------

def test_add_item_fallback(gemini_client):
    tool_calls = [
        {"name": "add_item", "args": {}, "result": {
            "status": "added",
            "item": {"name": "Classic Burger"},
            "order": {},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "Classic Burger" in result
    assert "Added" in result
    assert "Anything else?" in result


def test_add_item_fallback_missing_name(gemini_client):
    tool_calls = [
        {"name": "add_item", "args": {}, "result": {
            "status": "added",
            "item": {},
            "order": {},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "Added" in result
    assert "item" in result  # falls back to "item"


# ---------- modify_item ----------

def test_modify_item_fallback(gemini_client):
    tool_calls = [
        {"name": "modify_item", "args": {}, "result": {
            "status": "modified",
            "item": {"name": "Classic Burger"},
            "order": {},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "Updated" in result
    assert "Anything else?" in result


# ---------- remove_item ----------

def test_remove_item_fallback(gemini_client):
    tool_calls = [
        {"name": "remove_item", "args": {}, "result": {
            "status": "removed",
            "item": {"name": "French Fries"},
            "order": {},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "Removed" in result
    assert "French Fries" in result


# ---------- get_menu (Bug 1 & 3 — the turn-5 failure) ----------

def test_get_menu_fallback_returns_menu_text(gemini_client):
    """Reproduces the turn-5 bug: get_menu was called but response was empty."""
    menu_text = "- classic_burger: Classic Burger ($8.50)\n- fries: French Fries ($3.50)"
    tool_calls = [
        {"name": "get_menu", "args": {}, "result": {
            "status": "show_menu",
            "menu": menu_text,
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert result == menu_text


def test_get_menu_fallback_empty_menu(gemini_client):
    tool_calls = [
        {"name": "get_menu", "args": {}, "result": {
            "status": "show_menu",
            "menu": "",
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "unavailable" in result.lower()


# ---------- view_order (Bug 3) ----------

def test_view_order_fallback_with_items(gemini_client):
    tool_calls = [
        {"name": "view_order", "args": {}, "result": {
            "status": "view_order",
            "order": {
                "items": [
                    {"name": "Classic Burger", "quantity": 1, "line_total": 10.50},
                    {"name": "French Fries", "quantity": 1, "line_total": 3.50},
                ],
                "total": 14.00,
            },
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "Classic Burger" in result
    assert "French Fries" in result
    assert "$10.50" in result
    assert "$3.50" in result
    assert "$14.00" in result


def test_view_order_fallback_empty_order(gemini_client):
    tool_calls = [
        {"name": "view_order", "args": {}, "result": {
            "status": "view_order",
            "order": {"items": [], "total": 0},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "empty" in result.lower()


# ---------- confirm_order ----------

def test_confirm_order_fallback_with_summary(gemini_client):
    summary = "  - Classic Burger x1 -- $10.50\n  Total: $10.50"
    tool_calls = [
        {"name": "confirm_order", "args": {}, "result": {
            "status": "confirm",
            "order": {},
            "summary_text": summary,
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert summary in result
    assert "Ready to submit?" in result


def test_confirm_order_fallback_no_summary(gemini_client):
    tool_calls = [
        {"name": "confirm_order", "args": {}, "result": {
            "status": "confirm",
            "order": {},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "Ready to submit?" in result


# ---------- cancel_order ----------

def test_cancel_order_fallback(gemini_client):
    tool_calls = [
        {"name": "cancel_order", "args": {}, "result": {
            "status": "cancelled",
            "order": {"items": [], "total": 0},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "cancelled" in result.lower()


# ---------- submit_order ----------

def test_submit_order_success_fallback(gemini_client):
    tool_calls = [
        {"name": "submit_order", "args": {}, "result": {
            "status": "ready_to_submit",
            "mcp_result": {"success": True, "order_id": "ORD-999"},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "ORD-999" in result


def test_submit_order_failure_fallback(gemini_client):
    tool_calls = [
        {"name": "submit_order", "args": {}, "result": {
            "status": "ready_to_submit",
            "mcp_result": {"success": False, "error": "Server unavailable"},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "Server unavailable" in result


# ---------- set_special_instructions (Bug 3) ----------

def test_set_special_instructions_fallback(gemini_client):
    tool_calls = [
        {"name": "set_special_instructions", "args": {}, "result": {
            "status": "special_instructions_set",
            "instructions": "No onions please",
            "order": {},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "No onions please" in result
    assert "Anything else?" in result


# ---------- unknown tool ----------

def test_unknown_tool_fallback(gemini_client):
    tool_calls = [
        {"name": "fly_to_moon", "args": {}, "result": {"error": "Unknown tool"}},
    ]
    result = gemini_client._fallback_response(tool_calls)
    assert "Done!" in result or "Anything else?" in result


# ---------- multiple tool calls — uses last ----------

def test_multiple_tools_uses_last_call(gemini_client):
    """When multiple tools are called, fallback should use the last one."""
    tool_calls = [
        {"name": "add_item", "args": {}, "result": {
            "status": "added",
            "item": {"name": "Classic Burger"},
            "order": {},
        }},
        {"name": "add_item", "args": {}, "result": {
            "status": "added",
            "item": {"name": "French Fries"},
            "order": {},
        }},
    ]
    result = gemini_client._fallback_response(tool_calls)
    # Should reference the last item added
    assert "French Fries" in result
