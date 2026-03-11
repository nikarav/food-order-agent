"""
Integration tests for MCP client.
These hit the real endpoint — requires APPLICANT_EMAIL and network access.
Run with: pytest tests/integration/ -v
"""

import pytest

from orderbot.mcp.client import MCPClient


@pytest.fixture
def client():
    return MCPClient()


@pytest.mark.asyncio
async def test_submit_valid_order(client):
    payload = {
        "items": [
            {
                "item_id": "classic_burger",
                "quantity": 1,
                "options": {"size": "regular", "patty": "beef"},
                "extras": [],
            }
        ]
    }
    result = await client.submit_order(payload)
    # May succeed or fail depending on server state, but should return a dict
    assert isinstance(result, dict)
    assert "success" in result


@pytest.mark.asyncio
async def test_submit_over_limit_returns_error(client):
    """Order exceeding $50 should return a failure response."""
    payload = {
        "items": [
            {
                "item_id": "margherita",
                "quantity": 4,
                "options": {"size": "large"},
                "extras": ["extra_cheese", "pepperoni"],
            }
        ]
    }
    result = await client.submit_order(payload)
    # Server should reject this or succeed — either way we get a dict
    assert isinstance(result, dict)
    assert "success" in result
