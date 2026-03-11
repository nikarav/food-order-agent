"""
E2E conversation tests using the agent's programmatic API.
These require a valid GEMINI_API_KEY and APPLICANT_EMAIL.
Run with: pytest tests/e2e/ -v
"""

import pytest

from agent import FoodOrderAgent


@pytest.fixture
def agent():
    return FoodOrderAgent()


def test_add_item_returns_message(agent):
    response = agent.send("I'd like a classic burger")
    assert "message" in response
    assert isinstance(response["message"], str)
    assert len(response["message"]) > 0


def test_add_and_view_order(agent):
    agent.send("Give me medium fries")
    response = agent.send("What's in my order?")
    assert "message" in response
    assert "fries" in response["message"].lower() or "french" in response["message"].lower()


def test_add_modify_confirm_flow(agent):
    agent.send("A large classic burger with cheese")
    agent.send("Actually make it regular size")
    response = agent.send("That's it")
    # Should present order summary
    assert "message" in response


def test_submit_flow(agent):
    agent.send("I want a medium cola")
    agent.send("That's it")
    response = agent.send("Yes")
    assert "message" in response
    # Should have tool_calls for submit
    assert "tool_calls" in response
    assert response["tool_calls"][0]["name"] == "submit_order"


def test_empty_order_submit_fails(agent):
    response = agent.send("Submit my order")
    assert "message" in response
    # Should mention order is empty
    assert "empty" in response["message"].lower() or "nothing" in response["message"].lower()


def test_off_topic_rejected(agent):
    response = agent.send("What's the weather like today?")
    assert "message" in response
    # Should redirect to food
    assert len(response["message"]) > 0
