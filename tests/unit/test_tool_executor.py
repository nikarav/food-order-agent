import pytest

from src.models.menu import Menu
from src.order.manager import OrderManager
from src.tools.executor import ToolExecutor


@pytest.fixture
def menu():
    return Menu.from_yaml("data/menu.yaml")


@pytest.fixture
def executor(menu):
    om = OrderManager(menu)
    return ToolExecutor(om, menu.to_prompt_string())


@pytest.fixture
def executor_with_burger(menu):
    om = OrderManager(menu)
    ex = ToolExecutor(om, menu.to_prompt_string())
    ex.execute("add_item", {"item_id": "classic_burger"})
    return ex, om


# --- add_item ---

def test_add_item_basic(executor):
    result = executor.execute("add_item", {"item_id": "classic_burger"})
    assert result["status"] == "added"
    assert result["item"]["item_id"] == "classic_burger"
    assert result["order"]["item_count"] == 1


def test_add_item_with_options_and_extras(executor):
    result = executor.execute(
        "add_item",
        {"item_id": "classic_burger", "options": {"size": "large"}, "extras": ["cheese", "bacon"]},
    )
    assert result["status"] == "added"
    assert result["item"]["options"]["size"] == "large"
    assert "cheese" in result["item"]["extras"]
    # 8.50 + 2.00 (large) + 1.00 (cheese) + 1.50 (bacon) = 13.00
    assert result["item"]["unit_price"] == 13.00


def test_add_item_unknown_id_returns_error(executor):
    result = executor.execute("add_item", {"item_id": "unicorn_burger"})
    assert "error" in result


def test_add_multiple_items(executor):
    executor.execute("add_item", {"item_id": "classic_burger"})
    executor.execute("add_item", {"item_id": "fries"})
    result = executor.execute("add_item", {"item_id": "soda"})
    assert result["order"]["item_count"] == 3


# --- modify_item ---

def test_modify_item_change_size(executor_with_burger):
    ex, om = executor_with_burger
    result = ex.execute("modify_item", {"item_id": "classic_burger", "options": {"size": "large"}})
    assert result["status"] == "modified"
    assert result["item"]["options"]["size"] == "large"
    assert result["item"]["unit_price"] == 10.50


def test_modify_item_add_extra(executor_with_burger):
    ex, om = executor_with_burger
    result = ex.execute("modify_item", {"item_id": "classic_burger", "extras_add": ["cheese"]})
    assert "cheese" in result["item"]["extras"]


def test_modify_item_remove_extra(executor_with_burger):
    ex, om = executor_with_burger
    ex.execute("modify_item", {"item_id": "classic_burger", "extras_add": ["cheese", "bacon"]})
    result = ex.execute("modify_item", {"item_id": "classic_burger", "extras_remove": ["bacon"]})
    assert "bacon" not in result["item"]["extras"]
    assert "cheese" in result["item"]["extras"]


def test_modify_item_by_index(executor_with_burger):
    ex, om = executor_with_burger
    result = ex.execute("modify_item", {"target_index": 0, "options": {"size": "large"}})
    assert result["status"] == "modified"


# --- remove_item ---

def test_remove_item(executor_with_burger):
    ex, om = executor_with_burger
    result = ex.execute("remove_item", {"item_id": "classic_burger"})
    assert result["status"] == "removed"
    assert result["order"]["item_count"] == 0


def test_remove_item_from_empty_returns_error(executor):
    result = executor.execute("remove_item", {"item_id": "fries"})
    assert "error" in result


# --- view_order ---

def test_view_order_empty(executor):
    result = executor.execute("view_order", {})
    assert result["status"] == "view_order"
    assert result["order"]["item_count"] == 0


def test_view_order_with_items(executor_with_burger):
    ex, om = executor_with_burger
    result = ex.execute("view_order", {})
    assert result["order"]["item_count"] == 1


# --- confirm_order ---

def test_confirm_order_empty_returns_error(executor):
    result = executor.execute("confirm_order", {})
    assert "error" in result
    assert "empty" in result["error"].lower()


def test_confirm_order_returns_summary_text(executor_with_burger):
    ex, om = executor_with_burger
    result = ex.execute("confirm_order", {})
    assert result["status"] == "confirm"
    assert "summary_text" in result
    assert "$" in result["summary_text"]
    assert "Total" in result["summary_text"]


def test_confirm_order_summary_has_correct_total(executor):
    executor.execute("add_item", {"item_id": "fries", "options": {"size": "large"}})
    result = executor.execute("confirm_order", {})
    # fries large = 3.50 + 1.50 = 5.00
    assert "$5.00" in result["summary_text"]
    assert "Total: $5.00" in result["summary_text"]


# --- submit_order ---

def test_submit_order_empty_returns_error(executor):
    result = executor.execute("submit_order", {})
    assert "error" in result


def test_submit_order_with_items_returns_sentinel(executor_with_burger):
    ex, om = executor_with_burger
    result = ex.execute("submit_order", {})
    assert result["status"] == "ready_to_submit"


# --- cancel_order ---

def test_cancel_order_clears_items(executor_with_burger):
    ex, om = executor_with_burger
    result = ex.execute("cancel_order", {})
    assert result["status"] == "cancelled"
    assert result["order"]["item_count"] == 0


# --- set_special_instructions ---

def test_set_special_instructions(executor):
    result = executor.execute("set_special_instructions", {"instructions": "No onions please"})
    assert result["status"] == "special_instructions_set"
    assert result["order"]["special_instructions"] == "No onions please"


# --- unknown tool ---

def test_unknown_tool_returns_error(executor):
    result = executor.execute("fly_to_moon", {})
    assert "error" in result
    assert "Unknown tool" in result["error"]
