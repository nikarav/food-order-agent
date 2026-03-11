import pytest

from orderbot.models.menu import Menu
from orderbot.order.manager import OrderError, OrderManager


@pytest.fixture
def menu():
    return Menu.from_yaml("data/menu.yaml")


@pytest.fixture
def mgr(menu):
    return OrderManager(menu)


# --- add_item ---

def test_add_item_defaults(mgr):
    item = mgr.add_item(item_id="classic_burger")
    assert item.item_id == "classic_burger"
    assert item.options["size"] == "regular"
    assert item.options["patty"] == "beef"
    assert item.unit_price == 8.50


def test_add_item_large_with_extras(mgr):
    item = mgr.add_item(
        item_id="classic_burger",
        options={"size": "large"},
        extras=["cheese", "bacon"],
    )
    # 8.50 + 2.00 (large) + 1.00 (cheese) + 1.50 (bacon) = 13.00
    assert item.unit_price == 13.00
    assert "cheese" in item.extras
    assert "bacon" in item.extras


def test_add_item_quantity(mgr):
    item = mgr.add_item(item_id="fries", quantity=2)
    assert item.quantity == 2
    assert mgr.order.total == round(3.50 * 2, 2)


def test_add_item_invalid_id(mgr):
    with pytest.raises(OrderError, match="Unknown menu item"):
        mgr.add_item(item_id="unknown_item")


def test_add_item_invalid_extra(mgr):
    with pytest.raises(OrderError, match="Invalid extra"):
        mgr.add_item(item_id="classic_burger", extras=["mushrooms"])


def test_add_item_invalid_option(mgr):
    with pytest.raises(OrderError):
        mgr.add_item(item_id="classic_burger", options={"size": "xl"})


def test_add_item_sets_defaults_for_required(mgr):
    item = mgr.add_item(item_id="soda")
    assert item.options["size"] == "medium"
    assert item.options["flavor"] == "cola"
    assert item.unit_price == 2.00


# --- modify_item ---

def test_modify_item_change_size(mgr):
    mgr.add_item(item_id="classic_burger")
    item = mgr.modify_item(item_id="classic_burger", options={"size": "large"})
    assert item.options["size"] == "large"
    assert item.unit_price == 10.50  # 8.50 + 2.00


def test_modify_item_add_extra(mgr):
    mgr.add_item(item_id="classic_burger")
    item = mgr.modify_item(item_id="classic_burger", extras_add=["cheese"])
    assert "cheese" in item.extras
    assert item.unit_price == 9.50  # 8.50 + 1.00


def test_modify_item_remove_extra(mgr):
    mgr.add_item(item_id="classic_burger", extras=["cheese", "bacon"])
    item = mgr.modify_item(item_id="classic_burger", extras_remove=["bacon"])
    assert "bacon" not in item.extras
    assert "cheese" in item.extras
    assert item.unit_price == 9.50  # 8.50 + 1.00


def test_modify_item_change_quantity(mgr):
    mgr.add_item(item_id="fries")
    item = mgr.modify_item(item_id="fries", quantity=3)
    assert item.quantity == 3


def test_modify_item_resolves_last_item(mgr):
    mgr.add_item(item_id="fries")
    item = mgr.modify_item(options={"size": "large"})
    assert item.options["size"] == "large"


def test_modify_item_ambiguous_raises(mgr):
    mgr.add_item(item_id="classic_burger")
    mgr.add_item(item_id="classic_burger")
    with pytest.raises(OrderError, match="2 Classic Burgers"):
        mgr.modify_item(item_id="classic_burger", options={"size": "large"})


def test_modify_empty_order_raises(mgr):
    with pytest.raises(OrderError, match="No items"):
        mgr.modify_item(item_id="fries")


# --- remove_item ---

def test_remove_item(mgr):
    mgr.add_item(item_id="classic_burger")
    mgr.add_item(item_id="fries")
    removed = mgr.remove_item(item_id="fries")
    assert removed.item_id == "fries"
    assert len(mgr.order.items) == 1


def test_remove_only_item(mgr):
    mgr.add_item(item_id="classic_burger")
    mgr.remove_item(item_id="classic_burger")
    assert mgr.order.is_empty


def test_remove_from_empty_raises(mgr):
    with pytest.raises(OrderError):
        mgr.remove_item(item_id="fries")


def test_remove_by_index(mgr):
    mgr.add_item(item_id="classic_burger")
    mgr.add_item(item_id="fries")
    removed = mgr.remove_item(target_index=0)
    assert removed.item_id == "classic_burger"


# --- pre_submit_check ---

def test_pre_submit_empty(mgr):
    assert mgr.pre_submit_check() == "Your order is empty."


def test_pre_submit_valid(mgr):
    mgr.add_item(item_id="classic_burger")
    assert mgr.pre_submit_check() is None


# --- pricing ---

def test_calculate_price_pizza_small(mgr):
    item = mgr.add_item(item_id="margherita", options={"size": "small"})
    assert item.unit_price == 10.00  # 12.00 - 2.00


def test_calculate_price_milkshake_large_with_extras(mgr):
    item = mgr.add_item(
        item_id="milkshake",
        options={"size": "large", "flavor": "chocolate"},
        extras=["whipped_cream", "cherry_on_top"],
    )
    # 5.50 + 2.00 + 0.50 + 0.25 = 8.25
    assert item.unit_price == 8.25


# --- get_snapshot ---

def test_get_snapshot_structure(mgr):
    mgr.add_item(item_id="classic_burger")
    snap = mgr.get_snapshot()
    assert "items" in snap
    assert "total" in snap
    assert "item_count" in snap
    assert snap["item_count"] == 1
    assert snap["items"][0]["name"] == "Classic Burger"


# --- to_submit_payload ---

def test_to_submit_payload(mgr):
    mgr.add_item(item_id="classic_burger", options={"size": "large"}, extras=["cheese"])
    payload = mgr.order.to_submit_payload()
    assert "items" in payload
    assert payload["items"][0]["item_id"] == "classic_burger"
    assert payload["items"][0]["options"]["size"] == "large"
    assert "cheese" in payload["items"][0]["extras"]
