import pytest

from src.models.menu import Menu


@pytest.fixture
def menu():
    return Menu.from_yaml("data/menu.yaml")


def test_menu_loads(menu):
    assert len(menu.items) == 7


def test_find_by_id(menu):
    burger = menu.find_by_id("classic_burger")
    assert burger is not None
    assert burger.name == "Classic Burger"
    assert burger.base_price == 8.50


def test_find_by_id_missing(menu):
    assert menu.find_by_id("nonexistent") is None


def test_find_by_name_fuzzy(menu):
    results = menu.find_by_name_fuzzy("burger")
    assert len(results) == 2
    ids = {r.id for r in results}
    assert "classic_burger" in ids
    assert "spicy_burger" in ids


def test_get_extras_list(menu):
    burger = menu.find_by_id("classic_burger")
    extras = burger.get_extras_list()
    assert len(extras) == 4
    assert any(e.id == "cheese" for e in extras)
    assert any(e.id == "bacon" for e in extras)


def test_get_extra_price(menu):
    burger = menu.find_by_id("classic_burger")
    assert burger.get_extra_price("cheese") == 1.00
    assert burger.get_extra_price("bacon") == 1.50
    assert burger.get_extra_price("avocado") == 2.00
    assert burger.get_extra_price("nonexistent") is None


def test_get_option_modifier(menu):
    burger = menu.find_by_id("classic_burger")
    assert burger.get_option_modifier("size", "regular") == 0.0
    assert burger.get_option_modifier("size", "large") == 2.00
    assert burger.get_option_modifier("patty", "beef") == 0.0  # no modifier


def test_validate_option(menu):
    burger = menu.find_by_id("classic_burger")
    assert burger.validate_option("size", "large") is True
    assert burger.validate_option("size", "xl") is False
    assert burger.validate_option("nonexistent", "large") is False


def test_validate_extra(menu):
    burger = menu.find_by_id("classic_burger")
    assert burger.validate_extra("cheese") is True
    assert burger.validate_extra("mushrooms") is False


def test_menu_item_no_extras(menu):
    soda = menu.find_by_id("soda")
    assert soda.get_extras_list() == []


def test_to_prompt_string(menu):
    text = menu.to_prompt_string()
    assert "classic_burger" in text
    assert "Classic Burger" in text
    assert "cheese" in text
    assert "large" in text


def test_margherita_price_modifier(menu):
    pizza = menu.find_by_id("margherita")
    assert pizza.get_option_modifier("size", "small") == -2.00
    assert pizza.get_option_modifier("size", "large") == 4.00
