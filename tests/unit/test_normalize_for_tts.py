"""Unit tests for normalize_for_tts() — no API keys or audio hardware needed."""

from orderbot.voice.tts import normalize_for_tts


# ── Price formatting ─────────────────────────────────────────────────────────

def test_price_with_cents():
    assert normalize_for_tts("Total: $12.50") == "Total: 12 dollars and 50 cents"


def test_price_whole_dollars():
    assert normalize_for_tts("Burger costs $8") == "Burger costs 8 dollars"


def test_price_whole_dollars_not_double_converted():
    # $8.99 → "8 dollars and 99 cents", not "$8 dollars.99"
    result = normalize_for_tts("$8.99")
    assert "dollars" in result
    assert "$" not in result


def test_multiple_prices():
    result = normalize_for_tts("Burger $8.99, fries $3.50, total $12.49")
    assert "$" not in result
    assert "dollars" in result


def test_no_price_unchanged():
    assert normalize_for_tts("Add a burger please") == "Add a burger please"


# ── Quantity markers ─────────────────────────────────────────────────────────

def test_quantity_x1_removed():
    # "x1" is implied, should be stripped
    result = normalize_for_tts("Classic Burger x1")
    assert "x1" not in result
    assert "Classic Burger" in result


def test_quantity_x2_spoken():
    result = normalize_for_tts("Classic Burger x2")
    assert "times 2" in result
    assert "x2" not in result


def test_quantity_unicode_times():
    result = normalize_for_tts("Fries ×3")
    assert "times 3" in result


def test_quantity_x1_not_greedy():
    # "x12" should NOT be matched by x1 rule (negative lookahead)
    result = normalize_for_tts("Item x12")
    assert "times 12" in result


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_string():
    assert normalize_for_tts("") == ""


def test_no_symbols_unchanged():
    text = "Added Classic Burger to your order."
    assert normalize_for_tts(text) == text
