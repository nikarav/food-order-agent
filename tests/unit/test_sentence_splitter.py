"""Unit tests for split_sentences() — no API keys or audio hardware needed."""

from orderbot.voice.tts import split_sentences


# ── Basic splitting ──────────────────────────────────────────────────────────

def test_single_sentence_unchanged():
    assert split_sentences("Hello there.") == ["Hello there."]


def test_two_sentences_split():
    parts = split_sentences("Added Classic Burger. Anything else?")
    assert parts == ["Added Classic Burger.", "Anything else?"]


def test_three_sentences():
    text = "Item added. Your total is $12.50. Ready to submit?"
    parts = split_sentences(text)
    assert len(parts) == 3
    assert parts[0] == "Item added."
    assert parts[2] == "Ready to submit?"


def test_exclamation_split():
    parts = split_sentences("Great choice! Would you like fries?")
    assert len(parts) == 2


def test_question_split():
    parts = split_sentences("What size? Large or medium?")
    assert len(parts) == 2


# ── Abbreviation protection ───────────────────────────────────────────────────

def test_mr_not_split():
    parts = split_sentences("Mr. Smith ordered a burger.")
    assert len(parts) == 1


def test_dr_not_split():
    parts = split_sentences("Dr. Jones confirmed the order.")
    assert len(parts) == 1


def test_etc_not_split():
    parts = split_sentences("We have burgers, pizzas, etc. Would you like one?")
    # "etc." should not trigger a split
    assert len(parts) == 2
    assert parts[0].startswith("We have")


def test_eg_not_split():
    parts = split_sentences("Add extras, e.g. cheese or bacon. Anything else?")
    assert len(parts) == 2


# ── Price protection ──────────────────────────────────────────────────────────

def test_price_decimal_not_split():
    parts = split_sentences("Your total is $12.50. Confirm?")
    assert len(parts) == 2
    assert "$12.50" in parts[0]


def test_multiple_prices():
    parts = split_sentences("Burger is $8.99 and fries are $3.50. Total is $12.49.")
    # Prices should not create spurious splits
    for part in parts:
        assert part.strip()


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_string_returns_empty():
    assert split_sentences("") == []


def test_whitespace_only_returns_empty():
    assert split_sentences("   ") == []


def test_no_punctuation_single_item():
    result = split_sentences("hello world")
    assert result == ["hello world"]


def test_trailing_whitespace_stripped():
    parts = split_sentences("Done.  Ready?  ")
    assert all(p == p.strip() for p in parts)
    assert len(parts) == 2


def test_long_sentence_split_at_clause():
    # Sentences > 120 chars should be split at comma boundaries
    long = (
        "We have added a Classic Burger with large size and extra cheese to your order, "
        "and we also noted your special instructions for no pickles, and the total is now $15.50."
    )
    parts = split_sentences(long)
    assert len(parts) >= 2
    assert all(p.strip() for p in parts)


def test_already_short_sentence_not_split_at_comma():
    # Short sentences should not be split at commas
    text = "Burger, fries, and soda added."
    parts = split_sentences(text)
    assert len(parts) == 1


def test_result_contains_no_empty_strings():
    text = "Order confirmed. Please wait. Thank you!"
    parts = split_sentences(text)
    assert all(p.strip() for p in parts)
