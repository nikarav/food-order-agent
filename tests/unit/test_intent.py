from orderbot.models.intent import IntentType, ParsedIntent


def test_intent_type_values():
    assert IntentType.ADD_ITEM == "add_item"
    assert IntentType.CONFIRM_ORDER == "confirm_order"
    assert IntentType.UNKNOWN == "unknown"


def test_parsed_intent_defaults():
    intent = ParsedIntent(intent=IntentType.ADD_ITEM)
    assert intent.item_id is None
    assert intent.quantity is None
    assert intent.ambiguous is False
    assert intent.extras_add is None


def test_parsed_intent_full():
    intent = ParsedIntent(
        intent=IntentType.ADD_ITEM,
        item_id="classic_burger",
        quantity=2,
        options={"size": "large", "patty": "beef"},
        extras_add=["cheese", "bacon"],
        ambiguous=False,
    )
    assert intent.item_id == "classic_burger"
    assert intent.quantity == 2
    assert intent.options == {"size": "large", "patty": "beef"}
    assert intent.extras_add == ["cheese", "bacon"]


def test_parsed_intent_ambiguous():
    intent = ParsedIntent(
        intent=IntentType.ADD_ITEM,
        ambiguous=True,
        candidates=["classic_burger", "spicy_burger"],
        clarification_needed="Which burger?",
    )
    assert intent.ambiguous is True
    assert len(intent.candidates) == 2


def test_parsed_intent_modify():
    intent = ParsedIntent(
        intent=IntentType.MODIFY_ITEM,
        item_id="classic_burger",
        options={"size": "large"},
        extras_remove=["bacon"],
        target_index=0,
    )
    assert intent.intent == IntentType.MODIFY_ITEM
    assert intent.target_index == 0
    assert intent.extras_remove == ["bacon"]


def test_intent_serialization():
    intent = ParsedIntent(intent=IntentType.GREETING)
    data = intent.model_dump()
    assert data["intent"] == "greeting"
