from enum import Enum
from typing import Optional

from pydantic import BaseModel


class IntentType(str, Enum):
    ADD_ITEM = "add_item"
    MODIFY_ITEM = "modify_item"
    REMOVE_ITEM = "remove_item"
    VIEW_ORDER = "view_order"
    CONFIRM_ORDER = "confirm_order"
    SUBMIT_ORDER = "submit_order"
    CANCEL_ORDER = "cancel_order"
    GREETING = "greeting"
    ASK_MENU = "ask_menu"
    SPECIAL_INSTRUCTIONS = "special_instructions"
    UNKNOWN = "unknown"


class ParsedIntent(BaseModel):
    """Structured output from the LLM classification step."""

    intent: IntentType
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    quantity: Optional[int] = None
    options: Optional[dict] = None
    extras_add: Optional[list[str]] = None
    extras_remove: Optional[list[str]] = None
    target_uid: Optional[str] = None
    target_index: Optional[int] = None
    special_instructions: Optional[str] = None
    raw_text: Optional[str] = None
    ambiguous: bool = False
    candidates: Optional[list[str]] = None
    clarification_needed: Optional[str] = None
