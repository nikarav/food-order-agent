from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class OrderItem(BaseModel):
    """A single item in the order with resolved options and extras."""

    uid: str = Field(default_factory=lambda: uuid4().hex[:8])
    item_id: str
    name: str
    quantity: int = 1
    options: dict[str, str] = {}
    extras: list[str] = []
    unit_price: float = 0.0
    special_instructions: Optional[str] = None


class Order(BaseModel):
    """Full order state."""

    items: list[OrderItem] = []
    special_instructions: Optional[str] = None

    @property
    def total(self) -> float:
        return round(sum(item.unit_price * item.quantity for item in self.items), 2)

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def to_submit_payload(self) -> dict:
        """Convert to the exact schema expected by submit_order MCP tool."""
        payload = {
            "items": [
                {
                    "item_id": item.item_id,
                    "quantity": item.quantity,
                    "options": item.options,
                    "extras": item.extras,
                    **({"special_instructions": item.special_instructions}
                       if item.special_instructions else {}),
                }
                for item in self.items
            ],
            "special_instructions": self.special_instructions or "",
        }
        return payload
