from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    async def process_turn(
        self,
        user_message: str,
        order_snapshot: dict,
        menu_text: str,
        history: list,
        tool_executor: Any,
    ) -> dict:
        """
        Process a single conversation turn with tool calling.

        Returns:
            {
                "text": str,               # Final natural language response
                "tool_calls_made": list,   # All tools executed this turn
                "history_additions": list, # Content objects to append to history
            }
        """
        ...
