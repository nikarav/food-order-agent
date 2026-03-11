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

        :param user_message: The user's message
        :param order_snapshot: The current order snapshot
        :param menu_text: The menu text
        :param history: The conversation history
        :param tool_executor: The tool executor
        :return: A dictionary containing the final natural language response, all tools executed this turn, and the content objects to append to history
            {
                "text": str,               # Final natural language response
                "tool_calls_made": list,   # All tools executed this turn
                "history_additions": list, # Content objects to append to history
            }
        """
        ...
