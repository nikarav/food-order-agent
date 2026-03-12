"""
FoodOrderAgent: public API.
Orchestrates tool-calling loop: LLM picks tools → ToolExecutor runs them deterministically.
"""

import asyncio
import logging

from orderbot.llm.gemini import GeminiClient
from orderbot.utils.logger import ConversationLogger

logger = logging.getLogger(__name__)
from orderbot.utils.trace import ConversationTrace
from orderbot.mcp.client import MCPClient
from orderbot.models.menu import Menu
from orderbot.order.manager import OrderManager
from orderbot.tools.executor import ToolExecutor
from orderbot.utils.config import load_configurations

MAX_HISTORY_TURNS = 20


class FoodOrderAgent:
    def __init__(self):
        config = load_configurations("config/agent.yaml")
        self.menu = Menu.from_yaml(config.menu.path)
        self.order_manager = OrderManager(self.menu)
        self.llm = GeminiClient(config)
        self.mcp = MCPClient(config)
        self.menu_text = self.menu.to_prompt_string()
        self.menu_display_text = self.menu.to_display_string()
        self.tool_executor = ToolExecutor(
            self.order_manager, self.menu_text, self.menu_display_text
        )
        self.logger = ConversationLogger(config.log_level)
        self.trace = ConversationTrace()

        # Tracks whether confirm_order was called — gates submit_order
        self._awaiting_confirmation = False
        # Conversation history as google.genai Content objects
        self._history: list = []

        # Persistent event loop for synchronous send()
        self._loop = asyncio.new_event_loop()

    def send(self, message: str) -> dict:
        """
        Process user message synchronously. Returns response dict.

        :param message: The user's message
        :return: A dictionary containing the final natural language response,
                 all tools executed this turn, and the content objects to append to history
        """
        return self._loop.run_until_complete(self._process(message))

    def __del__(self):
        try:
            self._loop.close()
        except Exception:
            pass

    # --- Async core ---

    async def _process(self, message: str) -> dict:
        snapshot = self.order_manager.get_snapshot()

        result = await self.llm.process_turn(
            user_message=message,
            order_snapshot=snapshot,
            menu_text=self.menu_text,
            history=self._history,
            tool_executor=self.tool_executor,
        )

        # Compress turn to [user_msg, final_model_text] — order_snapshot re-injected each turn
        self._history.extend(self._compress_turn(result["history_additions"]))
        self._trim_history()

        # Check if submit_order sentinel was returned — do the real MCP call
        mcp_tool_calls = None
        for tc in result["tool_calls_made"]:
            if tc["name"] == "submit_order":
                if tc["result"].get("status") == "ready_to_submit":
                    if not self._awaiting_confirmation:
                        # LLM called submit without confirm — override response
                        result["text"] = (
                            "Please review your order first before submitting. "
                            "Just say \"that's it\" when you're done adding items."
                        )
                        break
                    payload = self.order_manager.order.to_submit_payload()
                    logger.debug("submitting order payload: %s", payload)
                    mcp_result = await self.mcp.submit_order(payload)
                    logger.debug("MCP result: %s", mcp_result)
                    mcp_tool_calls = [
                        {
                            "name": "submit_order",
                            "arguments": payload,
                            "result": mcp_result,
                        }
                    ]
                    self._awaiting_confirmation = False

        # Track confirmation state
        for tc in result["tool_calls_made"]:
            if tc["name"] == "confirm_order" and "error" not in tc["result"]:
                self._awaiting_confirmation = True
            elif tc["name"] in ("add_item", "modify_item", "remove_item", "cancel_order"):
                self._awaiting_confirmation = False

        self._log(message, result["tool_calls_made"], result["text"], mcp_tool_calls)
        return self._build_response(result["text"], mcp_tool_calls, result["tool_calls_made"])

    # --- Helpers ---

    @staticmethod
    def _compress_turn(history_additions: list) -> list:
        """
        Compress a completed turn to [user_message, final_model_text].

        Tool FC batches and tool-result contents are dropped because:
        - order_snapshot is re-injected fresh every turn via the system prompt.
        - The model only needs conversational context (what was said / confirmed).
        - Dropping intermediate objects cuts per-turn history cost by ~75%.

        Layout: [user_content, model_FC, tool_results..., final_model_text]
        Keep first (user) and last (final model text) only.
        """
        if len(history_additions) <= 2:
            return history_additions
        return [history_additions[0], history_additions[-1]]

    def _trim_history(self) -> None:
        """
        Hard cap: keep at most MAX_HISTORY_TURNS turns.

        After compression each turn is exactly 2 Content objects, so
        simple len() / 2 arithmetic is reliable.
        """
        max_objects = MAX_HISTORY_TURNS * 2
        if len(self._history) > max_objects:
            self._history = self._history[-max_objects:]
            logger.debug("trimmed history to %d turns", MAX_HISTORY_TURNS)

    def _build_response(
        self,
        message: str,
        tool_calls: list | None = None,
        tool_calls_made: list | None = None,
    ) -> dict:
        result: dict = {"message": message}
        if tool_calls:
            result["tool_calls"] = tool_calls
        if tool_calls_made:
            result["_tool_calls_made"] = tool_calls_made  # debug; prefixed with _ (not in spec)
        return result

    def _log(
        self,
        message: str,
        tool_calls_made: list,
        response: str,
        mcp_tool_calls: list | None = None,
    ) -> None:
        snapshot = self.order_manager.get_snapshot()
        self.logger.log_turn(
            user_message=message,
            tool_calls_made=tool_calls_made,
            response=response,
            mcp_tool_calls=mcp_tool_calls,
            order_snapshot=snapshot,
        )
        self.trace.add_turn(
            user_message=message,
            tool_calls_made=tool_calls_made,
            response=response,
            mcp_tool_calls=mcp_tool_calls,
            order_snapshot=snapshot,
        )
