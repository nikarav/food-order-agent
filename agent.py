"""
FoodOrderAgent: public API.
Orchestrates tool-calling loop: LLM picks tools → ToolExecutor runs them deterministically.
"""

import asyncio

from config.configs import load_configurations
from src.log.logger import ConversationLogger
from src.log.trace import ConversationTrace
from src.llm.gemini import GeminiClient
from src.mcp.client import MCPClient
from src.models.menu import Menu
from src.order.manager import OrderManager
from src.tools.executor import ToolExecutor

MAX_HISTORY_TURNS = 20


class FoodOrderAgent:
    def __init__(self):
        config = load_configurations("config/agent.yaml")
        self.menu = Menu.from_yaml(config.menu.path)
        self.order_manager = OrderManager(self.menu)
        self.llm = GeminiClient(config)
        self.mcp = MCPClient(config)
        self.menu_text = self.menu.to_prompt_string()
        self.tool_executor = ToolExecutor(self.order_manager, self.menu_text)
        self.logger = ConversationLogger(config.log_level)
        self.trace = ConversationTrace()

        # Tracks whether confirm_order was called — gates submit_order
        self._awaiting_confirmation = False
        # Conversation history as google.genai Content objects
        self._history: list = []

        # Persistent event loop for synchronous send()
        self._loop = asyncio.new_event_loop()

    def send(self, message: str) -> dict:
        """Process user message synchronously. Returns response dict."""
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

        # Extend conversation history
        self._history.extend(result["history_additions"])
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
                    mcp_result = await self.mcp.submit_order(payload)
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
        return self._build_response(result["text"], mcp_tool_calls)

    # --- Helpers ---

    def _trim_history(self) -> None:
        """Keep last MAX_HISTORY_TURNS user+model pairs to cap context size."""
        if len(self._history) > MAX_HISTORY_TURNS * 2:
            self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

    def _build_response(self, message: str, tool_calls: list | None = None) -> dict:
        result: dict = {"message": message}
        if tool_calls:
            result["tool_calls"] = tool_calls
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
