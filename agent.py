"""
FoodOrderAgent: public API.
Orchestrates tool-calling loop: LLM picks tools → ToolExecutor runs them deterministically.
"""

import asyncio
import logging
from uuid import uuid4

from orderbot.llm.gemini import GeminiClient
from orderbot.mcp.client import MCPClient
from orderbot.models.menu import Menu
from orderbot.order.manager import OrderManager
from orderbot.tools.executor import ToolExecutor
from orderbot.utils.config import load_configurations
from orderbot.utils.logger import ConversationLogger
from orderbot.utils.observability import get_langfuse_client, propagate_attributes, shutdown_langfuse

logger = logging.getLogger(__name__)

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
        self._session_id = uuid4().hex

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

    def shutdown(self) -> None:
        """Flush Langfuse data and close the event loop."""
        shutdown_langfuse()
        try:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self._loop.close()
        except Exception:
            pass

    # --- Async core ---

    async def _process(self, message: str) -> dict:
        lf = get_langfuse_client()
        with propagate_attributes(session_id=self._session_id, trace_name="turn"):
            with lf.start_as_current_observation(
                name="agent", as_type="span", input=message,
            ) as agent_span:
                response = await self._run_turn(message, lf)
                agent_span.update(output=response["message"])
                return response

    async def _run_turn(self, message: str, lf) -> dict:
        snapshot = self.order_manager.get_snapshot()

        result = await self.llm.process_turn(
            user_message=message,
            order_snapshot=snapshot,
            menu_text=self.menu_text,
            history=self._history,
            tool_executor=self.tool_executor,
        )

        # Check if submit_order sentinel was returned — do the real MCP call.
        # History compression happens AFTER this block so generate_mcp_error_response
        # receives the clean pre-turn history alongside the uncompressed turn_additions.
        mcp_tool_calls = None
        for tc in result["tool_calls_made"]:
            if tc["name"] == "submit_order":
                if tc["result"].get("status") == "ready_to_submit":
                    if not self._awaiting_confirmation:
                        result["text"] = (
                            "Please review your order first before submitting. "
                            "Just say \"that's it\" when you're done adding items."
                        )
                        break
                    payload = self.order_manager.order.to_submit_payload()
                    logger.debug("submitting order payload: %s", payload)
                    with lf.start_as_current_observation(
                        name="mcp_submit_order", as_type="span", input=payload
                    ) as span:
                        mcp_result = await self.mcp.submit_order(payload)
                        span.update(output=mcp_result)
                    logger.debug("MCP result: %s", mcp_result)
                    mcp_tool_calls = [
                        {
                            "name": "submit_order",
                            "arguments": payload,
                            "result": mcp_result,
                        }
                    ]
                    if not mcp_result.get("success"):
                        result["text"] = await self.llm.generate_mcp_error_response(
                            history=self._history,
                            turn_additions=result["history_additions"],
                            mcp_result=mcp_result,
                            order_snapshot=snapshot,
                            menu_text=self.menu_text,
                            lf=lf,
                        )
                        self._awaiting_confirmation = True
                    else:
                        self._awaiting_confirmation = False
                        order_id = mcp_result.get("order_id", "N/A")
                        total = mcp_result.get("total")
                        estimated_time = mcp_result.get("estimated_time")
                        total_str = f"${total:.2f}" if isinstance(total, (int, float)) else str(total)
                        result["text"] = (
                            f"Your order has been placed! 🎉\n"
                            f"Order ID: {order_id}\n"
                            f"Total: {total_str}\n"
                            f"Estimated time: {estimated_time}"
                        )

        # Safeguard: if the model claims submission without a real MCP call, block it.
        # This catches hallucinated success messages (e.g. after a failed retry).
        submit_tool_called = any(tc["name"] == "submit_order" for tc in result["tool_calls_made"])
        if not submit_tool_called and mcp_tool_calls is None:
            submission_phrases = ("submitted", "placed", "on its way", "order is in")
            if any(p in result["text"].lower() for p in submission_phrases):
                logger.warning("model claimed submission without calling submit_order — blocking")
                result["text"] = (
                    "I wasn't able to submit your order. Please say \"yes\" to try again."
                )

        # Sync history with the actual response shown to the user.
        # Several code paths above override result["text"] after the LLM produced its
        # response (MCP success/failure, confirmation gate, hallucination safeguard).
        # Without this sync the compressed history would contain the model's original
        # (stale/hallucinated) text, causing it to repeat the same pattern on future turns.
        if result["history_additions"]:
            from google.genai import types
            last = result["history_additions"][-1]
            if last.role == "model":
                result["history_additions"][-1] = types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=result["text"])],
                )

        self._history.extend(self._compress_turn(result["history_additions"]))
        self._trim_history()

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
        """
        Build the response dictionary.

        :param message: The message
        :param tool_calls: The tool calls
        :param tool_calls_made: The tool calls made
        :return: The response dictionary
        """
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
