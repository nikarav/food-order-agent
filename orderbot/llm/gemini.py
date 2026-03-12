import asyncio
import json
import logging

from google import genai
from google.genai import types

from orderbot.llm.base import LLMClient
from orderbot.tools.declarations import ORDER_TOOLS
from orderbot.utils.observability import get_langfuse_client

logger = logging.getLogger(__name__)

_AUTO_FC_OFF = types.AutomaticFunctionCallingConfig(disable=True)


class GeminiClient(LLMClient):
    def __init__(self, config):
        """
        Initialize the GeminiClient.

        :param config: The configuration
        """
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.model_name
        self.temperature = config.temperature
        self._max_retries = int(getattr(config, "max_retries", 2))
        with open(config.prompts.system) as f:
            self._system_template = f.read()

    async def process_turn(
        self,
        user_message: str,
        order_snapshot: dict,
        menu_text: str,
        history: list,
        tool_executor,
    ) -> dict:
        """
        Run one conversation turn:
          1. Send user message + history + tools to Gemini
          2. Execute any function calls in parallel via tool_executor
          3. Feed results back until Gemini returns a text response

        :param user_message: The user's message
        :param order_snapshot: The current order snapshot
        :param menu_text: The menu text
        :param history: The conversation history
        :param tool_executor: The tool executor
        :return: A dictionary containing the final natural language response,
                all tools executed this turn, and the content objects to append to history
        """
        lf = get_langfuse_client()
        with lf.start_as_current_observation(
            name="process_turn", as_type="span", input=user_message,
        ) as span:
            result = await self._process_turn_inner(
                user_message, order_snapshot, menu_text, history, tool_executor, lf,
            )
            span.update(output=result["text"])
            return result

    async def _process_turn_inner(
        self, user_message, order_snapshot, menu_text, history, tool_executor, lf,
    ) -> dict:
        system = self._build_system_prompt(menu_text, order_snapshot)
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        contents = list(history) + [user_content]

        history_additions = [user_content]
        tool_calls_made = []

        try:
            response = await self._generate(contents, system, lf)

            # Retry on empty response (model returned STOP but no content)
            for _retry in range(self._max_retries):
                if response.function_calls or self._has_text(response):
                    break
                logger.debug("empty response from model, retry %d/%d", _retry + 1, self._max_retries)
                response = await self._generate(contents, system, lf)

            # Tool-calling loop — each iteration may produce parallel calls
            while response.function_calls:
                model_content = response.candidates[0].content
                history_additions.append(model_content)

                tasks = [
                    self._run_tool(lf, tool_executor, fc.name, dict(fc.args) if fc.args else {})
                    for fc in response.function_calls
                ]
                results = await asyncio.gather(*tasks)

                function_response_parts = []
                for fc, result in zip(response.function_calls, results):
                    args = dict(fc.args) if fc.args else {}
                    tool_calls_made.append({"name": fc.name, "args": args, "result": result})
                    function_response_parts.append(
                        types.Part.from_function_response(name=fc.name, response=result)
                    )

                tool_response_content = types.Content(role="user", parts=function_response_parts)
                history_additions.append(tool_response_content)

                response = await self._generate(list(history) + history_additions, system, lf)

            # Final text response
            final_content = response.candidates[0].content
            if final_content and getattr(final_content, "parts", None) is not None:
                history_additions.append(final_content)
            text_raw = ""
            try:
                text_raw = (response.text or "").strip()
            except Exception:
                pass
            text = text_raw or self._fallback_response(tool_calls_made)

        except Exception as e:
            logger.warning(f"process_turn failed ({e}), returning fallback")
            text = self._fallback_response(tool_calls_made)
            fallback_model = types.Content(
                role="model", parts=[types.Part.from_text(text=text)],
            )
            history_additions = [user_content, fallback_model]

        return {
            "text": text,
            "tool_calls_made": tool_calls_made,
            "history_additions": history_additions,
        }

    # --- Private helpers ---

    async def _generate(self, contents: list, system: str, lf=None):
        """
        Single Gemini API call with shared model config.

        :param contents: The contents to generate
        :param system: The system prompt
        :param lf: Langfuse client (passed from parent to stay in trace context)
        :return: The generated content
        """
        lf = lf or get_langfuse_client()
        lf_input = self._last_text(contents)
        with lf.start_as_current_observation(
            name="gemini_generate",
            as_type="generation",
            model=self.model,
            model_parameters={"temperature": self.temperature},
            input=lf_input,
        ) as gen:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    tools=[ORDER_TOOLS],
                    automatic_function_calling=_AUTO_FC_OFF,
                    temperature=self.temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            # Output: text response, or list of function calls if this is a tool-calling turn
            if response.function_calls:
                lf_output = [
                    {"name": fc.name, "args": dict(fc.args) if fc.args else {}}
                    for fc in response.function_calls
                ]
            else:
                lf_output = response.text or ""
            usage = getattr(response, "usage_metadata", None)
            gen.update(
                output=lf_output,
                usage_details={
                    "input": getattr(usage, "prompt_token_count", 0) or 0,
                    "output": getattr(usage, "candidates_token_count", 0) or 0,
                    "total": getattr(usage, "total_token_count", 0) or 0,
                } if usage else {},
            )
            return response

    @staticmethod
    def _has_text(response) -> bool:
        """Check if the response contains any non-empty text."""
        try:
            return bool((response.text or "").strip())
        except Exception:
            return False

    @staticmethod
    def _last_text(contents: list) -> str:
        """Extract the text of the last content part (used as Langfuse generation input)."""
        if not contents:
            return ""
        for part in getattr(contents[-1], "parts", []):
            text = getattr(part, "text", None)
            if text:
                return text
        return ""

    @staticmethod
    async def _run_tool(lf, tool_executor, name: str, args: dict) -> dict:
        """
        Run one tool call in a thread pool so parallel calls don't block each other.

        :param lf: Langfuse client (passed from parent to stay in trace context)
        :param tool_executor: The tool executor
        :param name: The name of the tool to execute
        :param args: The arguments to the tool
        :return: The result of the tool execution
        """
        with lf.start_as_current_observation(name=f"tool:{name}", as_type="tool", input=args) as span:
            result = await asyncio.to_thread(tool_executor.execute, name, args)
            span.update(output=result)
            return result

    async def generate_mcp_error_response(
        self,
        history: list,
        turn_additions: list,
        mcp_result: dict,
        order_snapshot: dict,
        menu_text: str,
        lf=None,
    ) -> str:
        """
        Generate a helpful response after MCP submission failure.

        Replaces the sentinel tool result with the real MCP failure dict and does one
        more generate call so the model can advise the user (e.g. suggest removing items
        to get under a price limit) rather than returning a hardcoded string.

        turn_additions layout: [user_content, model_FC, tool_results, final_model_text]
        We keep only [user_content, model_FC] and inject the MCP failure as the real
        function response, then generate once for the text reply.

        :param history: Compressed conversation history before this turn
        :param turn_additions: history_additions returned by process_turn (uncompressed)
        :param mcp_result: The failure dict from MCPClient.submit_order
        :param order_snapshot: Current order snapshot
        :param menu_text: Menu text
        :param lf: Langfuse client
        :return: A user-facing error message string
        """
        lf = lf or get_langfuse_client()
        system = self._build_system_prompt(menu_text, order_snapshot)
        fallback = (
            f"Sorry, your order couldn't be submitted: "
            f"{mcp_result.get('error', 'Unknown error')}. Please try again."
        )

        # turn_additions[:2] = [user_content, model_FC_with_submit_order_call]
        if len(turn_additions) < 2:
            return fallback

        mcp_response_content = types.Content(
            role="user",
            parts=[types.Part.from_function_response(name="submit_order", response=mcp_result)],
        )
        contents = list(history) + list(turn_additions[:2]) + [mcp_response_content]

        try:
            response = await self._generate(contents, system, lf)
            return (response.text or "").strip() or fallback
        except Exception as e:
            logger.warning("MCP error response generation failed: %s", e)
            return fallback

    def _build_system_prompt(self, menu_text: str, order_snapshot: dict) -> str:
        """
        Build the system prompt.

        :param menu_text: The menu text
        :param order_snapshot: The order snapshot
        :return: The system prompt
        """
        return self._system_template.format_map(
            {
                "menu": menu_text,
                "order_snapshot": json.dumps(order_snapshot, indent=2),
            }
        )

    def _fallback_response(self, tool_calls_made: list) -> str:
        """
        Build the fallback response.

        :param tool_calls_made: The tool calls made
        :return: The fallback response
        """
        if not tool_calls_made:
            return "How can I help you?"
        last = tool_calls_made[-1]
        name = last["name"]
        result = last["result"]
        if name == "add_item":
            item = result.get("item", {})
            return f"Added {item.get('name', 'item')}. Anything else?"
        if name == "modify_item":
            return "Updated your order. Anything else?"
        if name == "remove_item":
            item = result.get("item", {})
            return f"Removed {item.get('name', 'item')}. Anything else?"
        if name == "submit_order":
            mcp = result.get("mcp_result", {})
            if mcp.get("success"):
                return f"Order submitted! Order #{mcp.get('order_id')}."
            return f"Submission failed: {mcp.get('error', 'unknown error')}."
        if name == "confirm_order":
            summary = result.get("summary_text", "")
            if summary:
                return f"{summary}\n\nReady to submit?"
            return "Here's your order. Ready to submit?"
        if name == "cancel_order":
            return "Order cancelled. Start fresh whenever you're ready!"
        if name == "get_menu":
            menu = result.get("menu", "")
            if menu:
                return menu
            return "Sorry, the menu is currently unavailable."
        if name == "view_order":
            order = result.get("order", {})
            if order.get("items"):
                lines = []
                for item in order["items"]:
                    lines.append(
                        f"  - {item['name']} x{item['quantity']} — ${item['line_total']:.2f}"
                    )
                lines.append(f"  Total: ${order['total']:.2f}")
                return "Here's your current order:\n" + "\n".join(lines)
            return "Your order is empty."
        if name == "set_special_instructions":
            instructions = result.get("instructions", "")
            return f'Special instructions noted: "{instructions}". Anything else?'
        return "Done! Anything else?"
