import asyncio
import json
import logging

from google import genai
from google.genai import types

from orderbot.llm.base import LLMClient
from orderbot.tools.declarations import ORDER_TOOLS

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
        system = self._build_system_prompt(menu_text, order_snapshot)
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        contents = list(history) + [user_content]

        history_additions = [user_content]
        tool_calls_made = []

        try:
            response = await self._generate(contents, system)

            # Tool-calling loop — each iteration may produce parallel calls
            while response.function_calls:
                model_content = response.candidates[0].content
                history_additions.append(model_content)

                # Execute all function calls for this iteration in parallel
                tasks = [
                    self._run_tool(tool_executor, fc.name, dict(fc.args) if fc.args else {})
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

                response = await self._generate(list(history) + history_additions, system)

            # Final text response
            final_content = response.candidates[0].content
            history_additions.append(final_content)
            text = (response.text or "").strip() or self._fallback_response(tool_calls_made)

        except Exception as e:
            logger.warning(f"process_turn failed ({e}), returning fallback")
            text = self._fallback_response(tool_calls_made)
            # Still add user message to history so context isn't lost
            history_additions = [user_content]

        return {
            "text": text,
            "tool_calls_made": tool_calls_made,
            "history_additions": history_additions,
        }

    # --- Private helpers ---

    async def _generate(self, contents: list, system: str):
        """
        Single Gemini API call with shared model config.

        :param contents: The contents to generate
        :param system: The system prompt
        :return: The generated content
        """
        return await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                tools=[ORDER_TOOLS],
                automatic_function_calling=_AUTO_FC_OFF,
                temperature=self.temperature,
            ),
        )

    @staticmethod
    async def _run_tool(tool_executor, name: str, args: dict) -> dict:
        """
        Run one tool call in a thread pool so parallel calls don't block each other.

        :param tool_executor: The tool executor
        :param name: The name of the tool to execute
        :param args: The arguments to the tool
        :return: The result of the tool execution
        """
        return await asyncio.to_thread(tool_executor.execute, name, args)

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
