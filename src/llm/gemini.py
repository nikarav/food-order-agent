import json
import logging
from pathlib import Path

from google import genai
from google.genai import types

from src.llm.base import LLMClient
from src.tools.declarations import ORDER_TOOLS

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    def __init__(self, config):
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.model_name
        self._system_template = (Path(config.prompts.dir) / "system.txt").read_text()

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
          2. Execute any function calls via tool_executor
          3. Feed results back until Gemini returns a text response
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
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    tools=[ORDER_TOOLS],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                    temperature=0.7,
                ),
            )

            # Tool-calling loop
            while response.function_calls:
                model_content = response.candidates[0].content
                history_additions.append(model_content)

                function_response_parts = []
                for fc in response.function_calls:
                    args = dict(fc.args) if fc.args else {}
                    result = tool_executor.execute(fc.name, args)
                    tool_calls_made.append(
                        {"name": fc.name, "args": args, "result": result}
                    )
                    function_response_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response=result,
                        )
                    )

                tool_response_content = types.Content(
                    role="user", parts=function_response_parts
                )
                history_additions.append(tool_response_content)

                # Send results back to model
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=list(history) + history_additions,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        tools=[ORDER_TOOLS],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                        temperature=0.7,
                    ),
                )

            # Final text response
            final_content = response.candidates[0].content
            history_additions.append(final_content)
            text = (response.text or "").strip() or "How can I help you?"

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

    def _build_system_prompt(self, menu_text: str, order_snapshot: dict) -> str:
        snapshot_str = json.dumps(order_snapshot, indent=2)
        return (
            self._system_template
            .replace("{menu}", menu_text)
            .replace("{order_snapshot}", snapshot_str)
        )

    def _fallback_response(self, tool_calls_made: list) -> str:
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
            return "Here's your order. Ready to submit?"
        if name == "cancel_order":
            return "Order cancelled. Start fresh whenever you're ready!"
        return "Done! Anything else?"
