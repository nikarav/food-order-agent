import logging

from src.models.intent import IntentType, ParsedIntent
from src.order.manager import OrderError, OrderManager

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Bridges LLM tool calls to deterministic OrderManager operations.
    ParsedIntent is used as an internal adapter — OrderManager interface is unchanged.
    """

    def __init__(self, order_manager: OrderManager, menu_text: str):
        self._om = order_manager
        self._menu_text = menu_text

    def execute(self, tool_name: str, tool_args: dict) -> dict:
        """Dispatch a tool call and return a result dict."""
        handler = getattr(self, f"_exec_{tool_name}", None)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(tool_args)
        except OrderError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.warning(f"Tool {tool_name} raised unexpected error: {e}")
            return {"error": f"Unexpected error: {e}"}

    # --- Tool handlers ---

    def _exec_add_item(self, args: dict) -> dict:
        intent = ParsedIntent(
            intent=IntentType.ADD_ITEM,
            item_id=args.get("item_id"),
            quantity=args.get("quantity"),
            options=args.get("options"),
            extras_add=args.get("extras"),
        )
        item = self._om.add_item(intent)
        return {
            "status": "added",
            "item": item.model_dump(),
            "order": self._om.get_snapshot(),
        }

    def _exec_modify_item(self, args: dict) -> dict:
        intent = ParsedIntent(
            intent=IntentType.MODIFY_ITEM,
            target_uid=args.get("target_uid"),
            target_index=args.get("target_index"),
            item_id=args.get("item_id"),
            options=args.get("options"),
            extras_add=args.get("extras_add"),
            extras_remove=args.get("extras_remove"),
            quantity=args.get("quantity"),
            special_instructions=args.get("special_instructions"),
        )
        item = self._om.modify_item(intent)
        return {
            "status": "modified",
            "item": item.model_dump(),
            "order": self._om.get_snapshot(),
        }

    def _exec_remove_item(self, args: dict) -> dict:
        intent = ParsedIntent(
            intent=IntentType.REMOVE_ITEM,
            target_uid=args.get("target_uid"),
            target_index=args.get("target_index"),
            item_id=args.get("item_id"),
        )
        item = self._om.remove_item(intent)
        return {
            "status": "removed",
            "item": item.model_dump(),
            "order": self._om.get_snapshot(),
        }

    def _exec_view_order(self, args: dict) -> dict:
        return {"status": "view_order", "order": self._om.get_snapshot()}

    def _exec_get_menu(self, args: dict) -> dict:
        return {"status": "show_menu", "menu": self._menu_text}

    def _exec_confirm_order(self, args: dict) -> dict:
        check = self._om.pre_submit_check()
        if check:
            return {"error": check}
        snapshot = self._om.get_snapshot()
        lines = []
        for item in snapshot["items"]:
            opts = ", ".join(f"{k}: {v}" for k, v in item["options"].items())
            extras = ", ".join(item["extras"]) if item["extras"] else ""
            desc = item["name"]
            if opts:
                desc += f" ({opts})"
            if extras:
                desc += f" + {extras}"
            qty = item["quantity"]
            line_total = item["line_total"]
            lines.append(f"  • {desc} x{qty} — ${line_total:.2f}")
        lines.append(f"  Total: ${snapshot['total']:.2f}")
        summary_text = "\n".join(lines)
        return {
            "status": "confirm",
            "order": snapshot,
            "summary_text": summary_text,
        }

    def _exec_submit_order(self, args: dict) -> dict:
        check = self._om.pre_submit_check()
        if check:
            return {"error": check}
        return {"status": "ready_to_submit"}

    def _exec_cancel_order(self, args: dict) -> dict:
        self._om.clear()
        return {"status": "cancelled", "order": self._om.get_snapshot()}

    def _exec_set_special_instructions(self, args: dict) -> dict:
        instructions = args.get("instructions", "")
        self._om.set_special_instructions(instructions)
        return {
            "status": "special_instructions_set",
            "instructions": instructions,
            "order": self._om.get_snapshot(),
        }
