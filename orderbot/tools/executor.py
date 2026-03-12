import logging

from orderbot.order.manager import OrderError, OrderManager

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Bridges LLM tool calls to deterministic OrderManager operations."""

    def __init__(self, order_manager: OrderManager, menu_text: str, menu_display_text: str = ""):
        """
        Initialize the ToolExecutor.

        :param order_manager: The order manager
        :param menu_text: The menu text for LLM prompts
        :param menu_display_text: Human-readable menu text for customer display
        """
        self._om = order_manager
        self._menu_text = menu_text
        self._menu_display_text = menu_display_text or menu_text

    def execute(self, tool_name: str, tool_args: dict) -> dict:
        """
        Dispatch a tool call and return a result dict.

        :param tool_name: The name of the tool to execute
        :param tool_args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
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
        """
        Execute the add_item tool.

        :param args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
        item = self._om.add_item(
            item_id=args["item_id"],
            quantity=args.get("quantity", 1),
            options=args.get("options"),
            extras=args.get("extras"),
            special_instructions=args.get("special_instructions"),
        )
        return {
            "status": "added",
            "item": item.model_dump(),
            "order": self._om.get_snapshot(),
        }

    def _exec_modify_item(self, args: dict) -> dict:
        """
        Execute the modify_item tool.

        :param args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
        item = self._om.modify_item(
            item_id=args.get("item_id"),
            target_uid=args.get("target_uid"),
            target_index=args.get("target_index"),
            options=args.get("options"),
            extras_add=args.get("extras_add"),
            extras_remove=args.get("extras_remove"),
            quantity=args.get("quantity"),
            special_instructions=args.get("special_instructions"),
        )
        return {
            "status": "modified",
            "item": item.model_dump(),
            "order": self._om.get_snapshot(),
        }

    def _exec_remove_item(self, args: dict) -> dict:
        """
        Execute the remove_item tool.

        :param args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
        item = self._om.remove_item(
            item_id=args.get("item_id"),
            target_uid=args.get("target_uid"),
            target_index=args.get("target_index"),
        )
        return {
            "status": "removed",
            "item": item.model_dump(),
            "order": self._om.get_snapshot(),
        }

    def _exec_view_order(self, args: dict) -> dict:
        """
        Execute the view_order tool.

        :param args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
        return {"status": "view_order", "order": self._om.get_snapshot()}

    def _exec_get_menu(self, args: dict) -> dict:
        """
        Execute the get_menu tool.

        :param args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
        return {"status": "show_menu", "menu": self._menu_display_text}

    def _exec_confirm_order(self, args: dict) -> dict:
        """
        Execute the confirm_order tool.

        :param args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
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
            line = f"  • {desc} x{qty} — ${line_total:.2f}"
            if item.get("special_instructions"):
                line += f'  [note: {item["special_instructions"]}]'
            lines.append(line)
        lines.append(f"  Total: ${snapshot['total']:.2f}")
        summary_text = "\n".join(lines)
        return {
            "status": "confirm",
            "order": snapshot,
            "summary_text": summary_text,
        }

    def _exec_submit_order(self, args: dict) -> dict:
        """
        Execute the submit_order tool.

        :param args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
        check = self._om.pre_submit_check()
        if check:
            return {"error": check}
        return {"status": "ready_to_submit"}

    def _exec_cancel_order(self, args: dict) -> dict:
        """
        Execute the cancel_order tool.

        :param args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
        self._om.clear()
        return {"status": "cancelled", "order": self._om.get_snapshot()}

    def _exec_set_special_instructions(self, args: dict) -> dict:
        """
        Execute the set_special_instructions tool.

        :param args: The arguments to the tool
        :return: A dictionary containing the result of the tool execution
        """
        instructions = args.get("instructions", "")
        self._om.set_special_instructions(instructions)
        return {
            "status": "special_instructions_set",
            "instructions": instructions,
            "order": self._om.get_snapshot(),
        }
