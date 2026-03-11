from src.models.intent import ParsedIntent
from src.models.menu import Menu, MenuItem
from src.models.order import Order, OrderItem


class OrderError(Exception):
    """Raised when an order operation fails validation."""

    pass


class OrderManager:
    def __init__(self, menu: Menu):
        self.menu = menu
        self.order = Order()

    def add_item(self, intent: ParsedIntent) -> OrderItem:
        """
        Add an item to the order. Applies defaults, validates, calculates price.

        :param intent: The parsed intent
        :return: The added order item
        """
        menu_item = self.menu.find_by_id(intent.item_id)
        if not menu_item:
            raise OrderError(f"Unknown menu item: {intent.item_id}")

        resolved_options = self._resolve_options(menu_item, intent.options or {})

        extras = intent.extras_add or []
        for extra_id in extras:
            if not menu_item.validate_extra(extra_id):
                raise OrderError(f"Invalid extra '{extra_id}' for {menu_item.name}")

        unit_price = self._calculate_unit_price(menu_item, resolved_options, extras)

        item = OrderItem(
            item_id=menu_item.id,
            name=menu_item.name,
            quantity=intent.quantity or 1,
            options=resolved_options,
            extras=extras,
            unit_price=unit_price,
            special_instructions=intent.special_instructions,
        )
        self.order.items.append(item)
        return item

    def modify_item(self, intent: ParsedIntent) -> OrderItem:
        """
        Modify an existing order item. Merges options/extras, recalculates price.

        :param intent: The parsed intent
        :return: The modified order item
        """
        target = self._resolve_target(intent)
        menu_item = self.menu.find_by_id(target.item_id)

        if intent.options:
            for key, value in intent.options.items():
                if not menu_item.validate_option(key, value):
                    opt = menu_item.options.get(key)
                    valid = opt.choices if opt else []
                    raise OrderError(f"Invalid choice '{value}' for {key}. Valid: {valid}")
                target.options[key] = value

        if intent.extras_add:
            for extra_id in intent.extras_add:
                if not menu_item.validate_extra(extra_id):
                    raise OrderError(f"Invalid extra '{extra_id}' for {menu_item.name}")
                if extra_id not in target.extras:
                    target.extras.append(extra_id)

        if intent.extras_remove:
            for extra_id in intent.extras_remove:
                if extra_id in target.extras:
                    target.extras.remove(extra_id)

        if intent.quantity is not None:
            target.quantity = intent.quantity

        if intent.special_instructions is not None:
            target.special_instructions = intent.special_instructions

        target.unit_price = self._calculate_unit_price(menu_item, target.options, target.extras)
        return target

    def remove_item(self, intent: ParsedIntent) -> OrderItem:
        """Remove an item from the order. Returns the removed item."""
        target = self._resolve_target(intent)
        self.order.items.remove(target)
        return target

    def clear(self) -> None:
        self.order = Order()

    def set_special_instructions(self, instructions: str) -> None:
        self.order.special_instructions = instructions

    def get_snapshot(self) -> dict:
        """Return a serializable snapshot of the current order."""
        return {
            "items": [
                {
                    "uid": item.uid,
                    "name": item.name,
                    "quantity": item.quantity,
                    "options": item.options,
                    "extras": item.extras,
                    "unit_price": item.unit_price,
                    "line_total": round(item.unit_price * item.quantity, 2),
                }
                for item in self.order.items
            ],
            "total": self.order.total,
            "item_count": len(self.order.items),
            "special_instructions": self.order.special_instructions,
        }

    def pre_submit_check(self) -> str | None:
        """Returns an error message if the order can't be submitted, else None."""
        if self.order.is_empty:
            return "Your order is empty."
        return None

    # --- Private helpers ---

    def _resolve_options(self, menu_item: MenuItem, user_options: dict | None) -> dict[str, str]:
        """
        Apply user-provided options and fill defaults for unspecified ones.

        :param menu_item: The menu item to resolve options for
        :param user_options: The user-provided options
        :return: The resolved options
        """
        resolved = {}
        for opt_name, opt_config in menu_item.options.items():
            if opt_name in user_options:
                choice = user_options[opt_name]
                if not menu_item.validate_option(opt_name, choice):
                    raise OrderError(
                        f"Invalid choice '{choice}' for {opt_name}. Valid: {opt_config.choices}"
                    )
                resolved[opt_name] = choice
            elif opt_config.default:
                resolved[opt_name] = opt_config.default
            elif opt_config.required:
                raise OrderError(f"Required option '{opt_name}' not specified for {menu_item.name}")
        return resolved

    def _calculate_unit_price(
        self, menu_item: MenuItem, options: dict[str, str], extras: list[str]
    ) -> float:
        """
        unit_price = base_price + option modifiers + extras prices.

        :param menu_item: The menu item to calculate the unit price for
        :param options: The options to calculate the unit price for
        :param extras: The extras to calculate the unit price for
        :return: The calculated unit price
        """
        price = menu_item.base_price
        for opt_name, choice in options.items():
            price += menu_item.get_option_modifier(opt_name, choice)
        for extra_id in extras:
            extra_price = menu_item.get_extra_price(extra_id)
            if extra_price:
                price += extra_price
        return round(price, 2)

    def _resolve_target(self, intent: ParsedIntent) -> OrderItem:
        """
        Find which OrderItem the user is referring to.

        :param intent: The parsed intent
        :return: The target order item
        """
        if self.order.is_empty:
            raise OrderError("No items in your order to modify.")

        # By UID
        if intent.target_uid:
            for item in self.order.items:
                if item.uid == intent.target_uid:
                    return item
            raise OrderError("Item not found in order.")

        # By index
        if intent.target_index is not None:
            if 0 <= intent.target_index < len(self.order.items):
                return self.order.items[intent.target_index]
            raise OrderError(f"Invalid item number. You have {len(self.order.items)} item(s).")

        # By item_id: if exactly one match, use it
        if intent.item_id:
            matches = [i for i in self.order.items if i.item_id == intent.item_id]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise OrderError(
                    f"You have {len(matches)} {matches[0].name}s in your order. "
                    f"Which one? (1-{len(matches)})"
                )

        # Last item added as fallback ("make it large", "add cheese to that")
        return self.order.items[-1]
