import threading

from orderbot.models.menu import Menu, MenuItem
from orderbot.models.order import Order, OrderItem


class OrderError(Exception):
    """Raised when an order operation fails validation."""

    pass


class OrderManager:
    def __init__(self, menu: Menu):
        """
        Initialize the OrderManager.

        :param menu: The menu
        """
        self.menu = menu
        self.order = Order()
        self._lock = threading.Lock()

    def add_item(
        self,
        item_id: str,
        quantity: int = 1,
        options: dict | None = None,
        extras: list[str] | None = None,
        special_instructions: str | None = None,
    ) -> OrderItem:
        """
        Add an item to the order. Applies defaults, validates, calculates price.

        :param item_id: The ID of the item to add
        :param quantity: The quantity of the item to add
        :param options: The options for the item
        :param extras: The extras for the item
        :param special_instructions: The special instructions for the item
        :return: The added order item
        """
        with self._lock:
            menu_item = self.menu.find_by_id(item_id)
            if not menu_item:
                raise OrderError(f"Unknown menu item: {item_id}")

            resolved_options = self._resolve_options(menu_item, options or {})

            extras = extras or []
            for extra_id in extras:
                if not menu_item.validate_extra(extra_id):
                    raise OrderError(f"Invalid extra '{extra_id}' for {menu_item.name}")

            unit_price = self._calculate_unit_price(menu_item, resolved_options, extras)

            item = OrderItem(
                item_id=menu_item.id,
                name=menu_item.name,
                quantity=quantity,
                options=resolved_options,
                extras=extras,
                unit_price=unit_price,
                special_instructions=special_instructions,
            )
            self.order.items.append(item)
            return item

    def modify_item(
        self,
        item_id: str | None = None,
        target_uid: str | None = None,
        target_index: int | None = None,
        options: dict | None = None,
        extras_add: list[str] | None = None,
        extras_remove: list[str] | None = None,
        quantity: int | None = None,
        special_instructions: str | None = None,
    ) -> OrderItem:
        """
        Modify an existing order item. Merges options/extras, recalculates price.

        :param item_id: The ID of the item to modify
        :param target_uid: The UID of the item to modify
        :param target_index: The index of the item to modify
        :param options: The options for the item
        :param extras_add: The extras to add to the item
        :param extras_remove: The extras to remove from the item
        :param quantity: The quantity of the item to modify
        :param special_instructions: The special instructions for the item
        :return: The modified order item
        """
        with self._lock:
            target = self._resolve_target(item_id, target_uid, target_index)
            menu_item = self.menu.find_by_id(target.item_id)

            if options:
                for key, value in options.items():
                    if not menu_item.validate_option(key, value):
                        opt = menu_item.options.get(key)
                        valid = opt.choices if opt else []
                        raise OrderError(f"Invalid choice '{value}' for {key}. Valid: {valid}")
                    target.options[key] = value

            if extras_add:
                for extra_id in extras_add:
                    if not menu_item.validate_extra(extra_id):
                        raise OrderError(f"Invalid extra '{extra_id}' for {menu_item.name}")
                    if extra_id not in target.extras:
                        target.extras.append(extra_id)

            if extras_remove:
                for extra_id in extras_remove:
                    if extra_id in target.extras:
                        target.extras.remove(extra_id)

            if quantity is not None:
                target.quantity = quantity

            if special_instructions is not None:
                target.special_instructions = special_instructions

            target.unit_price = self._calculate_unit_price(
                menu_item, target.options, target.extras
            )
            return target

    def remove_item(
        self,
        item_id: str | None = None,
        target_uid: str | None = None,
        target_index: int | None = None,
    ) -> OrderItem:
        """
        Remove an item from the order. Returns the removed item.

        :param item_id: The ID of the item to remove
        :param target_uid: The UID of the item to remove
        :param target_index: The index of the item to remove
        :return: The removed order item
        """
        with self._lock:
            target = self._resolve_target(item_id, target_uid, target_index)
            self.order.items.remove(target)
            return target

    def clear(self) -> None:
        """Clear the order."""
        with self._lock:
            self.order = Order()

    def set_special_instructions(self, instructions: str) -> None:
        """Set the special instructions for the order."""
        with self._lock:
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

    def _resolve_options(self, menu_item: MenuItem, user_options: dict) -> dict[str, str]:
        """
        Apply user-provided options and fill defaults for unspecified ones.

        :param menu_item: The menu item to resolve the options for
        :param user_options: The user options to resolve
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
        Calculate the unit price for an item.

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

    def _resolve_target(
        self,
        item_id: str | None,
        target_uid: str | None,
        target_index: int | None,
    ) -> OrderItem:
        """
        Find which OrderItem the user is referring to.

        :param item_id: The ID of the item to resolve the target for
        :param target_uid: The UID of the item to resolve the target for
        :param target_index: The index of the item to resolve the target for
        :return: The resolved target order item
        """
        if self.order.is_empty:
            raise OrderError("No items in your order to modify.")

        # By UID
        if target_uid:
            for item in self.order.items:
                if item.uid == target_uid:
                    return item
            raise OrderError("Item not found in order.")

        # By index
        if target_index is not None:
            if 0 <= target_index < len(self.order.items):
                return self.order.items[target_index]
            raise OrderError(f"Invalid item number. You have {len(self.order.items)} item(s).")

        # By item_id: if exactly one match, use it
        if item_id:
            matches = [i for i in self.order.items if i.item_id == item_id]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise OrderError(
                    f"You have {len(matches)} {matches[0].name}s in your order. "
                    f"Which one? (1-{len(matches)})"
                )

        # Last item added as fallback ("make it large", "add cheese to that")
        return self.order.items[-1]
