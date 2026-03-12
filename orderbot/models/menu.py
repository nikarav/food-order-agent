from typing import Optional

import yaml
from pydantic import BaseModel


class MenuExtra(BaseModel):
    id: str
    price: float


class MenuOptionConfig(BaseModel):
    type: str
    required: bool
    choices: list[str]
    default: Optional[str] = None
    price_modifier: Optional[dict[str, float]] = None


class MenuItem(BaseModel):
    id: str
    name: str
    base_price: float
    options: dict[str, MenuOptionConfig]
    extras: Optional[dict] = None

    def get_extras_list(self) -> list[MenuExtra]:
        """
        Get the list of extras for the menu item.

        :return: A list of menu extras
        """
        if not self.extras or "choices" not in self.extras:
            return []
        return [MenuExtra(**e) for e in self.extras["choices"]]

    def get_extra_price(self, extra_id: str) -> Optional[float]:
        """
        Get the price of an extra for the menu item.

        :param extra_id: The ID of the extra to get the price of
        :return: The price of the extra if found, None otherwise
        """
        for extra in self.get_extras_list():
            if extra.id == extra_id:
                return extra.price
        return None

    def get_option_modifier(self, option_name: str, choice: str) -> float:
        """
        Get the modifier for an option for the menu item.

        :param option_name: The name of the option to get the modifier for
        :param choice: The choice of the option to get the modifier for
        :return: The modifier for the option if found, 0.0 otherwise
        """
        opt = self.options.get(option_name)
        if opt and opt.price_modifier:
            return opt.price_modifier.get(choice, 0.0)
        return 0.0

    def validate_option(self, option_name: str, choice: str) -> bool:
        """
        Validate an option for the menu item.

        :param option_name: The name of the option to validate
        :param choice: The choice of the option to validate
        :return: True if the option is valid, False otherwise
        """
        opt = self.options.get(option_name)
        return opt is not None and choice in opt.choices

    def validate_extra(self, extra_id: str) -> bool:
        """
        Validate an extra for the menu item.

        :param extra_id: The ID of the extra to validate
        :return: True if the extra is valid, False otherwise
        """
        return any(e.id == extra_id for e in self.get_extras_list())


class Menu(BaseModel):
    items: list[MenuItem]

    def find_by_id(self, item_id: str) -> Optional[MenuItem]:
        """
        Find a menu item by its ID.

        :param item_id: The ID of the menu item to find
        :return: The menu item if found, None otherwise
        """
        return next((i for i in self.items if i.id == item_id), None)

    def find_by_name_fuzzy(self, query: str) -> list[MenuItem]:
        """
        Find menu items by fuzzy name matching.

        :param query: The query to search for
        :return: A list of menu items that match the query
        """
        query_lower = query.lower()
        return [i for i in self.items if query_lower in i.name.lower()]

    @classmethod
    def from_yaml(cls, path: str) -> "Menu":
        """
        Load a menu from a YAML file.

        :param path: The path to the YAML file
        :return: The loaded menu
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(items=[MenuItem(**item) for item in data["menu"]])

    def to_display_string(self) -> str:
        """
        Render the menu as clean, human-readable text for customer display.

        :return: Formatted menu string
        """
        lines = []
        for item in self.items:
            lines.append(f"  {item.name} — ${item.base_price:.2f}")

            size_opt = item.options.get("size")
            if size_opt and size_opt.price_modifier:
                sizes = []
                for choice in size_opt.choices:
                    mod = size_opt.price_modifier.get(choice, 0)
                    total = item.base_price + mod
                    sizes.append(f"{choice} ${total:.2f}")
                lines.append(f"    Sizes: {', '.join(sizes)}")

            for opt_name, opt in item.options.items():
                if opt_name == "size":
                    continue
                choices = ", ".join(c.replace("_", " ") for c in opt.choices)
                lines.append(f"    {opt_name.replace('_', ' ').title()}: {choices}")

            extras = item.get_extras_list()
            if extras:
                extras_str = ", ".join(f"{e.id.replace('_', ' ')} (+${e.price:.2f})" for e in extras)
                lines.append(f"    Add-ons: {extras_str}")

            lines.append("")
        return "\n".join(lines)

    def to_prompt_string(self) -> str:
        """
        Render the menu as structured text for LLM prompts.

        :return: The menu as a string
        """
        lines = []
        for item in self.items:
            lines.append(f"- {item.id}: {item.name} (base: ${item.base_price:.2f})")
            for opt_name, opt in item.options.items():
                if opt.price_modifier:
                    choices_str = ", ".join(
                        f"{c}(+${opt.price_modifier.get(c, 0):.2f})" for c in opt.choices
                    )
                else:
                    choices_str = ", ".join(opt.choices)
                req = "required" if opt.required else "optional"
                default_str = f", default: {opt.default}" if opt.default else ""
                lines.append(f"  {opt_name}: [{choices_str}] ({req}{default_str})")
            extras = item.get_extras_list()
            if extras:
                extras_str = ", ".join(f"{e.id}(+${e.price:.2f})" for e in extras)
                lines.append(f"  extras: {extras_str}")
            lines.append("")
        return "\n".join(lines)
