from google.genai import types

add_item_decl = types.FunctionDeclaration(
    name="add_item",
    description=(
        "Add a menu item to the order. Use exact item_id values from the menu "
        "(e.g. classic_burger, fries, soda). For multi-item requests, call this "
        "once per item."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "Exact menu item ID (e.g. classic_burger, margherita, fries)",
            },
            "quantity": {
                "type": "integer",
                "description": "Number of this item to add (default: 1)",
            },
            "options": {
                "type": "object",
                "description": (
                    "Item options as key-value pairs "
                    "(e.g. {\"size\": \"large\", \"patty\": \"chicken\"})"
                ),
                "additionalProperties": {"type": "string"},
            },
            "extras": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra topping/add-on IDs (e.g. [\"cheese\", \"bacon\"])",
            },
        },
        "required": ["item_id"],
    },
)

modify_item_decl = types.FunctionDeclaration(
    name="modify_item",
    description=(
        "Modify an existing item in the order. Identify the target by uid, "
        "0-based index, or item_id. If no target is given, defaults to the last item added."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "target_uid": {
                "type": "string",
                "description": "Unique ID of the specific order item to modify",
            },
            "target_index": {
                "type": "integer",
                "description": "0-based position in the order (0=first item, 1=second, etc.)",
            },
            "item_id": {
                "type": "string",
                "description": "Item ID to match when uid/index not provided",
            },
            "options": {
                "type": "object",
                "description": "Options to update (e.g. {\"size\": \"large\"})",
                "additionalProperties": {"type": "string"},
            },
            "extras_add": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra IDs to add",
            },
            "extras_remove": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra IDs to remove",
            },
            "quantity": {
                "type": "integer",
                "description": "New quantity for the item",
            },
            "special_instructions": {
                "type": "string",
                "description": "Item-level special instructions",
            },
        },
    },
)

remove_item_decl = types.FunctionDeclaration(
    name="remove_item",
    description="Remove an item from the order entirely.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "target_uid": {"type": "string", "description": "Unique ID of the order item"},
            "target_index": {
                "type": "integer",
                "description": "0-based position in the order",
            },
            "item_id": {"type": "string", "description": "Item ID to match"},
        },
    },
)

view_order_decl = types.FunctionDeclaration(
    name="view_order",
    description="Show the customer their current order with all items and the running total.",
    parameters_json_schema={"type": "object", "properties": {}},
)

get_menu_decl = types.FunctionDeclaration(
    name="get_menu",
    description="Show the full menu to the customer.",
    parameters_json_schema={"type": "object", "properties": {}},
)

confirm_order_decl = types.FunctionDeclaration(
    name="confirm_order",
    description=(
        "Show the complete order summary for customer review before submission. "
        "Call this when the customer says they are done ordering "
        "(e.g. \"that's it\", \"done\", \"nothing else\", \"place my order\")."
    ),
    parameters_json_schema={"type": "object", "properties": {}},
)

submit_order_decl = types.FunctionDeclaration(
    name="submit_order",
    description=(
        "Submit the confirmed order for processing. "
        "Only call AFTER confirm_order has been called AND the customer explicitly "
        "approves (e.g. \"yes\", \"go ahead\", \"place it\", \"submit\")."
    ),
    parameters_json_schema={"type": "object", "properties": {}},
)

cancel_order_decl = types.FunctionDeclaration(
    name="cancel_order",
    description="Cancel the current order and clear all items.",
    parameters_json_schema={"type": "object", "properties": {}},
)

set_special_instructions_decl = types.FunctionDeclaration(
    name="set_special_instructions",
    description="Set special instructions for the entire order (e.g. allergies, delivery notes).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "instructions": {
                "type": "string",
                "description": "The special instructions text",
            },
        },
        "required": ["instructions"],
    },
)

ORDER_TOOLS = types.Tool(
    function_declarations=[
        add_item_decl,
        modify_item_decl,
        remove_item_decl,
        view_order_decl,
        get_menu_decl,
        confirm_order_decl,
        submit_order_decl,
        cancel_order_decl,
        set_special_instructions_decl,
    ]
)
