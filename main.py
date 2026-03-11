import os

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from agent import FoodOrderAgent

console = Console()
VERBOSE = os.environ.get("VERBOSE", "").lower() in ("1", "true", "yes")


def _print_tool_calls(response: dict) -> None:
    """Print tool calls made this turn when VERBOSE=1."""
    tool_calls = response.get("tool_calls", [])      # MCP calls (submit_order)
    internal = response.get("_tool_calls_made", [])  # all LLM tool calls

    calls = internal or tool_calls
    if not calls:
        return

    lines = []
    for tc in calls:
        name = tc.get("name", "?")
        args = tc.get("args") or tc.get("arguments") or {}
        result = tc.get("result", {})
        status = result.get("status") or ("✓" if result.get("success") else result.get("error", ""))
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
        lines.append(f"  [dim]→ {name}({arg_str})  [{status}][/dim]")

    console.print(Text.from_markup("\n".join(lines)))


def main():
    agent = FoodOrderAgent()
    console.print(Panel.fit("Food Order Assistant", border_style="cyan"))
    hint = "Type [bold]quit[/bold] to exit  •  [bold]VERBOSE=1[/bold] to see tool calls"
    console.print(hint + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(Panel.fit("Goodbye!", border_style="cyan"))
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print(Panel.fit("Goodbye!", border_style="cyan"))
            break

        response = agent.send(user_input)

        if VERBOSE:
            _print_tool_calls(response)

        console.print(Panel(Markdown(response["message"]), title="Assistant", border_style="green"))
        console.print(Rule(style="dim"))


if __name__ == "__main__":
    main()
