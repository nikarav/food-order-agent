import argparse
import asyncio
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


def _run_text_mode() -> None:
    """Interactive text-based CLI (original behaviour, unchanged)."""
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


def _run_voice_mode() -> None:
    """Voice-based CLI — hands-free conversation via mic + speaker."""
    from orderbot.utils.config import load_configurations
    from orderbot.voice import VoiceConfig, VoiceSession

    config = load_configurations("config/agent.yaml")
    voice_data = config.get("voice", {})

    if not voice_data or not voice_data.get("elevenlabs_api_key"):
        console.print(
            "[red]Voice mode requires ELEVENLABS_API_KEY.[/red]\n"
            "Add it to your .env file, then re-run with --voice.\n"
            "Falling back to text mode.\n"
        )
        _run_text_mode()
        return

    try:
        voice_config = VoiceConfig(**voice_data)
    except Exception as exc:
        console.print(f"[red]Invalid voice config:[/red] {exc}\nFalling back to text mode.\n")
        _run_text_mode()
        return

    agent = FoodOrderAgent()
    session = VoiceSession(agent, voice_config)

    try:
        asyncio.run(session.start())
    except RuntimeError as exc:
        console.print(
            f"[yellow]Voice mode unavailable:[/yellow] {exc}\nFalling back to text mode.\n"
        )
        _run_text_mode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Food Order Assistant")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice mode (requires microphone + ELEVENLABS_API_KEY in .env)",
    )
    args = parser.parse_args()

    if args.voice:
        _run_voice_mode()
    else:
        _run_text_mode()


if __name__ == "__main__":
    main()
