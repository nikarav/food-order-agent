from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from agent import FoodOrderAgent

console = Console()


def main():
    agent = FoodOrderAgent()
    console.print(Panel.fit("Food Order Assistant", border_style="cyan"))
    console.print("Type [bold]quit[/bold] to exit, [bold]menu[/bold] to see what we offer\n")

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
        console.print(Panel(Markdown(response["message"]), title="Assistant", border_style="green"))
        console.print(Rule(style="dim"))


if __name__ == "__main__":
    main()
