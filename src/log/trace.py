from datetime import datetime, timezone
from uuid import uuid4


class ConversationTrace:
    """In-memory full conversation trace for debugging and eval."""

    def __init__(self):
        self.conversation_id = uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.turns: list[dict] = []

    def add_turn(
        self,
        user_message: str,
        tool_calls_made: list,
        response: str,
        mcp_tool_calls: list | None = None,
        order_snapshot: dict | None = None,
    ) -> None:
        self.turns.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "turn": len(self.turns) + 1,
                "user_message": user_message,
                "tool_calls_made": tool_calls_made,
                "response": response,
                "mcp_tool_calls": mcp_tool_calls,
                "order_snapshot": order_snapshot,
            }
        )

    def export(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "started_at": self.started_at,
            "total_turns": len(self.turns),
            "turns": self.turns,
        }
