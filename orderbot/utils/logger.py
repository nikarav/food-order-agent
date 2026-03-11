import structlog


def configure_logging(log_level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level, 20)
        ),
    )


class ConversationLogger:
    def __init__(self, log_level: str = "INFO"):
        configure_logging(log_level)
        self._log = structlog.get_logger()
        self.turn_count = 0

    def log_turn(
        self,
        user_message: str,
        tool_calls_made: list,
        response: str,
        mcp_tool_calls: list | None = None,
        order_snapshot: dict | None = None,
    ) -> None:
        self.turn_count += 1
        self._log.info(
            "conversation_turn",
            turn=self.turn_count,
            user_message=user_message,
            tool_calls_made=tool_calls_made,
            response=response,
            mcp_tool_calls=mcp_tool_calls,
            order_snapshot=order_snapshot,
        )
