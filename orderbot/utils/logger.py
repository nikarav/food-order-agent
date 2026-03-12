import structlog


def configure_logging(log_level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
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
        tool_summary = []
        for tc in tool_calls_made:
            args = tc.get("args", {})
            arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
            status = tc["result"].get("status", tc["result"].get("error", "?"))
            tool_summary.append(f"{tc['name']}({arg_str}) → {status}")

        self._log.debug(
            "conversation_turn",
            turn=self.turn_count,
            user=user_message,
            tools=tool_summary or None,
            response=response,
            mcp=mcp_tool_calls is not None,
        )
