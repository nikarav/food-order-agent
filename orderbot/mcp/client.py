import asyncio
import json
import logging

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Wraps MCP SDK to call submit_order with retry logic."""

    def __init__(self, config):
        self._server_url = config.mcp.server_url
        self._applicant_email = config.applicant_email
        self._max_retries = int(config.max_retries)

    async def submit_order(self, payload: dict) -> dict:
        """
        Submit order via MCP submit_order tool.

        Returns:
            dict with {"success": True/False, ...}
        """
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                result = await self._call_tool(payload)
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"MCP call attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)

        return {"success": False, "error": f"Failed after retries: {last_error}"}

    async def _call_tool(self, payload: dict) -> dict:
        headers = {"X-Applicant-Email": self._applicant_email}

        async with streamablehttp_client(
            url=self._server_url,
            headers=headers,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("submit_order", arguments=payload)

                if result.isError:
                    error_text = (
                        result.content[0].text if result.content else "Unknown MCP error"
                    )
                    logger.error(f"MCP tool returned error: {error_text}")
                    return {"success": False, "error": error_text}

                if not result.content:
                    return {"success": False, "error": "Empty response from MCP server"}

                content_text = result.content[0].text
                try:
                    return json.loads(content_text)
                except json.JSONDecodeError:
                    return {"success": False, "error": f"Invalid JSON response: {content_text}"}
