"""Agent.AI MCP client with OAuth token management."""

from __future__ import annotations

import json
import time
import logging

import httpx

logger = logging.getLogger("agentai.client")

DEFAULT_MCP_TOOL = "call_an_llm_copy"


def create_shared_http_client() -> httpx.AsyncClient:
    """Create a shared httpx.AsyncClient for connection reuse."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=30),
        headers={"Content-Type": "application/json"},
    )


class AgentAIClient:
    """Wraps Agent.AI MCP JSON-RPC calls with OAuth token management."""

    def __init__(
        self,
        *,
        access_token: str,
        refresh_token: str,
        client_id: str,
        agent_mcp_base: str,
        expires_at: float = 0,
        mcp_tool: str = "",
        shared_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.client_id = client_id
        self.agent_mcp_base = agent_mcp_base.rstrip("/")
        self.mcp_url = f"{self.agent_mcp_base}/mcp"
        self.expires_at = expires_at
        self.mcp_tool = mcp_tool
        self._owns_client = shared_client is None
        self.client = shared_client or create_shared_http_client()

    # Rate limit info (updated after each MCP call)
    rate_limit: int = 0
    rate_remaining: int = 0
    rate_reset: str = ""

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    # ── OAuth ───────────────────────────────────────────────────────

    def token_expired(self) -> bool:
        if not self.expires_at:
            return True
        # expires_at may be in ms or seconds
        exp = self.expires_at if self.expires_at < 1e12 else self.expires_at / 1000
        return time.time() > exp - 60

    async def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh_token. Returns True on success."""
        if not self.refresh_token or not self.client_id:
            return False
        try:
            resp = await self.client.post(
                f"{self.agent_mcp_base}/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self.client_id,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data["access_token"]
                if "refresh_token" in data:
                    self.refresh_token = data["refresh_token"]
                self.expires_at = time.time() + data.get("expires_in", 86400) - 60
                logger.info("Token refreshed, expires_at=%.0f", self.expires_at)
                return True
            logger.warning("Token refresh failed: %d %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.exception("Token refresh error: %s", e)
        return False

    async def ensure_token(self) -> str:
        if self.token_expired():
            await self.refresh_access_token()
        return self.access_token

    # ── MCP calls ───────────────────────────────────────────────────

    async def call_mcp(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request to the Agent.AI MCP endpoint."""
        token = await self.ensure_token()
        payload: dict = {"jsonrpc": "2.0", "method": method, "id": 1}
        if params:
            payload["params"] = params

        resp = await self.client.post(
            self.mcp_url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )

        if resp.status_code == 401:
            # Token might have just expired, try refresh once
            ok = await self.refresh_access_token()
            if ok:
                resp = await self.client.post(
                    self.mcp_url,
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    json=payload,
                )

        if resp.status_code != 200:
            raise RuntimeError(f"MCP HTTP {resp.status_code}: {resp.text[:500]}")

        # Capture rate limit headers
        self._parse_rate_headers(resp)

        result = resp.json()
        if "error" in result:
            raise RuntimeError(f"MCP error: {result['error']}")

        return result.get("result", {})

    def _parse_rate_headers(self, resp: httpx.Response) -> None:
        try:
            self.rate_limit = int(resp.headers.get("ratelimit-limit", "0").split(":")[0])
            self.rate_remaining = int(resp.headers.get("ratelimit-remaining", "0"))
            self.rate_reset = resp.headers.get("ratelimit-reset-date", "")
        except (ValueError, AttributeError):
            pass

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a specific MCP tool and return the text content.
        If tool_name fails with unknown tool, auto-discover and retry."""
        try:
            result = await self.call_mcp("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })
        except RuntimeError as e:
            if "Unknown tool" in str(e) and not self.mcp_tool:
                # Auto-discover available tool
                tools = await self.list_tools()
                if tools:
                    self.mcp_tool = tools[0].get("name", "")
                    logger.info("Auto-discovered tool: %s", self.mcp_tool)
                    result = await self.call_mcp("tools/call", {
                        "name": self.mcp_tool,
                        "arguments": arguments,
                    })
                else:
                    raise
            else:
                raise
        content_list = result.get("content", [])
        return "\n".join(
            c.get("text", "") for c in content_list if c.get("type") == "text"
        )

    async def list_tools(self) -> list:
        """List available MCP tools."""
        result = await self.call_mcp("tools/list")
        return result.get("tools", [])
