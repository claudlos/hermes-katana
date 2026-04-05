class GatewayHandler:
    async def handle_request(self, request: dict) -> dict:
        """Handle an incoming gateway request."""
        tool_name = request.get("tool")
        args = request.get("args", {})
        return {"tool": tool_name, "args": args}
