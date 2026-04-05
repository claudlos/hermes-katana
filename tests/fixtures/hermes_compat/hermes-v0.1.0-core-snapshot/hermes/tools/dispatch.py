from typing import Any


class ToolDispatcher:
    def get_tool(self, tool_name: str):
        raise NotImplementedError

    async def dispatch_tool(self, tool_name: str, args: dict) -> Any:
        """Dispatch a tool call."""
        tool = self.get_tool(tool_name)
        result = await tool.execute(**args)
        return result
