import os


class TerminalTool:
    async def execute(self, command: str, **kwargs) -> str:
        """Execute a shell command."""
        env = os.environ.copy()
        return command
