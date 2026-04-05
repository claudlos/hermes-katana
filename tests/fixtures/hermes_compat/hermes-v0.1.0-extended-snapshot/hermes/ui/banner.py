from rich.console import Console

console = Console()


class Banner:
    def _build_banner(self) -> str:
        return "Hermes"

    def show_banner(self) -> None:
        """Display the Hermes startup banner."""
        console.print(self._build_banner())
