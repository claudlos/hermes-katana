"""Rich rendering helpers for the CLI command module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from rich import box
from rich.table import Table


def display_scan_result(console: Any, result: Any, title: str) -> None:
    """Display a ScanResult using rich formatting."""
    verdict_colors = {
        "allow": "green",
        "warn": "yellow",
        "block": "red",
    }
    color = verdict_colors.get(result.verdict.value, "white")

    console.print("\n[bold]Scan Results[/bold]")
    console.print(f"   {title}")
    console.print(f"   Verdict: [{color}][bold]{result.verdict.value.upper()}[/bold][/{color}]")
    console.print(f"   Risk Score: {result.risk_score:.2f}")

    if result.has_findings:
        table = Table(title="Findings", box=box.SIMPLE)
        table.add_column("Category", style="bold")
        table.add_column("Severity")
        table.add_column("Details")

        for finding in result.injection_findings:
            table.add_row(
                f"Injection ({finding.category.value})",
                "[red]high[/red]",
                finding.description if hasattr(finding, "description") else str(finding),
            )

        for finding in result.secret_findings:
            sev = finding.severity.value if hasattr(finding, "severity") else "high"
            sev_color = {"critical": "red", "high": "red", "medium": "yellow", "low": "dim"}.get(sev, "white")
            table.add_row(
                f"Secret ({finding.category.value})",
                f"[{sev_color}]{sev}[/{sev_color}]",
                finding.description if hasattr(finding, "description") else str(finding),
            )

        for finding in result.command_findings:
            sev = finding.severity.value if hasattr(finding, "severity") else "high"
            sev_color = {"critical": "red", "high": "red", "medium": "yellow", "low": "dim"}.get(sev, "white")
            table.add_row(
                f"Command ({finding.category.value})",
                f"[{sev_color}]{sev}[/{sev_color}]",
                finding.description if hasattr(finding, "description") else str(finding),
            )

        for finding in result.content_findings:
            sev = finding.severity.value if hasattr(finding, "severity") else "medium"
            sev_color = {"critical": "red", "high": "red", "medium": "yellow", "low": "dim"}.get(sev, "white")
            table.add_row(
                f"Content ({finding.category.value})",
                f"[{sev_color}]{sev}[/{sev_color}]",
                finding.description if hasattr(finding, "description") else str(finding),
            )

        for finding in result.unicode_findings:
            sev = finding.severity.value if hasattr(finding, "severity") else "medium"
            sev_color = {"critical": "red", "high": "red", "medium": "yellow", "low": "dim"}.get(sev, "white")
            table.add_row(
                f"Unicode ({finding.category.value})",
                f"[{sev_color}]{sev}[/{sev_color}]",
                finding.description if hasattr(finding, "description") else str(finding),
            )

        console.print(table)
    else:
        console.print("   [dim]No findings.[/dim]")

    console.print(f"\n   [bold]Summary:[/bold] {result.summary}\n")


def build_installation_status_table(target_path: Path, install_status: Mapping[str, Any]) -> Table:
    """Build the Rich table for install status reporting."""
    table = Table(title=f"Installation: {target_path}", box=box.ROUNDED)
    table.add_column("Component", style="bold")
    table.add_column("Status")

    hermes_ok = install_status["hermes_detected"]
    table.add_row(
        "Hermes detected",
        "[green]OK[/green]" if hermes_ok else "[red]NO[/red]",
    )
    table.add_row(
        "Katana installed",
        "[green]OK[/green]" if install_status["installed"] else "[red]NO[/red]",
    )
    table.add_row(
        "Config exists",
        "[green]OK[/green]" if install_status["config_exists"] else "[red]NO[/red]",
    )
    table.add_row(
        "CA cert exists",
        "[green]OK[/green]" if install_status["ca_cert_exists"] else "[yellow]WARN[/yellow]",
    )

    patches = install_status["patches"]
    table.add_row(
        "Patches",
        f"{patches['applied']}/{patches['total']} applied",
    )
    return table


def print_installation_messages(console: Any, *, issues: Sequence[str], warnings: Sequence[str]) -> None:
    """Print install issues and warnings after the main status table."""
    if issues:
        console.print("\n   [red]Issues:[/red]")
        for issue in issues:
            console.print(f"     - {issue}")

    if warnings:
        console.print("\n   [yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"     - {warning}")


def build_modules_status_table(module_checks: Sequence[tuple[str, str]]) -> Table:
    """Build the Rich table describing module import availability."""
    table = Table(title="Modules", box=box.ROUNDED)
    table.add_column("Module", style="bold")
    table.add_column("Status")
    table.add_column("Info")

    for name, module_path in module_checks:
        try:
            __import__(module_path)
            table.add_row(name, "[green]Loaded[/green]", "")
        except ImportError as exc:
            table.add_row(name, "[yellow]Not available[/yellow]", str(exc))

    return table


def build_environment_table(env_vars: Sequence[str]) -> Table:
    """Build the Rich table for managed Katana environment variables."""
    table = Table(title="Environment", box=box.ROUNDED)
    table.add_column("Variable", style="bold")
    table.add_column("Value")

    for var in env_vars:
        value = os.environ.get(var)
        if value:
            table.add_row(var, value)
        else:
            table.add_row(var, "[dim]not set[/dim]")

    return table
