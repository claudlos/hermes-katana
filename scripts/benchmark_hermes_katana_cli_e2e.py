#!/usr/bin/env python3
"""HermesKatana Hermes CLI end-to-end benchmark scaffold.

This is intentionally separate from the sandbox benchmark because real Hermes CLI
runs include provider latency, token variance, auth state, and config loading. By
default it writes isolated HERMES_HOME profiles and a runnable plan; pass
``--execute`` to actually spawn Hermes CLI commands.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"


@dataclass(frozen=True)
class E2EVariant:
    name: str
    katana_enabled: bool
    scabbard_profile: str | None = None
    scabbard_backend: str | None = None
    scabbard_device: str | None = None


VARIANTS = [
    E2EVariant("base", False),
    E2EVariant("minilm", True, "katana_v15_minilm", "onnx", None),
    E2EVariant("v15-deberta", True, "katana_v15_large", "torch", "cuda"),
]

DEFAULT_PROMPTS = [
    "Use the file tools to inspect pyproject.toml and tell me the project name.",
    "Use terminal to print the current working directory, then stop.",
]


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value)


def write_variant_home(base_home: Path, variant: E2EVariant, provider: str, model: str, route_mode: str) -> Path:
    home = base_home / variant.name
    home.mkdir(parents=True, exist_ok=True)
    plugin_block = ""
    if variant.katana_enabled:
        plugin_block = f"""
plugins:
  katana:
    enabled: true
    audit_enabled: true
    audit_log_allow: true
    policy_preset: permissive
    scabbard_enabled: true
    scabbard_profile: {_yaml_scalar(variant.scabbard_profile)}
    scabbard_backend: {_yaml_scalar(variant.scabbard_backend)}
    scabbard_device: {_yaml_scalar(variant.scabbard_device)}
    scabbard_route_mode: {_yaml_scalar(route_mode)}
    scabbard_scan_outputs: true
    scabbard_audit_routes: true
"""
    config = f"""model:
  default: {_yaml_scalar(model)}
  provider: {_yaml_scalar(provider)}
toolsets:
- file
- terminal
agent:
  max_turns: 12
  tool_use_enforcement: auto
terminal:
  backend: local
  cwd: {_yaml_scalar(str(ROOT))}
  timeout: 120
compression:
  enabled: false
{plugin_block}
"""
    (home / "config.yaml").write_text(config)
    (home / ".env").write_text("# E2E benchmark home. Secrets are inherited from process env.\n")
    return home


def run_one(home: Path, prompt: str, provider: str, model: str, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PWD"] = str(ROOT)
    env["TERMINAL_CWD"] = str(ROOT)
    cmd = [
        "hermes",
        "chat",
        "--ignore-user-config",
        "--ignore-rules",
        "--source",
        "katana-e2e-benchmark",
        "--provider",
        provider,
        "--model",
        model,
        "--max-turns",
        "12",
        "-q",
        prompt,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=timeout)
    wall_ms = (time.perf_counter() - t0) * 1000
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_ms": wall_ms,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai-codex")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--variants", default="base,minilm,v15-deberta")
    ap.add_argument("--route-mode", choices=("off", "content_only", "balanced", "paranoid"), default="balanced")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument(
        "--execute", action="store_true", help="Actually run Hermes CLI. Default only writes runnable plan/configs."
    )
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (RESULTS_ROOT / f"hermes_katana_cli_e2e_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
    prompts = args.prompt or DEFAULT_PROMPTS
    homes_root = out_dir / "hermes_homes"

    selected = [v for v in VARIANTS if v.name in wanted]
    payload: dict[str, Any] = {
        "timestamp": ts,
        "root": str(ROOT),
        "provider": args.provider,
        "model": args.model,
        "route_mode": args.route_mode,
        "execute": args.execute,
        "environment": {"python": sys.version.replace("\n", " "), "platform": platform.platform()},
        "variants": [],
    }

    for variant in selected:
        home = write_variant_home(homes_root, variant, args.provider, args.model, args.route_mode)
        entry: dict[str, Any] = {"variant": variant.name, "home": str(home), "runs": []}
        for prompt in prompts:
            if args.execute:
                result = run_one(home, prompt, args.provider, args.model, args.timeout)
            else:
                result = {
                    "dry_run": True,
                    "prompt": prompt,
                    "command": " ".join(
                        [
                            "HERMES_HOME=" + str(home),
                            "hermes chat --ignore-user-config --ignore-rules",
                            "--source katana-e2e-benchmark",
                            f"--provider {args.provider}",
                            f"--model {args.model}",
                            "--max-turns 12",
                            "-q " + json.dumps(prompt),
                        ]
                    ),
                }
            entry["runs"].append(result)
        payload["variants"].append(entry)

    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    lines = [
        "# HermesKatana CLI e2e benchmark",
        "",
        f"- Timestamp: `{ts}`",
        f"- Provider/model: `{args.provider}` / `{args.model}`",
        f"- Route mode: `{args.route_mode}`",
        f"- Execute: `{args.execute}`",
        "",
        "## Variant homes",
        "",
    ]
    for variant in payload["variants"]:
        lines.append(f"- `{variant['variant']}`: `{variant['home']}`")
    lines.extend(["", "## Notes", ""])
    lines.append("- Default mode is dry-run so the script can be used in CI without provider calls.")
    lines.append("- Pass `--execute` for real Hermes CLI timing; results include provider/model latency.")
    lines.append("- Generated homes isolate config; commands also set PWD and TERMINAL_CWD to the repo root.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[e2e] report: {out_dir / 'report.md'}")
    print(f"[e2e] json:   {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
