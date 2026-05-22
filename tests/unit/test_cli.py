"""Regression tests for HermesKatana CLI wiring."""

from __future__ import annotations

import json
import os
from io import StringIO
from types import SimpleNamespace
from pathlib import Path

from click.testing import CliRunner
import pytest
from rich.console import Console

import hermes_katana.artifacts as artifacts_mod
import hermes_katana.bootstrap as bootstrap_mod
import hermes_katana.cli._support as cli_support_mod
import hermes_katana.cli.main as cli_main
import hermes_katana.config as config_mod
import hermes_katana.installer as installer_mod
import hermes_katana.proxy as proxy_mod


def _test_console(stream: StringIO) -> Console:
    return Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=120,
    )


class TestCLIContracts:
    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_policy_use_persists_selected_preset(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setattr(config_mod, "_CONFIG_DIR", tmp_dir)
        monkeypatch.setattr(config_mod, "_CONFIG_FILE", tmp_dir / "config.yaml")
        # `policy use` writes KATANA_POLICY_PRESET into os.environ. Round-trip via
        # setenv+delenv so monkeypatch registers "originally absent" for restoration
        # (delenv alone with raising=False doesn't track nonexistent vars).
        if "KATANA_POLICY_PRESET" in os.environ:
            monkeypatch.setenv("KATANA_POLICY_PRESET", os.environ["KATANA_POLICY_PRESET"])
        else:
            monkeypatch.setenv("KATANA_POLICY_PRESET", "")
            monkeypatch.delenv("KATANA_POLICY_PRESET")

        result = self.runner.invoke(cli_main.main, ["policy", "use", "max"])

        assert result.exit_code == 0
        config = config_mod.load_config()
        assert config.policy_preset == "max"
        assert config.policy_path is None

        engine, source = cli_main._load_policy_engine()
        assert engine.policy_set_name == "max"
        assert source == "preset max"

    def test_vault_list_uses_real_vault_helper(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        calls: list[bool] = []

        class FakeVault:
            def list_keys(self):
                return ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

        def fake_open_vault(*, auto_create: bool):
            calls.append(auto_create)
            return FakeVault()

        monkeypatch.setattr(cli_main, "_open_vault", fake_open_vault)

        result = self.runner.invoke(cli_main.main, ["vault", "list"])

        assert result.exit_code == 0
        assert calls == [False]
        assert "OPENAI_API_KEY" in stdout.getvalue()

    def test_audit_stats_uses_default_audit_trail(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        class FakeTrail:
            def stats(self):
                return {"total_entries": 3, "file_exists": True}

        monkeypatch.setattr(cli_main, "_open_audit_trail", lambda: FakeTrail())

        result = self.runner.invoke(cli_main.main, ["audit", "stats"])

        assert result.exit_code == 0
        assert "total_entries" in stdout.getvalue()

    def test_proxy_start_builds_proxy_config(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        class FakeProxy:
            last_config = None

            def __init__(self, config=None):
                FakeProxy.last_config = config

            def start(self):
                return 1234

        monkeypatch.setattr(proxy_mod, "KatanaProxy", FakeProxy)

        result = self.runner.invoke(
            cli_main.main,
            ["proxy", "start", "--host", "0.0.0.0", "--port", "9000"],
        )

        assert result.exit_code == 0
        assert FakeProxy.last_config is not None
        assert FakeProxy.last_config.host == "0.0.0.0"
        assert FakeProxy.last_config.port == 9000

    def test_proxy_status_reads_structured_status(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        class FakeProxy:
            def __init__(self, config=None):
                self.config = config

            def status(self):
                return {
                    "running": True,
                    "host": "127.0.0.1",
                    "port": 8080,
                    "config": {
                        "host": "127.0.0.1",
                        "port": 8080,
                        "tls_verify": True,
                        "inject_credentials": True,
                    },
                }

        monkeypatch.setattr(proxy_mod, "KatanaProxy", FakeProxy)

        result = self.runner.invoke(cli_main.main, ["proxy", "status"])

        assert result.exit_code == 0
        assert "http://127.0.0.1:8080" in stdout.getvalue()

    def test_status_renders_ml_runtime_section(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setattr(
            cli_main,
            "_collect_ml_runtime_status",
            lambda: {
                "packages": {
                    "torch": {"installed": True, "version": "2.2.0"},
                    "transformers": {"installed": True, "version": "4.40.0"},
                    "onnxruntime": {"installed": True, "version": "1.17.0"},
                    "sentence_transformers": {"installed": True, "version": "3.0.0"},
                    "xgboost": {"installed": True, "version": "2.0.0"},
                },
                "deberta": {
                    "ready": True,
                    "cpu_inference_ready": True,
                    "artifact_dir": "/tmp/deberta",
                    "onnx_path": "/tmp/deberta/model.onnx",
                    "error": None,
                },
                "semantic": {
                    "backend": "contrastive",
                    "reason": "contrastive artifacts available",
                },
                "scabbard": {
                    "standard_profile_ready": True,
                    "recommended_profile": "standard",
                    "missing": [],
                },
                "protectai": {
                    "dependencies_ready": True,
                    "model_id": "ProtectAI/model",
                },
                "artifact_manifest": {
                    "ready": True,
                    "manifest_path": "/tmp/runtime_artifact_manifest.json",
                    "verified": 7,
                    "total": 7,
                    "missing": [],
                    "mismatched": [],
                    "empty": [],
                    "errors": [],
                },
                "eval": {
                    "ready": True,
                    "blockers": [],
                    "warnings": [],
                },
            },
        )

        result = self.runner.invoke(cli_main.main, ["status"])

        assert result.exit_code == 0
        output = stdout.getvalue()
        assert "ML Runtime" in output
        assert "/tmp/deberta" in output
        assert "contrastive" in output
        assert "Eval sweep" in output
        assert "Hermetic gate" in output
        assert "Artifact manifest" in output
        assert "Scabbard default" in output

    def test_collect_ml_runtime_status_reports_deberta_artifact(self):
        status = cli_support_mod.collect_ml_runtime_status()
        assert "deberta" in status
        assert "packages" in status
        assert "artifact_dir" in status["deberta"]
        assert status["deberta"]["artifact_dir"] is None or isinstance(status["deberta"]["artifact_dir"], str)
        assert status["deberta"]["ready"] == (status["deberta"]["artifact_dir"] is not None)

    def test_collect_ml_runtime_status_reports_eval_blockers(self, monkeypatch):
        monkeypatch.setattr(
            cli_support_mod,
            "_resolve_deberta_artifact",
            lambda: {
                "models_dir": "/tmp/models",
                "override": None,
                "artifact_dir": "/tmp/models/deberta",
                "checkpoint_dir": "/tmp/models/deberta/best",
                "onnx_path": None,
                "ready": True,
                "error": None,
            },
        )
        monkeypatch.setattr(
            cli_support_mod,
            "_collect_scabbard_status",
            lambda packages: {
                "standard_profile_ready": False,
                "minimal_profile_ready": False,
                "missing": ["missing TF-IDF vectorizer at /tmp/models/tfidf_vectorizer.pkl"],
                "tfidf_path": "/tmp/models/tfidf_vectorizer.pkl",
            },
        )
        monkeypatch.setattr(
            cli_support_mod,
            "_collect_semantic_status",
            lambda: {
                "backend": "minilm_fallback",
                "reason": "missing semantic index",
                "full_backend_ready": False,
                "missing": ["semantic index missing at /tmp/index"],
            },
        )
        monkeypatch.setattr(
            cli_support_mod,
            "_collect_protectai_status",
            lambda packages: {
                "dependencies_ready": True,
                "model_id": "ProtectAI/model",
                "note": "ok",
            },
        )
        monkeypatch.setattr(
            cli_support_mod,
            "_package_probe",
            lambda module_name, distribution=None: {"installed": True, "version": "1.0"},
        )

        status = cli_support_mod.collect_ml_runtime_status()

        assert status["eval"]["ready"] is False
        assert any("TF-IDF" in entry for entry in status["eval"]["blockers"])
        assert any("semantic backend degraded" in entry for entry in status["eval"]["warnings"])

    def test_collect_ml_runtime_status_invalid_override_blocks_eval(self, monkeypatch):
        monkeypatch.setenv("HERMES_KATANA_DEBERTA_MODEL_DIR", "/tmp/missing-deberta")
        monkeypatch.setattr(
            cli_support_mod,
            "_collect_scabbard_status",
            lambda packages: {
                "standard_profile_ready": True,
                "minimal_profile_ready": True,
                "missing": [],
                "missing_dependencies": [],
                "tfidf_path": "/tmp/models/tfidf_vectorizer.pkl",
            },
        )
        monkeypatch.setattr(
            cli_support_mod,
            "_collect_semantic_status",
            lambda: {
                "backend": "contrastive",
                "reason": "contrastive artifacts available",
                "full_backend_ready": True,
                "missing": [],
            },
        )
        monkeypatch.setattr(
            cli_support_mod,
            "_collect_protectai_status",
            lambda packages: {
                "dependencies_ready": True,
                "model_id": "ProtectAI/model",
                "note": "ok",
            },
        )
        monkeypatch.setattr(
            cli_support_mod,
            "_package_probe",
            lambda module_name, distribution=None: {"installed": True, "version": "1.0"},
        )
        monkeypatch.setattr(
            cli_support_mod,
            "verify_runtime_artifact_manifest",
            lambda: {
                "ready": True,
                "manifest_path": "/tmp/runtime_artifact_manifest.json",
                "verified": 7,
                "total": 7,
                "missing": [],
                "mismatched": [],
                "empty": [],
                "errors": [],
            },
        )

        status = cli_support_mod.collect_ml_runtime_status()

        assert status["deberta"]["ready"] is False
        assert "invalid artifact" in status["deberta"]["error"]
        assert status["eval"]["ready"] is False
        assert any("invalid artifact" in entry for entry in status["eval"]["blockers"])

    def test_enforce_hermetic_ml_readiness_raises_on_manifest_drift(self, monkeypatch):
        monkeypatch.setenv(cli_support_mod.HERMETIC_ML_READY_ENV, "1")
        monkeypatch.setattr(
            cli_support_mod,
            "collect_ml_runtime_status",
            lambda: {
                "eval": {
                    "ready": False,
                    "blockers": ["checksum mismatch for runtime artifact"],
                    "warnings": [],
                }
            },
        )

        with pytest.raises(RuntimeError, match="checksum mismatch for runtime artifact"):
            cli_support_mod.enforce_hermetic_ml_readiness()

    def test_preflight_json_exits_nonzero_when_eval_not_ready(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setattr(
            cli_main,
            "_collect_ml_runtime_status",
            lambda: {
                "eval": {
                    "ready": False,
                    "blockers": ["missing semantic index"],
                    "warnings": ["fallback backend active"],
                }
            },
        )
        monkeypatch.setattr(cli_main, "_hermetic_ml_ready_required", lambda: True)

        result = self.runner.invoke(cli_main.main, ["preflight", "--json"])

        assert result.exit_code == cli_main.EXIT_ERROR
        payload = json.loads(stdout.getvalue())
        assert payload["ready"] is False
        assert payload["hermetic_gate_enabled"] is True
        assert payload["ml_runtime"]["eval"]["blockers"] == ["missing semantic index"]

    def test_preflight_json_succeeds_when_eval_ready(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setattr(
            cli_main,
            "_collect_ml_runtime_status",
            lambda: {
                "eval": {
                    "ready": True,
                    "blockers": [],
                    "warnings": [],
                }
            },
        )
        monkeypatch.setattr(cli_main, "_hermetic_ml_ready_required", lambda: False)

        result = self.runner.invoke(cli_main.main, ["preflight", "--json"])

        assert result.exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["ready"] is True
        assert payload["hermetic_gate_enabled"] is False

    def test_audit_show_renders_query_entries(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        entry = SimpleNamespace(
            timestamp=SimpleNamespace(isoformat=lambda: "2026-03-24T00:00:00+00:00"),
            event_type=SimpleNamespace(value="tool_call"),
            tool_name="terminal",
            decision="allow",
            details="ok",
        )

        class FakeTrail:
            def query(self, limit=20):
                return [entry]

        monkeypatch.setattr(cli_main, "_open_audit_trail", lambda: FakeTrail())

        result = self.runner.invoke(cli_main.main, ["audit", "show", "--limit", "1"])

        assert result.exit_code == 0
        assert "terminal" in stdout.getvalue()

    def test_run_uses_checkout_bootstrap_environment(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        checkout = tmp_dir / "fixture-hermes"
        checkout.mkdir()

        captured: dict[str, object] = {}

        monkeypatch.setattr(
            bootstrap_mod,
            "load_checkout_state",
            lambda checkout_root=None: SimpleNamespace(
                checkout_root=Path(checkout_root) if checkout_root else checkout,
                policy_source="preset max",
            ),
        )
        monkeypatch.setattr(
            bootstrap_mod,
            "compose_runtime_env",
            lambda env, checkout_root=None, start_proxy=False: {
                **env,
                "KATANA_ACTIVE": "1",
                "KATANA_CHECKOUT_ROOT": str(checkout),
                "KATANA_POLICY_SOURCE": "preset max",
                "KATANA_PROXY_URL": "http://127.0.0.1:9000",
            },
        )
        monkeypatch.setattr(cli_main.shutil, "which", lambda cmd: "hermes.exe")

        def fake_run(args, env=None):
            captured["args"] = args
            captured["env"] = env
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)

        result = self.runner.invoke(
            cli_main.main,
            ["run", "--target", str(checkout), "--", "--task", "hello"],
        )

        assert result.exit_code == 0
        assert captured["args"] == ["hermes.exe", "--task", "hello"]
        env = captured["env"]
        assert env["KATANA_CHECKOUT_ROOT"] == str(checkout)
        assert env["KATANA_POLICY_SOURCE"] == "preset max"
        assert env["KATANA_PROXY_URL"] == "http://127.0.0.1:9000"

    def test_restore_uses_installer_manifest(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        manifest = tmp_dir / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")

        class FakeInstaller:
            def restore(self, manifest_path, dry_run=False):
                return ["Restore hermes/tools/dispatch.py", "Remove .katana"]

        monkeypatch.setattr(installer_mod, "KatanaInstaller", FakeInstaller)

        result = self.runner.invoke(
            cli_main.main,
            ["restore", "--manifest", str(manifest), "--dry-run"],
        )

        assert result.exit_code == 0
        output = stdout.getvalue()
        assert "Restore hermes/tools/dispatch.py" in output
        assert "Dry run complete." in output

    def test_artifacts_status_lists_all_registered_models(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_dir / "artifacts"))

        result = self.runner.invoke(cli_main.main, ["artifacts", "status", "--all"])

        assert result.exit_code == 0
        output = stdout.getvalue()
        assert "katana_v15_distill_minilm_onnx" in output
        assert "katana_v15_large" in output

    def test_artifacts_setup_noninteractive_requires_explicit_choice(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_dir / "artifacts"))

        result = self.runner.invoke(cli_main.main, ["artifacts", "setup"])

        assert result.exit_code != 0
        assert "Non-interactive setup requires" in result.output

    def test_artifacts_setup_yes_downloads_small_only(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_dir / "artifacts"))
        calls = []

        def fake_download(spec, target_dir=None, *, force=False):
            path = Path(target_dir or tmp_dir / spec.name)
            calls.append((spec.name, path, force))
            return artifacts_mod.ArtifactStatus(spec=spec, path=path, present=True, missing_files=(), source="test")

        monkeypatch.setattr(artifacts_mod, "download_artifact", fake_download)

        result = self.runner.invoke(cli_main.main, ["artifacts", "setup", "--yes"])

        assert result.exit_code == 0
        assert [call[0] for call in calls] == ["katana_v15_distill_minilm_onnx"]

    def test_artifacts_setup_all_downloads_small_and_large(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_dir / "artifacts"))
        calls = []

        def fake_download(spec, target_dir=None, *, force=False):
            path = Path(target_dir or tmp_dir / spec.name)
            calls.append((spec.name, path, force))
            return artifacts_mod.ArtifactStatus(spec=spec, path=path, present=True, missing_files=(), source="test")

        monkeypatch.setattr(artifacts_mod, "download_artifact", fake_download)

        result = self.runner.invoke(
            cli_main.main,
            ["artifacts", "setup", "--all", "--target-dir", str(tmp_dir / "cache")],
        )

        assert result.exit_code == 0
        assert [call[0] for call in calls] == ["katana_v15_distill_minilm_onnx", "katana_v15_large"]
        assert calls[0][1] == tmp_dir / "cache" / "katana_v15_distill_minilm_onnx"
        assert calls[1][1] == tmp_dir / "cache" / "katana_v15_large"

    def test_setup_yes_downloads_small_only_and_skips_proving_ground(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_dir / "artifacts"))
        artifact_calls = []
        install_calls = []

        def fake_download(spec, target_dir=None, *, force=False):
            path = Path(target_dir or tmp_dir / spec.name)
            artifact_calls.append((spec.name, path, force))
            return artifacts_mod.ArtifactStatus(spec=spec, path=path, present=True, missing_files=(), source="test")

        def fake_install():
            install_calls.append("proving-ground")

        monkeypatch.setattr(artifacts_mod, "download_artifact", fake_download)
        monkeypatch.setattr(cli_main, "_install_proving_ground_extra", fake_install)

        result = self.runner.invoke(cli_main.main, ["setup", "--yes"])

        assert result.exit_code == 0
        assert [call[0] for call in artifact_calls] == ["katana_v15_distill_minilm_onnx"]
        assert install_calls == []

    def test_setup_proving_ground_installs_extra_without_artifact_choice(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_dir / "artifacts"))
        monkeypatch.setattr(cli_main, "_missing_proving_ground_dependencies", lambda: ["openai"])
        calls = []

        def fake_run(cmd, *, check):
            calls.append((cmd, check))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)

        result = self.runner.invoke(cli_main.main, ["setup", "--proving-ground"])

        assert result.exit_code == 0
        assert calls
        cmd, check = calls[0]
        assert check is True
        assert cmd[:4] == [cli_main.sys.executable, "-m", "pip", "install"]
        assert cmd[-2:] == ["-e", ".[proving-ground]"]

    def test_setup_rejects_conflicting_proving_ground_options(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_dir / "artifacts"))

        result = self.runner.invoke(cli_main.main, ["setup", "--proving-ground", "--no-proving-ground"])

        assert result.exit_code != 0
        assert "--proving-ground and --no-proving-ground cannot be used together" in result.output
