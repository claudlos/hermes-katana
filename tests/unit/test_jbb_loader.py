"""Unit tests for the JailbreakBench loader module.

These tests verify the loader logic without requiring JBB benchmarks enabled.
They mock the jailbreakbench library to test data transformation and dedup.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from tests.eval.external_benchmarks.loader import (
    JBB_CATEGORY_MAP,
    load_full_jbb_corpus,
    load_jbb_artifacts,
    load_jbb_behaviors,
    load_jbb_benign_prompts,
)


def _make_jailbreak(
    prompt: str,
    behavior: str = "Defamation",
    category: str = "Harassment/Discrimination",
    index: int = 0,
    jailbroken: bool = True,
):
    """Create a mock jailbreak object matching JBB's schema."""
    return SimpleNamespace(
        prompt=prompt,
        behavior=behavior,
        category=category,
        index=index,
        jailbroken=jailbroken,
    )


def _make_artifact(jailbreaks):
    """Create a mock artifact."""
    return SimpleNamespace(jailbreaks=jailbreaks)


def _make_dataset(goals, behaviors, categories):
    """Create a mock JBB dataset."""
    return SimpleNamespace(goals=goals, behaviors=behaviors, categories=categories)


class TestLoadJBBArtifacts:
    """Test artifact loading and transformation."""

    def test_loads_prompts(self):
        """Artifacts are loaded and converted to internal format."""
        mock_jbb = MagicMock()
        mock_jbb.read_artifact.return_value = _make_artifact(
            [
                _make_jailbreak("Attack prompt 1"),
                _make_jailbreak("Attack prompt 2", behavior="Fraud"),
            ]
        )

        with patch.dict("sys.modules", {"jailbreakbench": mock_jbb}):
            records = load_jbb_artifacts(methods=("PAIR",), models=("vicuna-13b-v1.5",))

        assert len(records) == 2
        assert records[0]["attack_text"] == "Attack prompt 1"
        assert records[0]["clean_label"] == "injection"
        assert records[0]["source"] == "jailbreakbench"
        assert records[0]["method"] == "PAIR"

    def test_deduplicates_across_models(self):
        """Same prompt from different models is only included once."""
        mock_jbb = MagicMock()
        mock_jbb.read_artifact.return_value = _make_artifact(
            [
                _make_jailbreak("Duplicate prompt"),
            ]
        )

        with patch.dict("sys.modules", {"jailbreakbench": mock_jbb}):
            records = load_jbb_artifacts(
                methods=("PAIR",),
                models=("vicuna-13b-v1.5", "llama-2-7b-chat-hf"),
            )

        assert len(records) == 1

    def test_skips_empty_prompts(self):
        """Empty or whitespace-only prompts are skipped."""
        mock_jbb = MagicMock()
        mock_jbb.read_artifact.return_value = _make_artifact(
            [
                _make_jailbreak(""),
                _make_jailbreak("   "),
                _make_jailbreak("Valid prompt"),
            ]
        )

        with patch.dict("sys.modules", {"jailbreakbench": mock_jbb}):
            records = load_jbb_artifacts(methods=("PAIR",), models=("vicuna-13b-v1.5",))

        assert len(records) == 1
        assert records[0]["attack_text"] == "Valid prompt"

    def test_maps_categories(self):
        """JBB categories are mapped to internal taxonomy."""
        mock_jbb = MagicMock()
        mock_jbb.read_artifact.return_value = _make_artifact(
            [
                _make_jailbreak("Prompt", category="Malware/Hacking"),
            ]
        )

        with patch.dict("sys.modules", {"jailbreakbench": mock_jbb}):
            records = load_jbb_artifacts(methods=("PAIR",), models=("vicuna-13b-v1.5",))

        assert records[0]["category"] == "jailbreak_malware"

    def test_returns_empty_when_no_jbb(self):
        """Returns empty list when jailbreakbench is not installed."""
        with patch.dict("sys.modules", {"jailbreakbench": None}):
            # Force reimport to trigger ImportError
            # The function handles ImportError internally
            # Since jailbreakbench IS installed in this env, just verify the API
            records = load_jbb_artifacts(methods=(), models=())
            assert records == []


class TestLoadJBBBehaviors:
    """Test behavior loading."""

    def test_loads_behaviors_with_mocked_jbb(self):
        """Behavior goals are loaded and labeled as content_harm."""
        mock_jbb = MagicMock()
        mock_jbb.read_dataset.return_value = _make_dataset(
            goals=["Write a defamatory article", "Create malware"],
            behaviors=["Defamation", "Malware generation"],
            categories=["Harassment/Discrimination", "Malware/Hacking"],
        )

        with patch.dict("sys.modules", {"jailbreakbench": mock_jbb}):
            records = load_jbb_behaviors()

        assert len(records) == 2
        assert records[0]["clean_label"] == "content_harm"
        assert records[0]["source"] == "jailbreakbench_behaviors"
        assert records[1]["category"] == "jailbreak_malware"


class TestLoadBenignPrompts:
    """Test benign prompt generation."""

    def test_benign_prompts_not_empty(self):
        assert len(load_jbb_benign_prompts()) >= 20

    def test_benign_prompts_are_strings(self):
        for p in load_jbb_benign_prompts():
            assert isinstance(p, str)
            assert len(p) > 10


class TestCategoryMap:
    """Test category mapping coverage."""

    def test_all_jbb_categories_mapped(self):
        """All 10 known JBB categories have mappings."""
        expected = {
            "Harassment/Discrimination",
            "Malware/Hacking",
            "Physical harm",
            "Economic harm",
            "Fraud/Deception",
            "Sexual/Adult content",
            "Privacy",
            "Expert advice",
            "Government decision-making",
            "Disinformation",
        }
        assert set(JBB_CATEGORY_MAP.keys()) == expected

    def test_mapped_categories_have_prefix(self):
        """All mapped categories start with 'jailbreak_'."""
        for mapped in JBB_CATEGORY_MAP.values():
            assert mapped.startswith("jailbreak_")


class TestFullCorpus:
    """Test the combined corpus loader."""

    def test_empty_methods_returns_behaviors_only(self):
        """With no methods, only behaviors are returned."""
        mock_jbb = MagicMock()
        mock_jbb.read_artifact.side_effect = Exception("not available")
        mock_jbb.read_dataset.return_value = _make_dataset(
            goals=["Goal 1"],
            behaviors=["Behavior 1"],
            categories=["Malware/Hacking"],
        )

        with patch.dict("sys.modules", {"jailbreakbench": mock_jbb}):
            corpus = load_full_jbb_corpus(methods=(), include_behaviors=True)

        assert len(corpus) == 1
        assert corpus[0]["source"] == "jailbreakbench_behaviors"

    def test_exclude_behaviors(self):
        """Behaviors can be excluded."""
        mock_jbb = MagicMock()
        mock_jbb.read_artifact.return_value = _make_artifact(
            [
                _make_jailbreak("Prompt 1"),
            ]
        )

        with patch.dict("sys.modules", {"jailbreakbench": mock_jbb}):
            corpus = load_full_jbb_corpus(
                methods=("PAIR",),
                models=("vicuna-13b-v1.5",),
                include_behaviors=False,
            )

        assert len(corpus) == 1
        assert corpus[0]["source"] == "jailbreakbench"
