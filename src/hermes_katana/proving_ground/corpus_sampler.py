"""Stratified corpus sampler — draws balanced samples from the attack corpus."""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .models import AttackSample
from .paths import default_corpus_path


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class CorpusSampler:
    """Sample attacks from the HermesKatana corpus with stratification."""

    def __init__(self, hermes_katana_root: str | Path | None = None, corpus_paths: list[str | Path] | None = None):
        """Create a sampler.

        The public repo intentionally does not bundle private training corpora.
        Pass ``corpus_paths`` or set ``KATANA_ATTACK_CORPUS`` to one or more
        JSONL files separated by ``os.pathsep``. If neither is provided, the
        tiny synthetic example corpus under ``examples/proving_ground`` is used
        for smoke tests.
        """
        self.root = Path.cwd() if hermes_katana_root is None else Path(hermes_katana_root)

        explicit_sources: list[Path] = []
        env_corpus = os.environ.get("KATANA_ATTACK_CORPUS")
        if env_corpus:
            explicit_sources.extend(Path(p).expanduser() for p in env_corpus.split(os.pathsep) if p.strip())
        if corpus_paths:
            explicit_sources.extend(Path(p).expanduser() for p in corpus_paths)

        # Source files in priority order
        self.sources = {
            **{f"explicit_{idx}": path for idx, path in enumerate(explicit_sources)},
            "sample": default_corpus_path(),
            "en_v4": self.root / "training/data_v3/translation_source_attacks_en_v4.jsonl",
            "backup_recovery": self.root / "training/data_v3/backup_recovery_normalized.jsonl",
            "combined_translated": self.root / "training/data_v3/factory/accepted/combined.jsonl",
        }

        # Label distribution for stratified sampling
        self._label_cache: Optional[dict[str, list[dict]]] = None

    def _load_by_label(self) -> dict[str, list[dict]]:
        """Load all attacks grouped by label."""
        if self._label_cache is not None:
            return self._label_cache

        by_label: dict[str, list[dict]] = defaultdict(list)

        seen_ids: set[str] = set()
        for source_name, source_path in self.sources.items():
            if source_name == "combined_translated" or not source_path.exists():
                continue
            for row in load_jsonl(str(source_path)):
                rid = str(row.get("id", row.get("source_id", row.get("_id", ""))))
                if rid and rid in seen_ids:
                    continue
                if rid:
                    seen_ids.add(rid)
                label = row.get("label", row.get("attack_type", "unknown"))
                row["_source"] = source_name
                by_label[label].append(row)

        self._label_cache = dict(by_label)
        return self._label_cache

    def sample(
        self,
        n: int = 500,
        labels: Optional[list[str]] = None,
        languages: Optional[list[str]] = None,
        seed: int = 42,
    ) -> list[AttackSample]:
        """Draw a stratified sample proportional to label distribution.

        If languages is specified, includes translations from the combined corpus.
        """
        rng = random.Random(seed)
        by_label = self._load_by_label()

        # Filter to requested labels
        if labels:
            by_label = {k: v for k, v in by_label.items() if k in labels}

        # Calculate total and per-label targets
        total_rows = sum(len(v) for v in by_label.values())
        if total_rows == 0:
            return []

        samples = []
        for label, rows in sorted(by_label.items()):
            proportion = len(rows) / total_rows
            target = max(1, round(n * proportion))
            selected = rng.sample(rows, min(target, len(rows)))

            for row in selected:
                text = row.get("text", row.get("prompt", ""))
                if not text:
                    continue

                samples.append(
                    AttackSample(
                        id=str(row.get("id", row.get("source_id", row.get("_id", "")))),
                        text=text,
                        label=label,
                        source_lang=row.get("lang", row.get("language", "en")),
                        origin=row.get("origin", "user_input"),
                        metadata={
                            k: v
                            for k, v in row.items()
                            if k
                            not in (
                                "id",
                                "source_id",
                                "_id",
                                "text",
                                "prompt",
                                "label",
                                "lang",
                                "language",
                                "origin",
                                "_source",
                            )
                        },
                    )
                )

        # If translations requested, add them from combined corpus
        if languages:
            combined_path = self.sources["combined_translated"]
            if combined_path.exists():
                lang_rows = []
                for row in load_jsonl(str(combined_path)):
                    lang = row.get("lang", row.get("language", ""))
                    if lang in languages:
                        lang_rows.append(row)

                # Sample proportional from translations
                lang_target = min(len(lang_rows), n)
                lang_selected = rng.sample(lang_rows, lang_target)
                for row in lang_selected:
                    text = row.get("text", row.get("prompt", ""))
                    if not text:
                        continue
                    samples.append(
                        AttackSample(
                            id=str(row.get("id", row.get("source_id", ""))),
                            text=text,
                            label=row.get("label", "unknown"),
                            source_lang=row.get("lang", row.get("language", "en")),
                            origin=row.get("origin", "user_input"),
                        )
                    )

        rng.shuffle(samples)
        return samples[:n]

    def get_distribution(self) -> dict[str, int]:
        """Return label distribution across all sources."""
        by_label = self._load_by_label()
        return {k: len(v) for k, v in sorted(by_label.items())}
