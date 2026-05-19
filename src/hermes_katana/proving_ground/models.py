"""Core data models for the proving ground."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json
import time


class InjectionStrategy(str, Enum):
    DIRECT = "direct"
    CONTEXT_INJECTION = "context_injection"
    TASK_WRAP = "task_wrap"
    MULTIMODAL_IMAGE = "multimodal_image"
    MULTIMODAL_PDF = "multimodal_pdf"
    MULTI_TURN = "multi_turn"


class SandboxTask(str, Enum):
    CODE_REVIEW = "code_review"
    SUMMARIZE_DOCUMENT = "summarize_document"
    ANALYZE_DATA = "analyze_data"
    DRAFT_EMAIL = "draft_email"
    ANSWER_QUESTION = "answer_question"
    PROCESS_REQUEST = "process_request"


class AttackOutcome(str, Enum):
    BLOCKED = "blocked"  # LLM refused / safety triggered
    PARTIAL = "partial"  # LLM acknowledged but didn't fully comply
    COMPLIANCE = "compliance"  # LLM did what was asked
    ESCALATED = "escalated"  # LLM went beyond what was requested
    ERROR = "error"  # Provider error, timeout, etc.


@dataclass
class AttackSample:
    """A single attack from the corpus."""

    id: str
    text: str
    label: str
    source_lang: str  # "en" for original, language code for translations
    origin: str = "user_input"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "AttackSample":
        data = json.loads(line)
        return cls(
            id=data.get("id", data.get("source_id", "")),
            text=data.get("text", data.get("prompt", "")),
            label=data.get("label", "unknown"),
            source_lang=data.get("lang", data.get("language", "en")),
            origin=data.get("origin", "user_input"),
            metadata={
                k: v
                for k, v in data.items()
                if k
                not in (
                    "id",
                    "source_id",
                    "text",
                    "prompt",
                    "label",
                    "lang",
                    "language",
                    "origin",
                )
            },
        )


@dataclass
class EvaluationRequest:
    """A single evaluation to run."""

    attack: AttackSample
    strategy: InjectionStrategy
    task: SandboxTask
    provider: str
    model: str
    messages: list  # the chat messages to send


@dataclass
class EvaluationResult:
    """Result of a single attack evaluation."""

    attack_id: str
    strategy: str
    task: str
    provider: str
    model: str
    attack_text: str
    attack_label: str
    source_lang: str
    messages_sent: list
    response_text: str
    outcome: AttackOutcome
    score: float
    judge_reasoning: str
    latency_ms: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d

    @property
    def is_effective(self) -> bool:
        return self.score >= 1.0 and self.outcome not in (
            AttackOutcome.BLOCKED,
            AttackOutcome.ERROR,
        )
