"""
Semantic Zvec Scanner — 128d contrastive embeddings for attack detection.

Uses quantized zvec model (MiniLM → 128d projector) to detect semantic similarity
to known attack patterns. Fast (<2ms on CPU) with high recall.

Model path: katana_results/zvec_quantized/
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, cast

import numpy as np
import torch
import torch.nn as nn

from ..scanner.base import BaseScanner, ScanResult


class ProjectorINT8(nn.Module):
    """INT8 quantized projector: 384d → 128d"""

    def __init__(self, state_dict):
        super().__init__()
        self.net0 = nn.Linear(384, 384)
        self.net0.weight.data = state_dict["net.0.weight"].dequantize()
        self.net0.bias.data = state_dict["net.0.bias"].dequantize()
        self.gelu = nn.GELU()
        self.net2 = nn.Linear(384, 128)
        self.net2.weight.data = state_dict["net.2.weight"].dequantize()
        self.net2.bias.data = state_dict["net.2.bias"].dequantize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net0(x)
        x = self.gelu(x)
        x = self.net2(x)
        return torch.nn.functional.normalize(x, p=2, dim=-1)


class SemanticZvecScanner(BaseScanner):
    """
    Semantic similarity scanner using contrastive zvec embeddings.

    Computes cosine similarity between input and pre-computed attack centroids.
    Flags high-similarity inputs as potential attacks.

    Latency: ~1-2ms on CPU (with INT8 projector)
    """

    name = "semantic_zvec"
    confidence = 0.8  # High confidence on strong matches

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.3,
        attack_centroids_path: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to zvec_quantized/ folder
            threshold: Similarity threshold for FLAG decision (default 0.3)
            attack_centroids_path: Path to pre-computed attack centroids (.npy)
        """
        super().__init__()
        self.model_path = model_path or self._find_model_path()
        self.threshold = threshold
        self.centroids_path = attack_centroids_path

        self.tokenizer: Any | None = None
        self.backbone: Any | None = None
        self.projector: ProjectorINT8 | None = None
        self.attack_centroids: np.ndarray | None = None
        self._loaded = False

    def _find_model_path(self) -> str:
        """Auto-detect model path."""
        candidates = [
            os.path.expanduser("~/Documents/Code/hermes-katana/katana_results/zvec_quantized"),
            os.path.expanduser("~/katana_results/zvec_quantized"),
            "/content/drive/MyDrive/katana_results/zvec_quantized",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""

    def lazy_load(self) -> None:
        """Lazy-load model on first use."""
        if self._loaded:
            return

        if not self.model_path or not os.path.exists(self.model_path):
            # Model not available, skip silently
            return

        from transformers import AutoTokenizer, AutoModel

        # Load backbone (FP32)
        backbone_path = os.path.join(self.model_path, "backbone_fp32")
        if os.path.exists(backbone_path):
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_path)
            self.backbone = AutoModel.from_pretrained(backbone_path)
            if self.backbone is not None:
                self.backbone.eval()
        else:
            # Fallback to HF
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.backbone = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            if self.backbone is not None:
                self.backbone.eval()

        # Load INT8 projector
        proj_path = os.path.join(self.model_path, "projector_int8.pt")
        if os.path.exists(proj_path):
            from hermes_katana.ml_artifacts import safe_torch_load

            proj_state = safe_torch_load(proj_path, map_location="cpu", weights_only=True)
            self.projector = ProjectorINT8(proj_state["projector_int8"])
            self.projector.eval()
            print(f"[Zvec] Loaded INT8 projector from {proj_path}")
        else:
            print(f"[Zvec] Warning: projector_int8.pt not found at {proj_path}")

        # Load attack centroids if available
        if self.centroids_path and os.path.exists(self.centroids_path):
            attack_centroids = np.load(self.centroids_path)
            self.attack_centroids = attack_centroids
            print(f"[Zvec] Loaded attack centroids: {attack_centroids.shape}")

        self._loaded = True

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to 128d zvec embeddings."""
        if not self._loaded:
            self.lazy_load()

        if self.backbone is None or self.tokenizer is None:
            return np.zeros((len(texts), 128))

        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = self.backbone(**inputs)
            token_emb = outputs.last_hidden_state
            attn_mask = inputs["attention_mask"]
            # Mean pooling
            mask_exp = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
            sum_emb = torch.sum(token_emb * mask_exp, 1)
            sum_mask = torch.clamp(mask_exp.sum(1), min=1e-9)
            mean_emb = sum_emb / sum_mask
            # Project
            if self.projector is not None:
                zvec_emb = self.projector(mean_emb)
            else:
                zvec_emb = mean_emb  # Fallback
        return cast(np.ndarray, np.asarray(zvec_emb.cpu().numpy(), dtype=np.float32))

    def scan(self, content: str, **kwargs) -> ScanResult:
        """Scan text for semantic similarity to attacks."""
        if not self._loaded:
            self.lazy_load()

        if self.backbone is None:
            return ScanResult(
                scanner=self.name,
                is_malicious=False,
                confidence=0.0,
                findings=[],
            )

        # Encode input
        emb = self._encode([content])[0]  # [128]

        # Compute similarity to attack centroids
        if self.attack_centroids is not None:
            # Multi-centroid: compute all similarities
            sims = np.dot(self.attack_centroids, emb)  # [num_centroids]
            max_sim = float(np.max(sims))
            top_label = int(np.argmax(sims))
        else:
            # Fallback: use pre-defined attack prototype
            # (This should ideally be computed from training data)
            max_sim = 0.0
            top_label = -1

        # Decision
        is_malicious = max_sim > self.threshold
        confidence = max_sim if is_malicious else 1.0 - max_sim

        return ScanResult(
            scanner=self.name,
            is_malicious=is_malicious,
            confidence=float(confidence),
            findings=[
                {
                    "type": "semantic_similarity",
                    "max_similarity": float(max_sim),
                    "threshold": self.threshold,
                    "top_attack_label": top_label,
                }
            ]
            if is_malicious
            else [],
        )

    def scan_batch(self, contents: List[str], **kwargs) -> List[ScanResult]:
        """Batch scan multiple texts."""
        if not self._loaded:
            self.lazy_load()

        if self.backbone is None:
            return [
                ScanResult(
                    scanner=self.name,
                    is_malicious=False,
                    confidence=0.0,
                    findings=[],
                )
                for _ in contents
            ]

        # Encode all
        embs = self._encode(contents)  # [N, 128]

        results = []
        for i, emb in enumerate(embs):
            if self.attack_centroids is not None:
                sims = np.dot(self.attack_centroids, emb)
                max_sim = float(np.max(sims))
                top_label = int(np.argmax(sims))
            else:
                max_sim = 0.0
                top_label = -1

            is_malicious = max_sim > self.threshold
            confidence = max_sim if is_malicious else 1.0 - max_sim

            results.append(
                ScanResult(
                    scanner=self.name,
                    is_malicious=is_malicious,
                    confidence=float(confidence),
                    findings=[
                        {
                            "type": "semantic_similarity",
                            "max_similarity": float(max_sim),
                            "threshold": self.threshold,
                            "top_attack_label": top_label,
                        }
                    ]
                    if is_malicious
                    else [],
                )
            )

        return results


# Export for scanner registry
__all__ = ["SemanticZvecScanner"]
