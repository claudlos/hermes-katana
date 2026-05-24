"""Hypothesis DAG — the research agent's long-term memory.

Chat history is ephemeral. The hypothesis DAG is not. When context
compaction happens, chat evaporates but the DAG survives. When the agent
restarts, the DAG is reloaded from disk. When the verifier audits a Result,
it walks up the DAG to see what evidence supports it.

Node shape:
  id           — same as registry hypothesis id (H-<date>-<slug>) OR exploratory "X-..."
  parent_id    — for refinement chains (H2 refines H1)
  title
  status       — proposed | preregistered | testing | resolved | abandoned
  claims[]     — list of Result event_ids attached to this node
  children[]   — list of child hypothesis ids
  elo          — running Elo score (for Phase 3 Simula-style calibration)
  created_at
  updated_at

Operations:
  add_node(hypothesis)            — adds a node, optionally linking parent
  attach_claim(node_id, event_id) — record that a Result relates to this node
  best_by_elo(top_k)              — ordering for BFTS search
  prune(below_elo)                — drop low-Elo subtrees to manage size
  to_dict() / from_dict()         — persistence
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


VALID_STATUS = {"proposed", "preregistered", "testing", "resolved", "abandoned"}


@dataclass
class Node:
    id: str
    title: str
    status: str = "proposed"
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)  # event_ids of Results
    elo: float = 1200.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class HypothesisDAG:
    def __init__(self, path: Path | None = None):
        self._nodes: dict[str, Node] = {}
        self.path = path
        if path and path.exists():
            self._load()

    # ------------------------------------------------------------------- I/O
    def _load(self) -> None:
        assert self.path is not None
        d = json.loads(self.path.read_text(encoding="utf-8"))
        for nid, raw in d.get("nodes", {}).items():
            self._nodes[nid] = Node(**raw)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "saved_at": time.time(),
            "nodes": {nid: asdict(n) for nid, n in self._nodes.items()},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # --------------------------------------------------------------- mutation
    def add_node(self, node: Node) -> Node:
        if node.id in self._nodes:
            raise ValueError(f"node {node.id} already in DAG")
        if node.status not in VALID_STATUS:
            raise ValueError(f"invalid status {node.status}")
        if node.parent_id:
            parent = self._nodes.get(node.parent_id)
            if not parent:
                raise ValueError(f"parent {node.parent_id} not in DAG")
            parent.children.append(node.id)
            parent.updated_at = time.time()
        self._nodes[node.id] = node
        return node

    def attach_claim(self, node_id: str, event_id: str) -> None:
        n = self._nodes.get(node_id)
        if not n:
            raise KeyError(node_id)
        n.claims.append(event_id)
        n.updated_at = time.time()

    def set_status(self, node_id: str, status: str) -> None:
        if status not in VALID_STATUS:
            raise ValueError(f"invalid status {status}")
        n = self._nodes.get(node_id)
        if not n:
            raise KeyError(node_id)
        n.status = status
        n.updated_at = time.time()

    def update_elo(self, node_id: str, new_elo: float) -> None:
        n = self._nodes.get(node_id)
        if not n:
            raise KeyError(node_id)
        n.elo = new_elo
        n.updated_at = time.time()

    # --------------------------------------------------------------- reading
    def get(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def roots(self) -> list[Node]:
        return [n for n in self._nodes.values() if n.parent_id is None]

    def best_by_elo(self, top_k: int = 5, status: str | None = None) -> list[Node]:
        ns = list(self._nodes.values())
        if status:
            ns = [n for n in ns if n.status == status]
        ns.sort(key=lambda n: -n.elo)
        return ns[:top_k]

    def prune(self, below_elo: float) -> int:
        """Drop nodes with elo < threshold AND status in {abandoned, rejected}.

        Returns count of pruned nodes. Won't prune a node with children that
        aren't also droppable (keeps the DAG connected).
        """
        droppable = {nid for nid, n in self._nodes.items() if n.elo < below_elo and n.status == "abandoned"}

        # Remove only if all descendants are also droppable, else keep.
        def _descendants(nid: str) -> set[str]:
            out: set[str] = set()
            stack = [nid]
            while stack:
                x = stack.pop()
                for c in self._nodes[x].children:
                    if c not in out:
                        out.add(c)
                        stack.append(c)
            return out

        safe_to_drop = {nid for nid in droppable if _descendants(nid) <= droppable}
        for nid in safe_to_drop:
            n = self._nodes.pop(nid)
            if n.parent_id and n.parent_id in self._nodes:
                self._nodes[n.parent_id].children.remove(nid)
        return len(safe_to_drop)

    def __len__(self) -> int:
        return len(self._nodes)
