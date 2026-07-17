"""
Task graph state.

The Senior writes this. Everything else reads/updates it. It's the single
source of truth for who owns what, what's approved, and what failed —
so you can always open task_graph.json and see exactly what happened,
instead of trusting anyone's prose summary.
"""
from __future__ import annotations
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone


@dataclass
class Subtask:
    id: str
    owner: str                     # worker slug from the Senior's plan, e.g. "worker_1" ... "worker_n"
    description: str
    file_scope: list[str]          # exact files/paths this worker is allowed to touch
    branch: str = ""
    status: str = "pending"        # pending -> in_progress -> reviewing -> approved/rejected -> merged
    attempts: int = 0
    last_rejection_reason: str = ""
    history: list[dict] = field(default_factory=list)

    def log(self, event: str, detail: str = ""):
        self.history.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "detail": detail,
        })


@dataclass
class TaskGraph:
    goal: str
    subtasks: list[Subtask] = field(default_factory=list)
    merge_order: list[str] = field(default_factory=list)  # subtask ids, in the order Senior decides to merge

    def get(self, subtask_id: str) -> Subtask:
        for s in self.subtasks:
            if s.id == subtask_id:
                return s
        raise KeyError(subtask_id)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "TaskGraph":
        d = json.loads(s)
        subtasks = [Subtask(**st) for st in d["subtasks"]]
        return TaskGraph(goal=d["goal"], subtasks=subtasks, merge_order=d.get("merge_order", []))

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "TaskGraph":
        with open(path) as f:
            return TaskGraph.from_json(f.read())