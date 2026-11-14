"""The journal of the agent: one JSON line per question, in the match directory.

What is written: the question, the tools called (name and arguments), tokens in and out, the
time, the model, the outcome of the validator and the estimated cost. The answer text and the
API key are not written.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any  # JSON arguments of the model's tool calls

from aicoach.errors import StorageError

JOURNAL_FILE = "agent_log.jsonl"


@dataclass(frozen=True)
class JournalEntry:
    """One question. `validator` is `first_try`, `retry`, `refused` or `error`."""

    question: str
    tools: list[dict[str, Any]]
    tokens_in: int
    tokens_out: int
    duration_ms: int
    model: str
    validator: str
    violations: list[str] = field(default_factory=list)
    n_links: int = 0
    cost_usd: float | None = None


def append_entry(match_dir: Path, entry: JournalEntry, now: datetime | None = None) -> None:
    """Append the entry to `agent_log.jsonl` (the file is created on the first write).

    Raises:
        StorageError: The file cannot be written.
    """
    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    line = json.dumps({"ts": stamp, **asdict(entry)}, ensure_ascii=False)
    try:
        with (match_dir / JOURNAL_FILE).open("a", encoding="utf-8") as out:
            out.write(line + "\n")
    except OSError as err:
        msg = f"cannot write the agent journal in {match_dir}"
        raise StorageError(msg) from err
