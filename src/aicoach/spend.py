"""The spending ledger and the monthly budget (D-067).

Every paid request writes one line to `<data_root>/spend_ledger.jsonl` (outside git): the UTC
time, the provider, the model, the tokens or the seconds of sound, the cost and its basis:

- `actual`: the service reported the cost (OpenRouter puts `usage.cost` in every response);
- `estimate`: no reported cost, the cost is tokens or seconds times the price table;
- `free`: a request to the free tier of the Gemini API, the cost is 0.

Before a paid request the month's total (UTC calendar month) plus the estimate of the request is
compared with `AICOACH_BUDGET_USD`; if it would pass the ceiling the request is not sent and
`BudgetExceededError` is raised. No ceiling set means no check. Free requests are never blocked.
The wrappers `MeteredLLM` and `MeteredTranscriber` do both around any client, so the clients and
the agent loop know nothing about money.
"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, Field, ValidationError, field_validator

from aicoach.agent.llm import LLMClient
from aicoach.agent.messages import HistoryItem, ModelTurn, ToolResults, ToolSpec, UserMessage
from aicoach.agent.pricing import cost_of, estimate_request_usd, estimate_video_request_usd
from aicoach.config import Provider, Settings
from aicoach.errors import BudgetExceededError, StorageError
from aicoach.i18n import ru
from aicoach.speech.pricing import estimate_transcription_usd
from aicoach.speech.transcriber import RawTranscript, Transcriber
from aicoach.vlm import VideoModel, VideoReply

logger = structlog.get_logger()

LEDGER_FILE = "spend_ledger.jsonl"
Basis = Literal["actual", "estimate", "free"]
Kind = Literal["agent", "transcribe", "vision"]
Clock = Callable[[], datetime]


class SpendEntry(BaseModel):
    """One paid (or free-tier) request. `cost_usd` is `None` when nothing is known about the
    price (an `estimate` for a model that has none)."""

    ts: str
    provider: str
    model: str
    kind: Kind
    tokens_in: int | None = None
    tokens_out: int | None = None
    audio_seconds: float | None = None
    # Finite and not negative: a NaN would turn the month's sum into NaN and the budget off.
    cost_usd: float | None = Field(ge=0, allow_inf_nan=False)
    basis: Basis

    @field_validator("ts")
    @classmethod
    def _ts_is_an_aware_time(cls, value: str) -> str:
        """The time must parse and carry a zone (a naive one would be read as local time)."""
        if datetime.fromisoformat(value).tzinfo is None:
            msg = "the time has no time zone"
            raise ValueError(msg)
        return value


def ledger_path(settings: Settings, match_dir: Path) -> Path:
    """The ledger file: in `AICOACH_DATA_ROOT`, else in the directory above the match directory
    (a match lives in `<data_root>/<match>/`, D-058)."""
    root = settings.aicoach_data_root or match_dir.resolve().parent
    return root / LEDGER_FILE


class SpendLedger:
    """The ledger file and the budget.

    Args:
        path: The JSON-lines file; created on the first write.
        budget_usd: The monthly ceiling; `None` means no ceiling.
        clock: Current time, timezone-aware (a fake in tests).
    """

    def __init__(
        self,
        path: Path,
        budget_usd: float | None,
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        self._path = path
        self._budget = budget_usd
        self._clock = clock

    def _entries(self) -> list[SpendEntry]:
        if not self._path.is_file():
            return []
        entries: list[SpendEntry] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as err:
            msg = f"cannot read the spending ledger {self._path.name}"
            raise StorageError(msg) from err
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                entries.append(SpendEntry.model_validate_json(line))
            except ValidationError as err:
                # A silently skipped line would lower the total and let the budget be passed.
                msg = f"spending ledger {self._path.name}, line {number}: not a valid entry"
                raise StorageError(msg) from err
        return entries

    def month_total(self) -> float:
        """The cost recorded in the current UTC calendar month (entries without a cost count 0)."""
        now = self._clock().astimezone(UTC)
        total = 0.0
        for entry in self._entries():
            when = datetime.fromisoformat(entry.ts).astimezone(UTC)
            if (when.year, when.month) == (now.year, now.month):
                total += entry.cost_usd or 0.0
        return total

    def check(self, estimate_usd: float | None) -> None:
        """Refuse a request that would pass the budget.

        Args:
            estimate_usd: The estimated cost of the request; `None` (no price known) counts 0.

        Raises:
            BudgetExceededError: The month's total plus the estimate is above the ceiling (a
                total exactly at the ceiling is allowed).
            StorageError: The ledger cannot be read.
        """
        if self._budget is None:
            return
        estimate = estimate_usd or 0.0
        spent = self.month_total()
        if spent + estimate > self._budget:
            logger.warning("budget_exceeded", stage="spend", spent_usd=spent, estimate_usd=estimate)
            raise BudgetExceededError(
                ru.BUDGET_EXCEEDED.format(spent=spent, estimate=estimate, budget=self._budget)
            )

    def record(
        self,
        provider: str,
        model: str,
        kind: Kind,
        cost_usd: float | None,
        basis: Basis,
        *,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        audio_seconds: float | None = None,
    ) -> SpendEntry:
        """Append one line.

        Raises:
            StorageError: The file cannot be written.
        """
        entry = SpendEntry(
            ts=self._clock().astimezone(UTC).isoformat(timespec="seconds"),
            provider=provider,
            model=model,
            kind=kind,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            audio_seconds=audio_seconds,
            cost_usd=cost_usd,
            basis=basis,
        )
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")
        except OSError as err:
            msg = f"cannot write the spending ledger {self._path.name}"
            raise StorageError(msg) from err
        logger.info(
            "spend_recorded",
            stage="spend",
            provider=provider,
            model=model,
            kind=kind,
            basis=basis,
            cost_usd=cost_usd,
        )
        return entry


def _prompt_chars(system: str, history: Sequence[HistoryItem], tools: Sequence[ToolSpec]) -> int:
    """The size of what a chat request carries, in characters (for the estimate only)."""
    total = len(system) + sum(len(t.description) + len(json.dumps(t.parameters)) for t in tools)
    for item in history:
        if isinstance(item, UserMessage):
            total += len(item.text)
        elif isinstance(item, ModelTurn):
            total += len(item.text) + sum(len(json.dumps(c.args)) for c in item.calls)
        elif isinstance(item, ToolResults):
            total += sum(len(json.dumps(r.result, ensure_ascii=False)) for r in item.results)
    return total


class MeteredLLM:
    """An `LLMClient` that checks the budget before each request and records it after.

    Args:
        inner: The real client.
        ledger: The ledger and the budget.
        provider: `openrouter` (paid: checked, recorded with the reported cost) or `gemini` (the
            free tier: recorded with cost 0 and basis `free`, never blocked).
    """

    def __init__(self, inner: LLMClient, ledger: SpendLedger, provider: Provider) -> None:
        self._inner = inner
        self._ledger = ledger
        self._provider = provider

    @property
    def model(self) -> str:
        """The model name of the inner client."""
        return self._inner.model

    def generate(
        self,
        system: str,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        *,
        allow_tools: bool = True,
    ) -> ModelTurn:
        """One request, see `LLMClient.generate`.

        Raises:
            BudgetExceededError: The request would pass the budget (it is not sent).
            ExternalServiceError: The service failed for good (nothing is recorded).
            StorageError: The ledger cannot be read or written.
        """
        paid = self._provider != "gemini"
        if paid:
            self._ledger.check(
                estimate_request_usd(self.model, _prompt_chars(system, history, tools))
            )
        turn = self._inner.generate(system, history, tools, allow_tools=allow_tools)
        usage = turn.usage
        cost: float | None
        basis: Basis
        if not paid:
            cost, basis = 0.0, "free"
        elif usage.cost_usd is not None:
            cost, basis = usage.cost_usd, "actual"
        else:
            cost, basis = cost_of(self.model, usage), "estimate"
        self._ledger.record(
            self._provider,
            self.model,
            "agent",
            cost,
            basis,
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
        )
        return turn


class MeteredTranscriber:
    """A `Transcriber` with the same budget check and record, per second of sound."""

    def __init__(self, inner: Transcriber, ledger: SpendLedger, provider: Provider) -> None:
        self._inner = inner
        self._ledger = ledger
        self._provider = provider

    @property
    def model(self) -> str:
        """The model name of the inner client."""
        return self._inner.model

    def transcribe(
        self,
        audio: bytes,
        mime_type: str,
        *,
        vocabulary: Sequence[str] | None = None,
        duration_s: float | None = None,
    ) -> RawTranscript:
        """One request, see `Transcriber.transcribe`.

        Raises:
            BudgetExceededError: The request would pass the budget (it is not sent).
            ExternalServiceError: The service failed for good (nothing is recorded).
            TranscriptionError: The answer cannot be read.
            StorageError: The ledger cannot be read or written.
        """
        paid = self._provider != "gemini"
        if paid:
            self._ledger.check(estimate_transcription_usd(self.model, duration_s or 0.0))
        raw = self._inner.transcribe(audio, mime_type, vocabulary=vocabulary, duration_s=duration_s)
        seconds = raw.audio_seconds if raw.audio_seconds is not None else duration_s
        cost: float | None
        basis: Basis
        if not paid:
            cost, basis = 0.0, "free"
        elif raw.cost_usd is not None:
            cost, basis = raw.cost_usd, "actual"
        else:
            cost = estimate_transcription_usd(self.model, seconds or 0.0)
            basis = "estimate"
        self._ledger.record(
            self._provider,
            self.model,
            "transcribe",
            cost,
            basis,
            tokens_in=raw.input_tokens,
            tokens_out=raw.output_tokens,
            audio_seconds=seconds,
        )
        return raw


class MeteredVideoModel:
    """A `VideoModel` with the same budget check and record: the estimate is the prompt plus the
    length of the clip at the table price; the record is what the service charged."""

    def __init__(self, inner: VideoModel, ledger: SpendLedger, provider: Provider) -> None:
        self._inner = inner
        self._ledger = ledger
        self._provider = provider

    @property
    def model(self) -> str:
        """The model name of the inner client."""
        return self._inner.model

    def describe(
        self, system: str, prompt: str, video: bytes, *, video_seconds: float | None = None
    ) -> VideoReply:
        """One request, see `VideoModel.describe`.

        Raises:
            BudgetExceededError: The request would pass the budget (it is not sent).
            ExternalServiceError: The service failed for good (nothing is recorded).
            StorageError: The ledger cannot be read or written.
        """
        paid = self._provider != "gemini"
        if paid:
            self._ledger.check(
                estimate_video_request_usd(
                    self.model, len(system) + len(prompt), video_seconds or 0.0
                )
            )
        reply = self._inner.describe(system, prompt, video, video_seconds=video_seconds)
        usage = reply.usage
        cost: float | None
        basis: Basis
        if not paid:
            cost, basis = 0.0, "free"
        elif usage.cost_usd is not None:
            cost, basis = usage.cost_usd, "actual"
        else:
            cost, basis = cost_of(self.model, usage), "estimate"
        self._ledger.record(
            self._provider,
            self.model,
            "vision",
            cost,
            basis,
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
        )
        return reply
