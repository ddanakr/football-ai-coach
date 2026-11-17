"""The spending ledger, the monthly budget and the metered clients."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aicoach.agent.messages import HistoryItem, ModelTurn, ToolSpec, Usage, UserMessage
from aicoach.config import load_settings
from aicoach.errors import BudgetExceededError, ExternalServiceError, StorageError
from aicoach.speech.transcriber import RawTranscript
from aicoach.spend import (
    LEDGER_FILE,
    MeteredLLM,
    MeteredTranscriber,
    SpendLedger,
    ledger_path,
)

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def ledger(tmp_path: Path, budget: float | None, now: datetime = NOW) -> SpendLedger:
    return SpendLedger(tmp_path / LEDGER_FILE, budget, clock=lambda: now)


def lines(tmp_path: Path) -> list[dict[str, object]]:
    text = (tmp_path / LEDGER_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def write_line(path: Path, ts: str, cost: float | None, basis: str = "actual") -> None:
    entry = {
        "ts": ts,
        "provider": "openrouter",
        "model": "m",
        "kind": "agent",
        "cost_usd": cost,
        "basis": basis,
    }
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(entry) + "\n")


@pytest.mark.unit
def test_record_appends_one_json_line_with_all_fields_and_creates_the_directory(
    tmp_path: Path,
) -> None:
    store = SpendLedger(tmp_path / "deep" / LEDGER_FILE, None, clock=lambda: NOW)

    store.record("openrouter", "google/x", "agent", 0.0123, "actual", tokens_in=10, tokens_out=2)
    store.record("openrouter", "scribe", "transcribe", 0.006, "estimate", audio_seconds=99.0)

    text = (tmp_path / "deep" / LEDGER_FILE).read_text(encoding="utf-8")
    rows = [json.loads(x) for x in text.splitlines()]
    assert rows[0] == {
        "ts": "2026-11-17T12:00:00+00:00",
        "provider": "openrouter",
        "model": "google/x",
        "kind": "agent",
        "tokens_in": 10,
        "tokens_out": 2,
        "audio_seconds": None,
        "cost_usd": 0.0123,
        "basis": "actual",
    }
    assert rows[1]["audio_seconds"] == 99.0
    assert rows[1]["basis"] == "estimate"


@pytest.mark.unit
def test_month_total_sums_only_the_current_utc_calendar_month(tmp_path: Path) -> None:
    path = tmp_path / LEDGER_FILE
    write_line(path, "2026-10-31T23:59:59+00:00", 5.0)  # last month
    write_line(path, "2026-11-01T00:00:00+00:00", 1.5)  # first second of this month
    write_line(path, "2026-11-17T08:00:00+00:00", 2.0)
    write_line(path, "2026-12-01T00:00:00+00:00", 9.0)  # a later month
    write_line(path, "2026-11-17T09:00:00+00:00", None, "estimate")  # no price: counts 0

    assert ledger(tmp_path, None).month_total() == pytest.approx(3.5)


@pytest.mark.unit
def test_month_total_uses_utc_for_a_timestamp_with_another_offset(tmp_path: Path) -> None:
    path = tmp_path / LEDGER_FILE
    # 2026-11-01 02:00 at +05:00 is 2026-10-31 21:00 UTC: October.
    write_line(path, "2026-11-01T02:00:00+05:00", 4.0)

    assert ledger(tmp_path, None).month_total() == 0.0


@pytest.mark.unit
def test_month_total_without_a_file_is_zero(tmp_path: Path) -> None:
    assert ledger(tmp_path, 1.0).month_total() == 0.0


@pytest.mark.unit
def test_a_new_month_starts_from_zero(tmp_path: Path) -> None:
    write_line(tmp_path / LEDGER_FILE, "2026-11-30T23:00:00+00:00", 10.0)
    december = datetime(2026, 12, 1, 0, 0, tzinfo=UTC)

    ledger(tmp_path, 10.0, december).check(9.0)  # nothing is spent this month yet


@pytest.mark.unit
def test_check_allows_a_request_that_ends_exactly_at_the_budget(tmp_path: Path) -> None:
    write_line(tmp_path / LEDGER_FILE, "2026-11-02T00:00:00+00:00", 4.0)

    ledger(tmp_path, 5.0).check(1.0)  # 4 + 1 == 5: allowed


@pytest.mark.unit
def test_check_refuses_a_request_that_would_pass_the_budget(tmp_path: Path) -> None:
    write_line(tmp_path / LEDGER_FILE, "2026-11-02T00:00:00+00:00", 4.0)

    with pytest.raises(BudgetExceededError, match=r"4\.0000.*1\.0100.*5\.00"):
        ledger(tmp_path, 5.0).check(1.01)


@pytest.mark.unit
def test_check_with_no_budget_never_refuses_and_reads_nothing(tmp_path: Path) -> None:
    (tmp_path / LEDGER_FILE).write_text("garbage\n", encoding="utf-8")  # would fail if read

    ledger(tmp_path, None).check(1_000_000.0)


@pytest.mark.unit
def test_a_budget_of_zero_refuses_any_priced_request(tmp_path: Path) -> None:
    with pytest.raises(BudgetExceededError):
        ledger(tmp_path, 0.0).check(0.0001)
    ledger(tmp_path, 0.0).check(0.0)


@pytest.mark.unit
def test_an_unknown_estimate_counts_zero_but_an_exceeded_month_still_refuses(
    tmp_path: Path,
) -> None:
    write_line(tmp_path / LEDGER_FILE, "2026-11-02T00:00:00+00:00", 6.0)

    ledger(tmp_path, 10.0).check(None)
    with pytest.raises(BudgetExceededError):
        ledger(tmp_path, 5.0).check(None)


@pytest.mark.unit
def test_a_broken_ledger_line_is_a_loud_error_not_a_lower_total(tmp_path: Path) -> None:
    write_line(tmp_path / LEDGER_FILE, "2026-11-02T00:00:00+00:00", 6.0)
    with (tmp_path / LEDGER_FILE).open("a", encoding="utf-8") as out:
        out.write("{not json\n")

    with pytest.raises(StorageError, match="line 2"):
        ledger(tmp_path, 5.0).month_total()


@pytest.mark.unit
def test_blank_lines_in_the_ledger_are_skipped(tmp_path: Path) -> None:
    write_line(tmp_path / LEDGER_FILE, "2026-11-02T00:00:00+00:00", 1.0)
    with (tmp_path / LEDGER_FILE).open("a", encoding="utf-8") as out:
        out.write("\n\n")

    assert ledger(tmp_path, None).month_total() == 1.0


@pytest.mark.unit
def test_record_into_an_unwritable_place_is_a_storage_error(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("x", encoding="utf-8")  # a file where a directory is needed

    with pytest.raises(StorageError, match="cannot write"):
        SpendLedger(blocked / LEDGER_FILE, None).record("p", "m", "agent", 0.0, "free")


@pytest.mark.unit
def test_ledger_lives_in_the_data_root_or_above_the_match_directory(tmp_path: Path) -> None:
    match_dir = tmp_path / "data" / "match01"
    match_dir.mkdir(parents=True)

    assert ledger_path(load_settings(), match_dir) == tmp_path / "data" / LEDGER_FILE
    root = tmp_path / "root"
    assert ledger_path(load_settings(data_root=root), match_dir) == root / LEDGER_FILE


class _LLM:
    """A scripted client that counts the requests it really gets."""

    def __init__(self, usage: Usage, model: str = "google/gemini-3.5-flash-lite") -> None:
        self.model = model
        self.usage = usage
        self.calls = 0

    def generate(
        self,
        system: str,  # noqa: ARG002
        history: Sequence[HistoryItem],  # noqa: ARG002
        tools: Sequence[ToolSpec],  # noqa: ARG002
        *,
        allow_tools: bool = True,  # noqa: ARG002
    ) -> ModelTurn:
        self.calls += 1
        return ModelTurn("ок", (), self.usage)


def ask(llm: MeteredLLM) -> ModelTurn:
    return llm.generate("система", [UserMessage("вопрос?")], [])


@pytest.mark.unit
def test_openrouter_request_is_recorded_with_the_cost_the_service_reported(
    tmp_path: Path,
) -> None:
    inner = _LLM(Usage(1000, 200, 0.00099))

    ask(MeteredLLM(inner, ledger(tmp_path, 5.0), "openrouter"))

    (row,) = lines(tmp_path)
    assert (row["provider"], row["kind"], row["basis"]) == ("openrouter", "agent", "actual")
    assert row["cost_usd"] == 0.00099
    assert (row["tokens_in"], row["tokens_out"]) == (1000, 200)
    assert row["model"] == "google/gemini-3.5-flash-lite"


@pytest.mark.unit
def test_openrouter_request_without_a_reported_cost_is_an_estimate_by_the_price_table(
    tmp_path: Path,
) -> None:
    inner = _LLM(Usage(1_000_000, 1_000_000))  # 0.30 + 2.50 for gemini-3.5-flash-lite

    ask(MeteredLLM(inner, ledger(tmp_path, None), "openrouter"))

    (row,) = lines(tmp_path)
    assert row["basis"] == "estimate"
    assert row["cost_usd"] == pytest.approx(2.80)


@pytest.mark.unit
def test_openrouter_request_of_an_unpriced_model_without_cost_records_no_cost(
    tmp_path: Path,
) -> None:
    ask(MeteredLLM(_LLM(Usage(5, 5), "vendor/unknown"), ledger(tmp_path, 5.0), "openrouter"))

    (row,) = lines(tmp_path)
    assert (row["basis"], row["cost_usd"]) == ("estimate", None)


@pytest.mark.unit
def test_a_request_over_the_budget_is_not_sent_and_nothing_is_recorded(tmp_path: Path) -> None:
    write_line(tmp_path / LEDGER_FILE, "2026-11-02T00:00:00+00:00", 5.0)
    inner = _LLM(Usage(10, 10, 0.001))

    with pytest.raises(BudgetExceededError):
        ask(MeteredLLM(inner, ledger(tmp_path, 5.0), "openrouter"))

    assert inner.calls == 0
    assert len(lines(tmp_path)) == 1  # only the old line


@pytest.mark.unit
def test_the_estimate_of_a_long_prompt_can_stop_a_request_the_month_still_affords(
    tmp_path: Path,
) -> None:
    inner = _LLM(Usage(1, 1, 0.0), "google/gemini-3.7-flash")
    long_history: list[HistoryItem] = [UserMessage("я" * 2_000_000)]  # ~1M tokens: 0.75 USD
    write_line(tmp_path / LEDGER_FILE, "2026-11-02T00:00:00+00:00", 0.5)

    with pytest.raises(BudgetExceededError):
        MeteredLLM(inner, ledger(tmp_path, 1.0), "openrouter").generate("s", long_history, [])

    assert inner.calls == 0


@pytest.mark.unit
def test_free_gemini_requests_are_recorded_at_zero_and_never_blocked(tmp_path: Path) -> None:
    write_line(tmp_path / LEDGER_FILE, "2026-11-02T00:00:00+00:00", 99.0)  # far over the budget
    inner = _LLM(Usage(100, 10), "gemini-3.5-flash-lite")

    ask(MeteredLLM(inner, ledger(tmp_path, 1.0), "gemini"))

    assert inner.calls == 1
    new = lines(tmp_path)[-1]
    assert (new["provider"], new["basis"], new["cost_usd"]) == ("gemini", "free", 0.0)


@pytest.mark.unit
def test_a_failed_request_is_not_recorded(tmp_path: Path) -> None:
    class Failing(_LLM):
        def generate(
            self,
            system: str,  # noqa: ARG002
            history: Sequence[HistoryItem],  # noqa: ARG002
            tools: Sequence[ToolSpec],  # noqa: ARG002
            *,
            allow_tools: bool = True,  # noqa: ARG002
        ) -> ModelTurn:
            msg = "down"
            raise ExternalServiceError(msg)

    with pytest.raises(ExternalServiceError):
        ask(MeteredLLM(Failing(Usage()), ledger(tmp_path, 5.0), "openrouter"))

    assert not (tmp_path / LEDGER_FILE).exists()


@pytest.mark.unit
def test_the_metered_client_reports_the_model_of_the_inner_one(tmp_path: Path) -> None:
    metered = MeteredLLM(_LLM(Usage(), "x/y"), ledger(tmp_path, None), "openrouter")

    assert metered.model == "x/y"


class _STT:
    def __init__(self, raw: RawTranscript, model: str = "elevenlabs/scribe-v2") -> None:
        self.model = model
        self.raw = raw
        self.calls = 0

    def transcribe(
        self,
        audio: bytes,  # noqa: ARG002
        mime_type: str,  # noqa: ARG002
        *,
        vocabulary: Sequence[str] | None = None,  # noqa: ARG002
        duration_s: float | None = None,  # noqa: ARG002
    ) -> RawTranscript:
        self.calls += 1
        return self.raw


@pytest.mark.unit
def test_transcription_is_recorded_with_seconds_and_the_reported_cost(tmp_path: Path) -> None:
    inner = _STT(RawTranscript("т", (), 0, 0, 99.5, 0.00608))

    MeteredTranscriber(inner, ledger(tmp_path, 1.0), "openrouter").transcribe(
        b"a", "audio/mp3", duration_s=99.0
    )

    (row,) = lines(tmp_path)
    assert (row["kind"], row["basis"], row["cost_usd"]) == ("transcribe", "actual", 0.00608)
    assert row["audio_seconds"] == 99.5  # the service's figure wins over the caller's


@pytest.mark.unit
def test_transcription_without_a_reported_cost_is_estimated_per_second(tmp_path: Path) -> None:
    inner = _STT(RawTranscript("т", (), 0, 0))

    MeteredTranscriber(inner, ledger(tmp_path, None), "openrouter").transcribe(
        b"a", "audio/mp3", duration_s=100.0
    )

    (row,) = lines(tmp_path)
    assert row["basis"] == "estimate"
    assert row["cost_usd"] == pytest.approx(100 * 0.0000611111)
    assert row["audio_seconds"] == 100.0


@pytest.mark.unit
def test_a_transcription_over_the_budget_is_not_sent(tmp_path: Path) -> None:
    write_line(tmp_path / LEDGER_FILE, "2026-11-02T00:00:00+00:00", 0.999)
    inner = _STT(RawTranscript("т", (), 0, 0, 100.0, 0.006))

    with pytest.raises(BudgetExceededError):
        MeteredTranscriber(inner, ledger(tmp_path, 1.0), "openrouter").transcribe(
            b"a",
            "audio/mp3",
            duration_s=100.0,  # estimated 0.0061
        )

    assert inner.calls == 0


@pytest.mark.unit
def test_free_gemini_transcription_is_recorded_at_zero(tmp_path: Path) -> None:
    inner = _STT(RawTranscript("т", (), 2477, 365), "gemini-3.5-transcribe")

    MeteredTranscriber(inner, ledger(tmp_path, 0.0), "gemini").transcribe(
        b"a", "audio/mp3", duration_s=99.0
    )

    (row,) = lines(tmp_path)
    assert (row["provider"], row["basis"], row["cost_usd"]) == ("gemini", "free", 0.0)
    assert row["audio_seconds"] == 99.0
    assert row["tokens_in"] == 2477
