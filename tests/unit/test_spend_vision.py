"""`MeteredVideoModel`: the budget check before a video request and the record after it."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aicoach.agent.pricing import VIDEO_TOKENS_PER_S, estimate_video_request_usd
from aicoach.errors import BudgetExceededError, ExternalServiceError
from aicoach.spend import LEDGER_FILE, MeteredVideoModel, SpendLedger
from tests.fixtures.episode_synth import FakeVideoModel, reply

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
PRICED = "google/gemini-3.7-flash"


def make(
    tmp_path: Path, inner: FakeVideoModel, budget: float | None, provider: str = "openrouter"
) -> MeteredVideoModel:
    ledger = SpendLedger(tmp_path / LEDGER_FILE, budget, clock=lambda: NOW)
    return MeteredVideoModel(inner, ledger, provider)  # type: ignore[arg-type]  # str is a Provider


def rows(tmp_path: Path) -> list[dict[str, object]]:
    text = (tmp_path / LEDGER_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


@pytest.mark.unit
def test_a_request_is_recorded_as_vision_with_the_cost_the_service_charged(
    tmp_path: Path,
) -> None:
    metered = make(tmp_path, FakeVideoModel([reply(cost=0.0035)], PRICED), budget=1.0)

    result = metered.describe("s", "p", b"clip", video_seconds=20.0)

    assert result.text  # the reply passes through unchanged
    (row,) = rows(tmp_path)
    assert row["kind"] == "vision"
    assert row["basis"] == "actual"
    assert row["cost_usd"] == 0.0035
    assert (row["tokens_in"], row["tokens_out"]) == (2100, 480)
    assert (row["provider"], row["model"]) == ("openrouter", PRICED)
    assert metered.model == PRICED


@pytest.mark.unit
def test_without_a_reported_cost_the_table_price_is_the_estimate(tmp_path: Path) -> None:
    metered = make(tmp_path, FakeVideoModel([reply(cost=None)], PRICED), budget=None)

    metered.describe("s", "p", b"clip", video_seconds=20.0)

    (row,) = rows(tmp_path)
    assert row["basis"] == "estimate"
    assert row["cost_usd"] == pytest.approx((2100 * 0.75 + 480 * 3.75) / 1_000_000)


@pytest.mark.unit
def test_a_request_over_the_budget_is_not_sent_and_nothing_is_recorded(tmp_path: Path) -> None:
    inner = FakeVideoModel([], PRICED)
    metered = make(tmp_path, inner, budget=0.0)

    with pytest.raises(BudgetExceededError):
        metered.describe("s", "p", b"clip", video_seconds=20.0)

    assert inner.requests == []
    assert not (tmp_path / LEDGER_FILE).exists()


@pytest.mark.unit
def test_a_model_without_a_price_passes_the_check_and_is_recorded_without_a_cost(
    tmp_path: Path,
) -> None:
    metered = make(tmp_path, FakeVideoModel([reply(cost=None)], "unknown/vlm"), budget=0.0)

    metered.describe("s", "p", b"clip", video_seconds=20.0)

    (row,) = rows(tmp_path)
    assert row["cost_usd"] is None
    assert row["basis"] == "estimate"


@pytest.mark.unit
def test_a_failed_request_records_nothing(tmp_path: Path) -> None:
    inner = FakeVideoModel([ExternalServiceError("boom")], PRICED)

    with pytest.raises(ExternalServiceError):
        make(tmp_path, inner, budget=5.0).describe("s", "p", b"clip", video_seconds=20.0)

    assert not (tmp_path / LEDGER_FILE).exists()


@pytest.mark.unit
def test_the_free_provider_is_recorded_at_zero_and_never_blocked(tmp_path: Path) -> None:
    metered = make(tmp_path, FakeVideoModel([reply()], PRICED), budget=0.0, provider="gemini")

    metered.describe("s", "p", b"clip", video_seconds=20.0)

    (row,) = rows(tmp_path)
    assert (row["cost_usd"], row["basis"]) == (0.0, "free")


@pytest.mark.unit
def test_the_estimate_grows_with_the_length_of_the_clip_and_has_no_price_for_unknown_models() -> (
    None
):
    short = estimate_video_request_usd(PRICED, 1000, 10.0)
    long = estimate_video_request_usd(PRICED, 1000, 20.0)

    assert short is not None
    assert long is not None
    assert long - short == pytest.approx(10 * VIDEO_TOKENS_PER_S * 0.75 / 1_000_000)
    assert estimate_video_request_usd("unknown/vlm", 1000, 20.0) is None
    assert estimate_video_request_usd(PRICED, -5, -1.0) == estimate_video_request_usd(PRICED, 0, 0)
