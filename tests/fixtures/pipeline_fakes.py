"""Fake stage functions for the pipeline tests: they record their calls and can fail."""

from typing import Any

from aicoach.errors import PlatesError
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.schemas import CandidatesResult
from aicoach.pipeline import Stages
from aicoach.plate_names.schemas import PlateNamesResult
from aicoach.plates.schemas import PlatesResult
from aicoach.review.schemas import ReviewCandidatesResult
from aicoach.scoreboard.schemas import ScoreboardResult
from aicoach.speech.transcript import TranscriptResult
from tests.fixtures.fusion_synth import plates_result
from tests.fixtures.plates_synth import scoreboard_result


# Any: the fakes accept whatever arguments the real stage functions are called with.
class Recorder:
    """Fake stages that record their calls in order; one of them can be made to fail."""

    def __init__(self, fail_at: str | None = None, error: BaseException | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.fail_at = fail_at
        self.error = error or PlatesError("no scoreboard.json")

    def _record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls.append((name, args, kwargs))
        if name == self.fail_at:
            raise self.error

    def scoreboard(self, *args: Any, **kwargs: Any) -> ScoreboardResult:
        self._record("scoreboard", args, kwargs)
        return scoreboard_result()

    def plates(self, *args: Any, **kwargs: Any) -> PlatesResult:
        self._record("plates", args, kwargs)
        return plates_result([])

    def plate_names(self, *args: Any, **kwargs: Any) -> PlateNamesResult | None:
        self._record("plate_names", args, kwargs)
        return None  # as a skipped step: the ingest must not depend on the names

    def transcribe(self, *args: Any, **kwargs: Any) -> TranscriptResult | None:
        self._record("transcribe", args, kwargs)
        return None  # as a skipped step: the ingest must not depend on the commentary

    def review_candidates(self, *args: Any, **kwargs: Any) -> ReviewCandidatesResult:
        self._record("review_candidates", args, kwargs)
        return ReviewCandidatesResult(
            match_id="m",
            cache_key="k",
            events_version="a",
            model_version="b",
            scoreboard_version="c",
            model_file="tdeed_x.json",
            merge_radius_s=3.0,
            n_model=0,
            n_existing=0,
            n_merged=0,
            n_outside_halves=0,
            half_bounds={},
            candidates=[],
            counts={},
        )

    def fusion(self, *args: Any, **kwargs: Any) -> CandidatesResult:
        self._record("fusion", args, kwargs)
        return CandidatesResult(
            match_id="m",
            cache_key="k",
            run_id=None,
            mode="no_model",
            scoreboard_version="a",
            plates_version="b",
            model_version=None,
            model_file=None,
            config=FuseConfig(),
            half_bounds={},
            candidates=[],
            dropped=[],
            warnings=[],
            counts={},
        )

    def stages(self) -> Stages:
        return Stages(
            scoreboard=self.scoreboard,
            plates=self.plates,
            fusion=self.fusion,
            plate_names=self.plate_names,
            transcribe=self.transcribe,
            review_candidates=self.review_candidates,
        )
