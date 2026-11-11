"""Fake stage functions for the pipeline tests: they record their calls and can fail."""

from typing import Any

from aicoach.errors import PlatesError
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.schemas import CandidatesResult
from aicoach.pipeline import Stages
from aicoach.plates.schemas import PlatesResult
from aicoach.scoreboard.schemas import ScoreboardResult
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
        return Stages(scoreboard=self.scoreboard, plates=self.plates, fusion=self.fusion)
