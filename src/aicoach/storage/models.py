"""Pydantic models at the edge of the event store."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aicoach.domain.events import EventType

# Video seconds: finite and non-negative; strict so that "12" or true is not read as a time.
VideoTime = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Source = Literal["plates", "model", "speech", "merge"]
Team = Literal["home", "away"]
EditAction = Literal["add", "move", "retype", "delete", "verify"]

# Which of (target, type, t_v) each action must carry.
_REQUIRED_FIELDS: dict[str, tuple[bool, bool, bool]] = {
    "add": (False, True, True),
    "move": (True, False, True),
    "retype": (True, True, False),
    "delete": (True, False, False),
    "verify": (True, False, False),
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class NewPrediction(_Frozen):
    """One prediction to be stored. `half` is 1, 2 or None when unknown."""

    type: EventType
    t_v: VideoTime
    source: Source
    half: Literal[1, 2] | None = None
    score: Annotated[float, Field(allow_inf_nan=False)] | None = None
    status: str = "auto"
    team: Team | None = None


class NewEdit(_Frozen):
    """A user edit before it is written; checks which fields each action needs.

    `add` needs `type` and `t_v` and no target; `move` needs a target and `t_v`; `retype`
    needs a target and `type`; `delete` and `verify` need only a target.
    """

    match_id: str = Field(min_length=1)
    action: EditAction
    author: str = Field(min_length=1)
    target: str | None = Field(default=None, pattern=r"^[pe]:[0-9]+$")
    type: EventType | None = None
    t_v: VideoTime | None = None

    @model_validator(mode="after")
    def _fields_match_action(self) -> Self:
        needs = _REQUIRED_FIELDS[self.action]
        has = (self.target is not None, self.type is not None, self.t_v is not None)
        if has != needs:
            msg = f"action {self.action!r} needs (target, type, t_v) presence {needs}, got {has}"
            raise ValueError(msg)
        return self


class EditRecord(_Frozen):
    """A stored row of the `edits` journal."""

    id: int
    match_id: str
    action: Literal["add", "move", "retype", "delete", "verify", "undo"]
    target: str | None
    type: EventType | None
    t_v: float | None
    undo_of: int | None
    author: str
    created_at: str

    @property
    def event_id(self) -> str | None:
        """For an `add`, the id of the event it created (`e:<edit id>`); otherwise `None`."""
        return f"e:{self.id}" if self.action == "add" else None


class CurrentEvent(_Frozen):
    """One row of the `events_current` view.

    `deleted` events are hidden by `get_current_events` unless asked for; `n_edits` is the
    number of effective edits that target the event. Events added by a user have
    `origin="user"`, `status="manual"`, and no `half`, `score`, `team` or `source`.
    """

    event_id: str
    match_id: str
    type: EventType
    t_v: float
    half: Literal[1, 2] | None
    score: float | None
    status: str
    team: Team | None
    source: Source | None
    origin: Literal["prediction", "user"]
    verified: bool
    deleted: bool
    n_edits: int
