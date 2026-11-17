"""Provider-neutral messages between the agent loop and a language-model client."""

from dataclasses import dataclass
from typing import Any  # JSON exchanged with the model: its shape is the model's, not ours


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call: `parameters` is a JSON schema (the OpenAPI subset)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A call the model asked for."""

    name: str
    args: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """The answer to one call, in the order of the calls."""

    name: str
    result: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class Usage:
    """Tokens of one request; thinking tokens are counted as output (they are billed so).

    `cost_usd` is what the service itself charged (OpenRouter reports it in every response);
    `None` when the service does not say. Sums keep a cost only when no part lacks one.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: "Usage") -> "Usage":
        # An empty `Usage()` is the start of a sum: its missing cost must not hide the other's.
        if self == Usage():
            return other
        if other == Usage():
            return self
        cost = None
        if self.cost_usd is not None and other.cost_usd is not None:
            cost = self.cost_usd + other.cost_usd
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens, cost
        )


@dataclass(frozen=True)
class UserMessage:
    """Text from the user (the question, or the validator's complaint about the last answer)."""

    text: str


@dataclass(frozen=True)
class ModelTurn:
    """What the model said: `text`, or `calls` (or both).

    `raw` is the provider's own object for this turn; a client must get it back unchanged in the
    history (some models need their hidden reasoning signatures returned with the next request).
    """

    text: str
    calls: tuple[ToolCall, ...]
    usage: Usage
    raw: object | None = None
    truncated: bool = False  # the model stopped at its length limit: the text is cut off


@dataclass(frozen=True)
class ToolResults:
    """The results of the calls of the previous `ModelTurn`."""

    results: tuple[ToolResult, ...]


HistoryItem = UserMessage | ModelTurn | ToolResults
