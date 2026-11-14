"""Project exception hierarchy. Every error raised on purpose derives from `AicoachError`."""


class AicoachError(Exception):
    """Base class for all errors raised by the aicoach package."""


class EvalError(AicoachError):
    """Invalid input to the evaluation module (labels file, tolerance, parameters)."""


class StorageError(AicoachError):
    """The event store rejected an operation (invalid edit, unknown target, SQLite failure)."""


class ScoreboardParseError(AicoachError):
    """The scoreboard stage cannot continue: no digit templates, a broken video, bad input."""


class ConfigError(AicoachError):
    """Invalid configuration: a bad value in a setting, an unknown log level."""


class PlatesError(AicoachError):
    """The plates stage cannot continue: no scoreboard.json, mismatched decode mode, bad input."""


class SpottingError(AicoachError):
    """Invalid input to the spotting adapter: a broken T-DEED file or a bad parameter."""


class FusionError(AicoachError):
    """The fusion stage cannot continue: missing or stale inputs, no half bounds, lost edits."""


class ExportError(AicoachError):
    """A clip cannot be exported: unknown event, bad window, missing or failing FFmpeg."""


class IngestError(AicoachError):
    """A match cannot be ingested: bad link, unavailable video, wrong frame size, no audio."""
