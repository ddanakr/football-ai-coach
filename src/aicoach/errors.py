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
