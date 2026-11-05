"""Project exception hierarchy. Every error raised on purpose derives from `AicoachError`."""


class AicoachError(Exception):
    """Base class for all errors raised by the aicoach package."""


class EvalError(AicoachError):
    """Invalid input to the evaluation module (labels file, tolerance, parameters)."""
