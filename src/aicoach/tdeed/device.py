"""Which device T-DEED runs on: the video card, or the processor when asked for explicitly."""

import importlib
from typing import Literal

from aicoach.errors import SpottingError
from aicoach.i18n import ru
from aicoach.tdeed.config import NOTEBOOK_URL

Device = Literal["cuda", "cpu"]
DEVICE_CHOICES = ("auto", "cuda", "cpu")


def cuda_available() -> bool:
    """Whether torch is installed and sees a CUDA video card.

    torch is an optional dependency (`pip install .[gpu]`), so it is imported here and not at
    the top of the module; a missing or broken install means "no card".
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False
    try:
        # Any: torch is imported by name and ships no stubs here; `is_available` returns a bool.
        return bool(torch.cuda.is_available())
    except (RuntimeError, AttributeError):
        return False


def resolve_device(requested: str, cuda: bool) -> Device:
    """The device to run on.

    `auto` takes the video card and fails without one: T-DEED on the processor is far slower, so
    it needs the explicit `cpu`. `cuda` without a card fails the same way.

    Raises:
        SpottingError: An unknown name, or no CUDA for `auto` or `cuda` (the message points to the
            notebook).
    """
    if requested == "cpu":
        return "cpu"
    if requested not in ("auto", "cuda"):
        msg = f"unknown device {requested!r}: use one of {', '.join(DEVICE_CHOICES)}"
        raise SpottingError(msg)
    if not cuda:
        raise SpottingError(ru.TDEED_NO_CUDA.format(url=NOTEBOOK_URL))
    return "cuda"
