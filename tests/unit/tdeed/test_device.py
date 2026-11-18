"""Which device T-DEED runs on, and the message without CUDA."""

import sys
import types

import pytest

from aicoach.errors import SpottingError
from aicoach.tdeed.config import NOTEBOOK_URL
from aicoach.tdeed.device import cuda_available, resolve_device


@pytest.mark.unit
@pytest.mark.parametrize(
    ("requested", "cuda", "expected"),
    [("auto", True, "cuda"), ("cuda", True, "cuda"), ("cpu", True, "cpu"), ("cpu", False, "cpu")],
)
def test_resolve_device_valid_choices(requested: str, cuda: bool, expected: str) -> None:
    assert resolve_device(requested, cuda) == expected


@pytest.mark.unit
@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_resolve_device_without_cuda_names_the_notebook_and_the_cpu_flag(requested: str) -> None:
    with pytest.raises(SpottingError) as caught:
        resolve_device(requested, False)

    message = str(caught.value)
    assert NOTEBOOK_URL in message
    assert "--device cpu" in message


@pytest.mark.unit
@pytest.mark.parametrize("requested", ["tpu", "", "CUDA"])
def test_resolve_device_unknown_name_raises(requested: str) -> None:
    with pytest.raises(SpottingError, match="unknown device"):
        resolve_device(requested, True)


def _fake_torch(monkeypatch: pytest.MonkeyPatch, available: object) -> None:
    def is_available() -> object:
        if isinstance(available, Exception):
            raise available
        return available

    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=is_available)  # type: ignore[attr-defined]  # a stand-in module
    monkeypatch.setitem(sys.modules, "torch", module)


@pytest.mark.unit
@pytest.mark.parametrize(("answer", "expected"), [(True, True), (False, False)])
def test_cuda_available_follows_torch(
    monkeypatch: pytest.MonkeyPatch, answer: bool, expected: bool
) -> None:
    _fake_torch(monkeypatch, answer)

    assert cuda_available() is expected


@pytest.mark.unit
def test_cuda_available_without_torch_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # an import of it raises ImportError

    assert cuda_available() is False


@pytest.mark.unit
def test_cuda_available_broken_driver_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch, RuntimeError("driver"))

    assert cuda_available() is False
