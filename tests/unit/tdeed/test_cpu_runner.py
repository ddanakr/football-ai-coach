"""The launcher that runs `inference.py` on the processor, tested with a stand-in torch."""

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from aicoach.tdeed.cpu_runner import main, patch_torch


class _Module:
    def __init__(self) -> None:
        self.moved_to: list[Any] = []

    def to(self, *args: Any, **kwargs: Any) -> "_Module":
        self.moved_to.append((args, kwargs))
        return self

    def cuda(self) -> str:
        return "on card"


class _Tensor(_Module):
    pass


class _Device:
    type = "cuda"


def _torch() -> Any:
    loads: list[dict[str, Any]] = []

    def load(*_args: Any, **kwargs: Any) -> str:
        loads.append(kwargs)
        return "weights"

    return types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=_Module), Tensor=_Tensor, load=load, loads=loads
    )


@pytest.mark.unit
def test_patch_torch_turns_cuda_into_cpu_everywhere() -> None:
    torch = _torch()
    patch_torch(torch)
    module, tensor = torch.nn.Module(), torch.Tensor()

    module.to("cuda")
    module.to(device="cuda:0")
    tensor.to(_Device())
    tensor.to("cpu", non_blocking=True)

    assert module.moved_to == [(("cpu",), {}), ((), {"device": "cpu"})]
    assert tensor.moved_to == [(("cpu",), {}), (("cpu",), {"non_blocking": True})]


@pytest.mark.unit
def test_patch_torch_cuda_calls_return_the_object_and_load_goes_to_the_cpu() -> None:
    torch = _torch()
    patch_torch(torch)
    module = torch.nn.Module()

    assert module.cuda() is module
    assert torch.Tensor().cuda().__class__ is _Tensor
    assert torch.load("file") == "weights"
    assert torch.load("file", map_location="elsewhere") == "weights"
    assert torch.loads == [{"map_location": "cpu"}, {"map_location": "elsewhere"}]


@pytest.mark.unit
def test_main_runs_the_script_as_main_with_its_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = _torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    script = tmp_path / "inference.py"
    script.write_text(
        "import sys\n"
        "if __name__ == '__main__':\n"
        "    open(sys.argv[1], 'w').write(' '.join(sys.argv[2:]))\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.txt"
    saved_argv, saved_path = list(sys.argv), list(sys.path)

    try:
        code = main([str(script), str(out), "--model", "x"])
    finally:
        sys.argv, sys.path[:] = saved_argv, saved_path

    assert code == 0
    assert out.read_text() == "--model x"
    assert torch.nn.Module.to.__name__ == "to_module"  # the patch was applied
