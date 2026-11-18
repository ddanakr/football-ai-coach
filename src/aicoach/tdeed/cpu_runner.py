"""Runs T-DEED's `inference.py` on the processor, without changing the file.

`inference.py` names the video card in its code (`TDEEDModel(device='cuda')`, `torch.load` without
a target device), so on a machine without one it stops at once. This launcher is a separate
process: it makes torch read "cuda" as "cpu" and then runs the script as `__main__`. Nothing of
T-DEED is imported by the rest of the package, and the script on disk stays as it is (D-046).

Usage: `python cpu_runner.py <inference.py> <its arguments>`; it imports nothing of the package.
"""

import importlib
import runpy
import sys
from pathlib import Path
from typing import Any


def _as_cpu(value: Any) -> Any:  # Any: a device argument is a str, a torch.device or anything else
    """`cuda`, `cuda:0` and `torch.device('cuda')` become `cpu`; everything else is unchanged."""
    if isinstance(value, str) and value.startswith("cuda"):
        return "cpu"
    if getattr(value, "type", None) == "cuda":
        return "cpu"
    return value


def patch_torch(torch: Any) -> None:  # Any: torch ships no stubs here
    """Make torch ignore requests for the video card."""
    module_to = torch.nn.Module.to
    tensor_to = torch.Tensor.to
    load = torch.load

    def to_module(self: Any, *args: Any, **kwargs: Any) -> Any:
        return module_to(
            self, *[_as_cpu(a) for a in args], **{k: _as_cpu(v) for k, v in kwargs.items()}
        )

    def to_tensor(self: Any, *args: Any, **kwargs: Any) -> Any:
        return tensor_to(
            self, *[_as_cpu(a) for a in args], **{k: _as_cpu(v) for k, v in kwargs.items()}
        )

    def load_on_cpu(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("map_location", "cpu")
        return load(*args, **kwargs)

    torch.nn.Module.to = to_module
    torch.Tensor.to = to_tensor
    torch.nn.Module.cuda = lambda self, *_a, **_k: self
    torch.Tensor.cuda = lambda self, *_a, **_k: self
    torch.load = load_on_cpu


def main(argv: list[str]) -> int:
    """Patch torch, then run `argv[0]` (the script) with the arguments after it."""
    # By name: torch is an optional dependency and has no stubs here.
    patch_torch(importlib.import_module("torch"))
    script = Path(argv[0]).resolve()
    if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
        sys.path.pop(0)  # this directory holds modules (`config`, `output`) that must not shadow
    sys.argv = [str(script), *argv[1:]]
    sys.path.insert(0, str(script.parent))  # what running the script by its path does
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":  # pragma: no cover  # needs torch and the T-DEED code
    sys.exit(main(sys.argv[1:]))
