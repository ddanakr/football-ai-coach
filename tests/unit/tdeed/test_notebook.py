"""The Colab notebook of the `tdeed` command: valid, cleared, and free of personal paths."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "tdeed_colab.ipynb"


def _load() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return data


@pytest.mark.unit
def test_notebook_is_valid_nbformat_4_with_a_gpu_accelerator() -> None:
    data = _load()

    assert data["nbformat"] == 4
    assert data["metadata"]["accelerator"] == "GPU"
    assert all(cell["cell_type"] in ("code", "markdown") for cell in data["cells"])
    ids = [cell["id"] for cell in data["cells"]]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_notebook_has_no_outputs_and_no_execution_counts() -> None:
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.unit
def test_notebook_has_no_windows_paths_user_homes_or_tokens() -> None:
    text = " ".join("".join(c["source"]) for c in _load()["cells"])  # the decoded cell text

    assert not re.search(r"\b[A-Za-z]:[\\/]", text)  # a drive path: C:\ or D:/
    assert "/Users/" not in text
    assert not re.search(r"(ghp_|github_pat_|hf_|sk-|AIza|api[_-]?key|token)", text, re.IGNORECASE)


@pytest.mark.unit
def test_notebook_parameters_come_first_and_the_steps_are_in_order() -> None:
    cells = _load()["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]

    for name in ("REPO_URL", "BRANCH", "VIDEO_ON_DRIVE", "FROM", "TO"):
        assert re.search(rf"^{name} = ", sources[0], re.MULTILINE), name
    steps = [
        "nvidia-smi",
        "git clone",
        ".[gpu]",
        "drive.mount",
        "aicoach.cli tdeed",
        "destination.exists",
        "status.json",
    ]
    positions = [next(i for i, s in enumerate(sources) if step in s) for step in steps]
    assert positions == sorted(positions)


@pytest.mark.unit
def test_notebook_never_overwrites_files_on_drive_and_does_not_import_tdeed_code() -> None:
    text = "".join("".join(c["source"]) for c in _load()["cells"])

    assert "destination.exists()" in text  # the copy back is no-clobber
    assert "inference.py" not in text  # the notebook only calls the command
