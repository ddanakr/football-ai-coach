"""The work directory of one run: what `inference.py` reads from and writes into its current
directory, kept apart from the cached copy of T-DEED so that the copy stays as it was fetched."""

import json
import shutil
from pathlib import Path

from aicoach.errors import SpottingError
from aicoach.tdeed.config import BATCH_SIZE, MODEL, MODEL_DATASET, NUM_WORKERS

RAW_OUTPUT = Path("inference_output") / "results_inference.json"


def prepare_workdir(workdir: Path, repo_dir: Path, checkpoint: Path) -> None:
    """Lay out `config/`, `data/` and `checkpoints/` for `inference.py`.

    The config is a copy of the model's own with the batch size and the number of workers set
    (`inference.py` takes both from it and has no flags for them); the class list is copied as it
    is and the checkpoint is linked (copied where links are not possible).

    Raises:
        SpottingError: A file of the T-DEED copy is missing or not readable, or the work directory
            cannot be written.
    """
    config_source = repo_dir / "config" / MODEL_DATASET / f"{MODEL}.json"
    classes_source = repo_dir / "data" / "soccernet" / "class.txt"
    config_target = workdir / "config" / MODEL_DATASET / f"{MODEL}.json"
    classes_target = workdir / "data" / "soccernet" / "class.txt"
    checkpoint_target = workdir / "checkpoints" / MODEL_DATASET / MODEL / "checkpoint_best.pt"
    try:
        config = json.loads(config_source.read_text(encoding="utf-8"))
        config["batch_size"] = BATCH_SIZE
        config["num_workers"] = NUM_WORKERS
        for target in (config_target, classes_target, checkpoint_target):
            target.parent.mkdir(parents=True, exist_ok=True)
        config_target.write_text(json.dumps(config, indent=4), encoding="utf-8")
        shutil.copyfile(classes_source, classes_target)
        _link_or_copy(checkpoint, checkpoint_target)
    except (OSError, ValueError) as err:
        msg = f"cannot prepare the T-DEED work directory {workdir}: {err}"
        raise SpottingError(msg) from err


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        target.hardlink_to(source)
    except OSError:  # another drive, or a file system without links
        shutil.copyfile(source, target)
