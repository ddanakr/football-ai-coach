"""Fixtures of the T-DEED tests."""

import pytest

from tests.fixtures.tdeed_synth import CHECKPOINT_SHA, Fakes


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    """Fakes of git, FFmpeg, `inference.py` and the download, with the known checkpoint hash
    replaced by that of the fake checkpoint."""
    monkeypatch.setattr("aicoach.tdeed.stage.CHECKPOINT_SHA256", CHECKPOINT_SHA)
    return Fakes()
