import pytest

import aicoach


@pytest.mark.unit
def test_package_import_succeeds() -> None:
    assert aicoach.__version__
