from pathlib import Path

import pytest


@pytest.fixture
def small_sample_csv_path() -> str:
    return str(Path("data/samples/joint_info_sample.csv"))
