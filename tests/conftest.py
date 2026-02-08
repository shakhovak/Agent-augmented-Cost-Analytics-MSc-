from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)


@pytest.fixture
def small_sample_csv_path() -> str:
    return str(Path("data/samples/joint_info_sample.csv"))
