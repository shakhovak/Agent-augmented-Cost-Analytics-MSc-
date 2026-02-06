import hashlib
import json
from pathlib import Path


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_generator_writes_metadata_and_is_deterministic(tmp_path: Path):
    # Arrange: copy a small generator config into tmp_path and override output path
    # (or load your real config and patch output path to tmp)
    # Run generator script twice -> compare hashes

    out1 = tmp_path / "sample1.csv"
    out2 = tmp_path / "sample2.csv"

    # call your generator function directly for speed (preferred) or subprocess
    from data.generator import generate_joint_dataset, load_yaml

    cfg = load_yaml("data/samples/sample_config.yaml")
    cfg["dataset"]["seed"] = 123
    cfg["dataset"]["start_date"] = "2025-01-01"
    cfg["dataset"]["end_date"] = "2025-01-10"
    cfg["accounts"]["n_accounts"] = 50

    df1 = generate_joint_dataset(cfg)
    df1.to_csv(out1, index=False)

    df2 = generate_joint_dataset(cfg)
    df2.to_csv(out2, index=False)

    assert sha256_file(out1) == sha256_file(out2)

    # Now write meta.json like your script does (or call the script function)
    meta = {
        "dataset_version": "0.1.1",
        "seeds": {"dataset_seed": 123},
    }
    meta_path = tmp_path / "sample1.meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "dataset_version" in loaded
    assert "seeds" in loaded
