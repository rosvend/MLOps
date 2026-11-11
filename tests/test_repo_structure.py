from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = yaml.safe_load((ROOT / "config/ci/repo_structure.yaml").read_text())["required"]


def test_required_list_is_not_empty():
    assert REQUIRED


@pytest.mark.parametrize("path", REQUIRED)
def test_required_path_exists(path):
    target = ROOT / path
    assert target.is_dir() if path.endswith("/") else target.is_file(), f"missing: {path}"
