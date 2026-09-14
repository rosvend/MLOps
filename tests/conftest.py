from pathlib import Path

import pandas as pd
import pytest

from src.data.csv_source import CsvDataSource

_FIXTURE_CSV = Path(__file__).parent / "fixtures" / "creditos_sample.csv"


@pytest.fixture
def fixture_csv() -> Path:
    return _FIXTURE_CSV


@pytest.fixture
def sample_source(fixture_csv) -> CsvDataSource:
    return CsvDataSource(path=fixture_csv, separator=";")


@pytest.fixture
def raw(sample_source) -> pd.DataFrame:
    return sample_source.read()
