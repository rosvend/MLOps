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


class _FrameSource:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def read(self) -> pd.DataFrame:
        return self._frame.copy()


@pytest.fixture(scope="session")
def real_source() -> _FrameSource:
    """The full portfolio from the configured source; skipped where cloud access is absent."""
    from google.api_core.exceptions import GoogleAPICallError
    from google.auth.exceptions import GoogleAuthError

    from src.config import load_config
    from src.data.factory import build_source

    try:
        return _FrameSource(build_source(load_config().data_source).read())
    except (GoogleAuthError, GoogleAPICallError, OSError) as error:
        pytest.skip(f"datos reales no disponibles: {type(error).__name__}")
