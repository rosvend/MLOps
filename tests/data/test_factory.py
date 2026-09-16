import pytest

from src.config import DataSourceConfig
from src.data.csv_source import CsvDataSource
from src.data.factory import build_source


def test_builds_the_csv_adapter(fixture_csv):
    config = DataSourceConfig(type="csv", path=str(fixture_csv), separator=";")

    assert isinstance(build_source(config), CsvDataSource)


def test_built_source_reads_the_configured_file(fixture_csv):
    config = DataSourceConfig(type="csv", path=str(fixture_csv), separator=";")

    assert build_source(config).read().shape == (14, 24)  # 23 source columns + cliente_id


def test_unknown_type_fails_loudly():
    config = DataSourceConfig(type="postgres", path="anywhere")

    with pytest.raises(ValueError, match="postgres"):
        build_source(config)
