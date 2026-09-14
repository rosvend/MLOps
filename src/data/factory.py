from src.config import DataSourceConfig
from src.data.csv_source import CsvDataSource
from src.data.source import DataSource

SOURCES = {"csv": CsvDataSource}


def build_source(config: DataSourceConfig) -> DataSource:
    if config.type not in SOURCES:
        raise ValueError(f"Unknown data source type {config.type!r}. Available: {sorted(SOURCES)}")
    return SOURCES[config.type](**config.options)
