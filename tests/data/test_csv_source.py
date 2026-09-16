import pandas as pd

from src.data.csv_source import CsvDataSource


def test_reads_every_row_and_column(raw):
    assert raw.shape == (14, 24)  # 23 source columns + cliente_id


def test_honours_the_configured_separator(tmp_path):
    path = tmp_path / "comma.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    assert CsvDataSource(path=path, separator=",").read().columns.tolist() == ["a", "b"]
    assert CsvDataSource(path=path, separator=";").read().columns.tolist() == ["a,b"]


def test_leaves_comma_decimals_untouched(raw):
    assert raw["puntaje"].iloc[0] == "95,227787"


def test_strips_the_utf8_byte_order_mark(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes("﻿a;b\n1;2\n".encode("utf-8"))

    assert CsvDataSource(path=path, separator=";").read().columns.tolist() == ["a", "b"]


def test_returns_a_dataframe(raw):
    assert isinstance(raw, pd.DataFrame)
