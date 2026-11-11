import pytest
from pydantic import ValidationError

from src.automl.spec import default_automl_spec


def test_default_spec_loads_with_hard_budget_cap():
    spec = default_automl_spec()
    assert spec.budget_milli_node_hours == 1000
    assert spec.optimization_objective == "maximize-au-prc"


def test_label_and_split_are_never_features():
    spec = default_automl_spec()
    assert spec.label_column not in spec.column_specs
    assert spec.split_column not in spec.column_specs


def test_column_specs_cover_every_csv_feature():
    import csv
    from pathlib import Path

    path = Path("data/gcp/loans_split.csv")
    if not path.exists():
        pytest.skip("local training file not exported")
    header = next(csv.reader(open(path)))
    spec = default_automl_spec()
    assert set(spec.column_specs) == set(header) - {spec.label_column, spec.split_column}


def test_column_specs_types():
    specs = default_automl_spec().column_specs
    assert specs["tipo_credito"] == "categorical"
    assert specs["tiene_mora_bureau"] == "categorical"
    assert specs["capital_prestado"] == "numeric"


def test_budget_above_cap_is_rejected():
    spec = default_automl_spec().model_dump()
    spec["budget_milli_node_hours"] = 2000
    with pytest.raises(ValidationError):
        type(default_automl_spec())(**spec)


def test_column_cannot_be_both_kinds():
    spec = default_automl_spec().model_dump()
    spec["numeric"] = spec["numeric"] + ["tipo_credito"]
    with pytest.raises(ValidationError):
        type(default_automl_spec())(**spec)
