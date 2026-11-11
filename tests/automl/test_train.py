from unittest.mock import MagicMock

from src.automl.spec import default_automl_spec
from src.automl.train import submit


def test_submit_wires_spec_into_sdk_without_leaking_test_or_exceeding_budget():
    sdk = MagicMock()
    spec = default_automl_spec()
    submit(spec, sdk)

    sdk.init.assert_called_once_with(project=spec.project, location=spec.region)
    sdk.TabularDataset.create.assert_called_once()
    assert sdk.TabularDataset.create.call_args.kwargs["gcs_source"] == spec.dataset_uri

    init = sdk.AutoMLTabularTrainingJob.call_args.kwargs
    assert init["optimization_prediction_type"] == "classification"
    assert init["optimization_objective"] == "maximize-au-prc"
    assert init["column_specs"] == spec.column_specs

    run = sdk.AutoMLTabularTrainingJob.return_value.run.call_args.kwargs
    assert run["target_column"] == "defaulted"
    assert run["predefined_split_column_name"] == "split"
    assert run["budget_milli_node_hours"] == 1000
    assert run["disable_early_stopping"] is False
    assert run["sync"] is False
