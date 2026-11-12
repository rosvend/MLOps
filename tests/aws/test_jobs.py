import pytest

from src.aws.jobs import billable_cost, build_training_job, startup_overhead

TR = [
    {"Status": "Starting", "StartTime": 0.0, "EndTime": 40.0},
    {"Status": "Downloading", "StartTime": 40.0, "EndTime": 50.0},
    {"Status": "Training", "StartTime": 50.0, "EndTime": 90.0},
    {"Status": "Uploading", "StartTime": 90.0, "EndTime": 95.0},
]


def test_job_spec_has_entrypoint_data_and_tags():
    spec = build_training_job(
        "logistic", "job-1", "img:latest", "arn:role", "s3://b/runs/r1", "s3://b/datasets/x.csv",
        instance_type="ml.m5.large", volume_gb=5, max_runtime=600, tags={"project": "p"},
    )
    assert spec["TrainingJobName"] == "job-1"
    assert spec["AlgorithmSpecification"]["ContainerEntrypoint"][-1] == "src.aws.train"
    assert spec["HyperParameters"] == {"candidate": "logistic"}
    assert spec["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == "s3://b/datasets/x.csv"
    assert spec["OutputDataConfig"]["S3OutputPath"] == "s3://b/runs/r1"
    assert spec["Tags"] == [{"Key": "project", "Value": "p"}]


def test_startup_overhead_is_everything_before_training():
    assert startup_overhead(TR) == 50.0


def test_billable_cost():
    assert billable_cost(36, 0.115, 1) == pytest.approx(36 / 3600 * 0.115)
