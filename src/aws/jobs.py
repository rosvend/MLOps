"""Pure helpers: the CreateTrainingJob request and the timing/cost arithmetic."""


def build_training_job(
    candidate: str, job_name: str, image: str, role_arn: str, output_uri: str, data_uri: str,
    instance_type: str, volume_gb: int, max_runtime: int, tags: dict[str, str], instance_count: int = 1,
) -> dict:
    return {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "-m", "src.aws.train"],
        },
        "HyperParameters": {"candidate": candidate},
        "InputDataConfig": [{
            "ChannelName": "train",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": data_uri, "S3DataDistributionType": "FullyReplicated"}},
        }],
        "OutputDataConfig": {"S3OutputPath": output_uri},
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": instance_count, "VolumeSizeInGB": volume_gb},
        "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime},
        "Tags": [{"Key": k, "Value": v} for k, v in tags.items()],
    }


def startup_overhead(transitions: list[dict]) -> float:
    """Seconds spent before the Training phase begins (Starting + Downloading)."""
    return sum(t["EndTime"] - t["StartTime"] for t in transitions if t["Status"] in ("Starting", "Downloading"))


def billable_cost(billable_seconds: float, price_per_hour: float, instances: int = 1) -> float:
    return billable_seconds / 3600 * price_per_hour * instances
