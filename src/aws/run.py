"""Submit the three candidates as SageMaker training jobs, wait, and write timings + a compare table."""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml

from src.aws.jobs import billable_cost, build_training_job, startup_overhead

CFG = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "aws_training" / "default.yaml").read_text())


def main(bucket: str, image: str, role_arn: str, out: str) -> None:
    sm = boto3.Session(profile_name=CFG["profile"], region_name=CFG["region"]).client("sagemaker")
    run = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    names, submitted = {}, time.time()
    for cand in CFG["candidates"]:
        job = f"mlops-loan-train-{cand}-{run}"
        sm.create_training_job(**build_training_job(
            cand, job, image, role_arn, f"s3://{bucket}/{CFG['runs_prefix']}/{run}",
            f"s3://{bucket}/{CFG['dataset_key']}", CFG["instance_type"], CFG["volume_gb"],
            CFG["max_runtime_seconds"], CFG["tags"], CFG["instance_count"],
        ))
        names[cand] = job
    rows = {}
    while len(rows) < len(names):
        time.sleep(15)
        for cand, job in names.items():
            if cand in rows:
                continue
            d = sm.describe_training_job(TrainingJobName=job)
            if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
                tr = [{"Status": t["Status"], "StartTime": t["StartTime"].timestamp(), "EndTime": t["EndTime"].timestamp()}
                      for t in d["SecondaryStatusTransitions"] if "EndTime" in t]
                rows[cand] = {
                    "job": job, "status": d["TrainingJobStatus"], "failure": d.get("FailureReason"),
                    "submit_to_done_s": d["TrainingEndTime"].timestamp() - d["CreationTime"].timestamp() if "TrainingEndTime" in d else None,
                    "last_modified_to_creation_s": d["LastModifiedTime"].timestamp() - d["CreationTime"].timestamp(),
                    "transitions": [{**t, "seconds": round(t["EndTime"] - t["StartTime"], 1)} for t in tr],
                    "startup_overhead_s": startup_overhead(tr),
                    "billable_s": d.get("BillableTimeInSeconds"),
                    "cost_usd": billable_cost(d.get("BillableTimeInSeconds", 0), CFG["price_per_hour"][CFG["instance_type"]]),
                    "output": d["ModelArtifacts"]["S3ModelArtifacts"] if "ModelArtifacts" in d else None,
                }
    rows["wall_clock_s"] = time.time() - submitted
    Path(out).write_text(json.dumps({"run": run, "jobs": rows}, indent=2, default=str))
    print(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:5])
