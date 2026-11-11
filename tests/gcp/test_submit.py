from src.gcp.spec import default_pipeline_spec
from src.gcp.submit import parse_args, submit_job


class FakeJob:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.submitted_with = None
        self.waited = False

    def submit(self, **kwargs):
        self.submitted_with = kwargs

    def wait(self):
        self.waited = True


class FakeVertex:
    def __init__(self):
        self.jobs = []

    def PipelineJob(self, **kwargs):
        job = FakeJob(**kwargs)
        self.jobs.append(job)
        return job


def test_job_is_submitted_as_the_pipeline_service_account_and_awaited():
    vertex, spec = FakeVertex(), default_pipeline_spec()
    submit_job(vertex, spec, "t.yaml", "run-1", "gs://b/r/run-1", "abc")
    job = vertex.jobs[0]
    assert job.submitted_with == {"experiment": spec.experiment, "service_account": spec.service_account}
    assert job.kwargs["parameter_values"] == {"run_dir": "gs://b/r/run-1", "git_sha": "abc"}
    assert job.waited


def test_image_tag_is_optional():
    assert parse_args([]).image_tag is None
    assert parse_args(["--image-tag", "abc"]).image_tag == "abc"
