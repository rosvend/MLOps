"""The Vertex AI pipeline (KFP v2): one training step per candidate, then a comparison step.

Each step runs in the training image and only calls a function from src/gcp/tasks.py.
"""

from kfp import compiler, dsl

from src.gcp.spec import PipelineSpec, default_pipeline_spec
from src.models.training_spec import default_training_spec


def build_pipeline(spec: PipelineSpec):
    image = spec.image
    names = list(spec.candidates)
    metric = default_training_spec().selection.primary_metric
    reference = spec.reference

    @dsl.component(base_image=image, install_kfp_package=False)
    def train_candidate(name: str, run_dir: str):
        from src.gcp.tasks import run_candidate

        run_candidate(name, run_dir)

    @dsl.component(base_image=image, install_kfp_package=False)
    def compare_candidates(run_dir: str, names: list, reference: dict, metric: str):
        from src.gcp.tasks import run_compare

        run_compare(run_dir, names, reference, metric)

    @dsl.pipeline(name=spec.pipeline_name, pipeline_root=f"gs://{spec.bucket}/pipeline-root")
    def pipeline(run_dir: str):
        trained = []
        for name in names:
            task = train_candidate(name=name, run_dir=run_dir).set_display_name(f"train-{name}")
            task.set_cpu_limit(spec.cpu).set_memory_limit(spec.memory)
            trained.append(task)
        compare = compare_candidates(run_dir=run_dir, names=names, reference=reference, metric=metric)
        compare.set_cpu_limit(spec.cpu).set_memory_limit(spec.memory)
        compare.after(*trained)

    return pipeline


def compile_pipeline(path: str, spec: PipelineSpec | None = None) -> str:
    spec = spec or default_pipeline_spec()
    compiler.Compiler().compile(build_pipeline(spec), package_path=path)
    return path
