"""Build and upload the imputed training file.

    GOOGLE_CLOUD_QUOTA_PROJECT=mlops-roysandoval uv run --group gcp python -m src.automl.prepare
"""

import io
from pathlib import Path

import pandas as pd
from google.cloud import storage

from src.automl.impute import impute
from src.automl.spec import AUTOML_PATH, AutomlSpec, load_automl_spec


def prepare(spec: AutomlSpec, df: pd.DataFrame) -> pd.DataFrame:
    i = spec.imputation
    return impute(df, spec.numeric, spec.categorical, spec.split_column, i.train_value, i.missing_label)


def main(config: Path = AUTOML_PATH.with_name("imputed.yaml")) -> None:
    spec = load_automl_spec(config)
    client = storage.Client(project=spec.project)

    def blob(uri):
        bucket, _, path = uri.removeprefix("gs://").partition("/")
        return client.bucket(bucket).blob(path)

    df = pd.read_csv(io.BytesIO(blob(spec.imputation.source_uri).download_as_bytes()))
    out = prepare(spec, df)
    assert not out.isna().any().any()
    blob(spec.dataset_uri).upload_from_string(out.to_csv(index=False), content_type="text/csv")
    print(f"uploaded {len(out)} rows to {spec.dataset_uri}")


if __name__ == "__main__":
    main()
