"""Feature views over the stage-5 feature table.

Both the column groups and the dtypes come from config/features/default.yaml and from the
parquet itself, so the registry cannot drift away from the pipeline that produced it.
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feast import FeatureView, FileSource, Field  # noqa: E402

from src.features.feast_types import parquet_types  # noqa: E402
from src.features.spec import default_spec  # noqa: E402

from entities import cliente  # noqa: E402  (feast puts the repo dir on the path)

_RAIZ = Path(__file__).resolve().parents[1]
_SPEC = default_spec()
_PARQUET = _RAIZ / _SPEC.output_path

# Read at import: every Feast entry point needs the table to exist, and parquet_types
# says so by name rather than letting a bare FileNotFoundError surface from pyarrow.
_TIPOS = parquet_types(_PARQUET)

fuente = FileSource(
    name="credito_batch",
    path=str(_PARQUET),
    timestamp_field=_SPEC.event_timestamp,
    description="Row-independent features from make features. One row per credit application.",
)


def _vista(nombre: str, columnas: list[str]) -> FeatureView:
    return FeatureView(
        name=nombre,
        entities=[cliente],
        # Unlimited: a non-zero TTL would silently drop loans older than the window from
        # every point-in-time join, which on a 15-month book is most of them.
        ttl=timedelta(days=0),
        schema=[Field(name=c, dtype=_TIPOS[c]) for c in columnas],
        source=fuente,
        online=False,
    )


credito_solicitud = _vista("credito_solicitud", _SPEC.feature_views["credito_solicitud"])
credito_bureau = _vista("credito_bureau", _SPEC.feature_views["credito_bureau"])
credito_derivadas = _vista("credito_derivadas", _SPEC.feature_views["credito_derivadas"])
credito_faltantes = _vista("credito_faltantes", _SPEC.feature_views["credito_faltantes"])
