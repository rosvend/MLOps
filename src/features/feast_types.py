"""The one translation between the parquet's Arrow types and Feast's.

It lives in src/ rather than feature_repo/ so the feature views and the verification
script read the same table: the registry is built from it and checked against it, and a
type outside it raises instead of being guessed at.
"""

from pathlib import Path

import pyarrow.parquet as pq
from feast.types import Bool, FeastType, Float64, Int64, String, UnixTimestamp

ARROW_A_FEAST: dict[str, FeastType] = {
    "int64": Int64,
    "double": Float64,
    "bool": Bool,
    "string": String,
    "large_string": String,
}


def feast_type(arrow: str) -> FeastType:
    """Any zone is accepted for the timestamp; everything else must be in the table."""
    if arrow.startswith("timestamp["):
        return UnixTimestamp
    if arrow not in ARROW_A_FEAST:
        raise TypeError(f"tipo Arrow sin equivalente en Feast: {arrow}")
    return ARROW_A_FEAST[arrow]


def parquet_types(ruta: str | Path) -> dict[str, FeastType]:
    """Feast dtype per column of the feature table, read from the parquet itself."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"falta la tabla de features en {ruta}; corre `make features`")
    return {campo.name: feast_type(str(campo.type)) for campo in pq.read_schema(ruta)}
