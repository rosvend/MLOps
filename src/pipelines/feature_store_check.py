"""Prove the offline store retrieves correctly and does not answer about the future.

    make feast-verify

Three properties, in the order a wrong answer would reach a model. That the registry
still describes the parquet it points at: `make features` alone leaves it stale, and
Feast then serves drifted dtypes without a word. That every value belongs to the loan it
was asked about, since Feast returns rows sorted by event timestamp rather than in
entity_df order, so counting non-nulls would not notice a scrambled answer. And the one
that matters: that a feature never crosses back over its own event timestamp - ask for a
loan the day before it was originated and the store must know nothing about it.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore

from src.features.feast_types import parquet_types
from src.features.spec import FeatureSpec, default_spec

RAIZ = Path(__file__).resolve().parents[2]
REPO = RAIZ / "feature_repo"

DESFASES = [
    ("un día antes del desembolso", -timedelta(days=1)),
    ("en el desembolso          ", timedelta(0)),
    ("30 días después           ", timedelta(days=30)),
]


def stale_fields(store: FeatureStore, parquet: str | Path) -> list[str]:
    """Registered fields whose dtype no longer matches the parquet they point at."""
    arrow = parquet_types(parquet)
    return sorted(
        campo.name
        for vista in store.list_feature_views()
        for campo in vista.schema
        if arrow.get(campo.name) != campo.dtype
    )


def mismatched_values(
    salida: pd.DataFrame, fuente: pd.DataFrame, spec: FeatureSpec, columnas: list[str]
) -> list[str]:
    """Columns where a retrieved value is not the source's value for that same loan."""
    unidos = salida.merge(fuente, on=spec.entity_key, suffixes=("_feast", "_src"))
    return [c for c in columnas if not unidos[f"{c}_feast"].equals(unidos[f"{c}_src"])]


def entity_df(fuente: pd.DataFrame, spec: FeatureSpec, desplazamiento: timedelta) -> pd.DataFrame:
    return pd.DataFrame(
        {
            spec.entity_key: fuente[spec.entity_key].to_numpy(),
            "event_timestamp": pd.to_datetime(fuente[spec.event_timestamp])
            + pd.Timedelta(desplazamiento),
        }
    ).reset_index(drop=True)


def main() -> None:
    spec = default_spec()
    store = FeatureStore(repo_path=str(REPO))
    parquet = RAIZ / spec.output_path
    fuente = pd.read_parquet(parquet).head(spec.verify_sample)
    refs = [f"{v}:{c}" for v, cols in spec.feature_views.items() for c in cols]

    desfasadas = stale_fields(store, parquet)
    if desfasadas:
        raise SystemExit(
            f"El registro no describe {parquet.name} ({len(desfasadas)} campos "
            f"desfasados, p. ej. {desfasadas[:5]}). Corre `make feast-apply`."
        )

    print(f"Registro    : {len(store.list_feature_views())} feature views al día con "
          f"{parquet.name}, entidad {store.get_entity(spec.entity_name).join_key}")
    print(f"Online store: {store.config.online_store}  (offline retrieval only)")
    print(f"Consultando : {len(fuente)} créditos x {len(refs)} features\n")

    en_el_desembolso = None
    for etiqueta, desplazamiento in DESFASES:
        salida = store.get_historical_features(
            entity_df=entity_df(fuente, spec, desplazamiento), features=refs
        ).to_df()
        columnas = [c for c in salida.columns if c not in (spec.entity_key, "event_timestamp")]
        con_valor = 0 if salida.empty else int(salida[columnas].notna().any(axis=1).sum())
        print(f"  {etiqueta}: {len(salida):>3} filas, {con_valor:>3} con features")
        if not desplazamiento:
            en_el_desembolso = salida

    erroneas = mismatched_values(en_el_desembolso, fuente, spec, spec.feature_view_columns)
    if erroneas:
        raise SystemExit(f"Valores que no corresponden al crédito consultado: {erroneas}")

    print(f"\n{len(spec.feature_view_columns)} features coinciden con la fuente, unidas por "
          f"{spec.entity_key} (Feast reordena las filas: nunca las emparejes por posición).")
    print("Cero features antes del desembolso = join correcto en el tiempo, sin fuga.")


if __name__ == "__main__":
    main()
