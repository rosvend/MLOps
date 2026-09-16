"""Prove the offline store retrieves correctly and does not answer about the future.

    make feast-verify

Retrieval alone is not the interesting property. The one worth checking is that a
feature never crosses back over its own event timestamp: ask for a loan the day before
it was originated and the store must know nothing about it.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore

from src.features.spec import default_spec

RAIZ = Path(__file__).resolve().parents[2]
REPO = RAIZ / "feature_repo"
MUESTRA = 200


def _entity_df(fuente: pd.DataFrame, spec, desplazamiento: timedelta) -> pd.DataFrame:
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
    fuente = pd.read_parquet(RAIZ / spec.output_path).head(MUESTRA)
    refs = [f"{v}:{c}" for v, cols in spec.feature_views.items() for c in cols]

    print(f"Registro   : {len(store.list_feature_views())} feature views, entidad "
          f"{store.get_entity('cliente').join_key}")
    print(f"Online store: {store.config.online_store}  (offline retrieval only)")
    print(f"Consultando : {len(fuente)} créditos x {len(refs)} features\n")

    for etiqueta, desplazamiento in [
        ("un día antes del desembolso", -timedelta(days=1)),
        ("en el desembolso          ", timedelta(0)),
        ("30 días después           ", timedelta(days=30)),
    ]:
        salida = store.get_historical_features(
            entity_df=_entity_df(fuente, spec, desplazamiento), features=refs
        ).to_df()
        columnas = [c for c in salida.columns if c not in (spec.entity_key, "event_timestamp")]
        con_valor = 0 if salida.empty else int(salida[columnas].notna().any(axis=1).sum())
        print(f"  {etiqueta}: {len(salida):>3} filas, {con_valor:>3} con features")

    print("\nCero features antes del desembolso = join correcto en el tiempo, sin fuga.")


if __name__ == "__main__":
    main()
