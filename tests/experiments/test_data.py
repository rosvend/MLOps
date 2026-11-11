import pandas as pd

from experiments.signal.data import development_only


def test_test_months_are_dropped():
    meses = ["2024-01"] * 4 + ["2024-02"] * 4 + ["2024-03"] * 2
    df = pd.DataFrame({"mes_prestamo": meses, "Pago_atiempo": [True] * 10, "x": range(10)})
    dev = development_only(df, 0.8)
    assert set(dev.vintage) == {"2024-01"}
    assert len(dev.X) == 4 and "mes_prestamo" not in dev.X and (dev.y == 0).all()
