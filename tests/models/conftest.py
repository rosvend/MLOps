import pytest


@pytest.fixture
def neutral_record():
    """A mid-band applicant, so each test can move one variable at a time."""

    def build(**overrides):
        record = {
            "puntaje_datacredito": 790,
            "huella_consulta": 5,
            "rango_edad": "56-65",
            "edad_cliente": 60,
            "tipo_laboral": "Empleado",
            "ratio_ingreso_declarado_bureau": 2.5,
            "tendencia_ingresos": "Estable",
            "capital_prestado": 1_000_000,
            "plazo_meses": 6,
            "promedio_ingresos_datacredito": 2_000_000,
        }
        return record | overrides

    return build
