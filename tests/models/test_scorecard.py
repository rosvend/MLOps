import pytest
from pydantic import ValidationError

from src.features.spec import default_spec
from src.features.spec import default_spec
from src.models.scorecard import Scorecard, default_scorecard


@pytest.fixture
def spec() -> dict:
    return default_scorecard().model_dump()


def test_the_shipped_scorecard_loads(spec):
    assert Scorecard(**spec).threshold == 4


def test_the_weights_are_reproducible_from_the_documented_formula(spec):
    """points = round((band_rate / base_rate - 1) * lift_multiplier) must be recorded."""
    assert spec["base_rate"] == pytest.approx(0.0475)
    assert spec["lift_multiplier"] == 4


def test_a_renamed_age_band_is_rejected(spec):
    spec["points"]["rango_edad"] = {
        ("66-100" if b == "66+" else b): v for b, v in spec["points"]["rango_edad"].items()
    }

    with pytest.raises(ValidationError, match="rango_edad"):
        Scorecard(**spec)


def test_an_age_band_without_a_weight_is_rejected(spec):
    spec["points"]["rango_edad"].pop(default_spec().age_bands.labels[-1])

    with pytest.raises(ValidationError, match="rango_edad"):
        Scorecard(**spec)


def test_a_renamed_income_trend_is_rejected(spec):
    spec["points"]["tendencia"] = {t.upper(): v for t, v in spec["points"]["tendencia"].items()}

    with pytest.raises(ValidationError, match="tendencia"):
        Scorecard(**spec)


def test_the_income_gap_weights_must_cover_every_cut_point(spec):
    spec["points"]["brecha_ingreso"] = spec["points"]["brecha_ingreso"][:-1]

    with pytest.raises(ValidationError, match="brecha_ingreso"):
        Scorecard(**spec)


def test_a_typo_in_a_key_fails_loudly_instead_of_keeping_a_default(spec):
    spec["cut_points"]["edad_joven_"] = 36

    with pytest.raises(ValidationError):
        Scorecard(**spec)


def test_the_shipped_vocabularies_match_the_pipeline(spec):
    assert set(spec["points"]["rango_edad"]) == set(default_spec().age_bands.labels)
    assert set(spec["points"]["tendencia"]) == set(default_spec().vocabularies.tendencias)


@pytest.mark.parametrize(
    "corte,invertido",
    [
        ("puntaje_bureau_terciles", [813, 770]),
        ("huella_bandas", [6, 3]),
        ("brecha_ingreso_cuartiles", [3.594, 1.081, 1.807]),
    ],
)
def test_unsorted_cut_points_are_rejected(spec, corte, invertido):
    """An inverted override would mis-band every applicant instead of failing."""
    spec["cut_points"][corte] = invertido

    with pytest.raises(ValidationError, match=corte):
        Scorecard(**spec)


def test_a_renamed_inquiry_band_is_rejected(spec):
    spec["points"]["huella"] = {"0-2": -1, "3-6": 0, "7+": 2}

    with pytest.raises(ValidationError, match="huella"):
        Scorecard(**spec)


def test_a_renamed_bureau_band_is_rejected(spec):
    spec["points"]["puntaje_bureau"] = {"low": 2, "mid": -1, "high": -2}

    with pytest.raises(ValidationError, match="puntaje_bureau"):
        Scorecard(**spec)
