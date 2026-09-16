import pytest
from pydantic import ValidationError

from src.features.cleaning import TENDENCIAS
from src.features.derive import RANGO_EDAD_LABELS
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
    spec["points"]["rango_edad"].pop(RANGO_EDAD_LABELS[-1])

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
    assert set(spec["points"]["rango_edad"]) == set(RANGO_EDAD_LABELS)
    assert set(spec["points"]["tendencia"]) == set(TENDENCIAS)
