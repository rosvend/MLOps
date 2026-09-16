"""The spec is the only place config can be wrong; every drift must fail at load."""

import pytest
import yaml
from pydantic import ValidationError

from src.features.spec import SPEC_PATH, FeatureSpec, default_spec


@pytest.fixture
def raw():
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def _variando(**cambios) -> FeatureSpec:
    return FeatureSpec(**{**default_spec().model_dump(), **cambios})


def test_the_shipped_config_loads():
    assert default_spec().entity_key == "cliente_id"


def test_the_shipped_config_is_valid(raw):
    assert FeatureSpec(**raw).entity_key == default_spec().entity_key


# --- column roles and bands ---------------------------------------------------


def test_age_bands_must_cover_the_accepted_age_range():
    """An age inside bounds.edad but outside the bands gets rango_edad = NaN, and the
    scorecard then pays it the "age unknown" points although the age is known."""
    with pytest.raises(ValidationError, match="age_bands"):
        _variando(bounds={**default_spec().bounds.model_dump(), "edad": [18, 110]})


def test_age_bands_must_reach_below_the_youngest_accepted_age():
    with pytest.raises(ValidationError, match="age_bands"):
        _variando(age_bands={"bins": [25, 45, 100], "labels": ["26-45", "46+"]})


def test_a_column_cannot_hold_two_roles():
    with pytest.raises(ValidationError, match="dos roles"):
        _variando(withheld=["puntaje", "mes_prestamo", "cliente_id"])


def test_a_log_column_must_be_clipped_first():
    spec = default_spec()
    with pytest.raises(ValidationError, match="log_scale"):
        _variando(
            engineering={
                **spec.engineering.model_dump(),
                "log_scale": [*spec.engineering.log_scale, "plazo_meses"],
            }
        )


# --- feature store settings ---------------------------------------------------


def test_the_entity_name_is_configured(raw):
    """feast apply and the verification script must agree without repeating the name."""
    assert default_spec().entity_name == raw["entity_name"]


def test_an_unknown_timezone_is_rejected(raw):
    """A typo used to survive validation and blow up mid-pipeline in tz_localize."""
    raw["event_timestamp_timezone"] = "Not/AZone"

    with pytest.raises(ValueError, match="event_timestamp_timezone"):
        FeatureSpec(**raw)


def test_a_real_timezone_is_accepted(raw):
    raw["event_timestamp_timezone"] = "America/Bogota"

    assert FeatureSpec(**raw).event_timestamp_timezone == "America/Bogota"


@pytest.mark.parametrize("tamano", [0, -1])
def test_an_empty_verification_sample_is_rejected(raw, tamano):
    raw["verify_sample"] = tamano

    with pytest.raises(ValueError, match="verify_sample"):
        FeatureSpec(**raw)
