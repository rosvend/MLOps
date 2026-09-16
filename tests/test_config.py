import pytest
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pydantic import ValidationError

from src.config import CONFIG_DIR, Config, load_config


def test_the_shipped_config_composes():
    config = load_config()

    assert config.data_source.type == "csv"
    assert config.model.threshold == 4


def test_a_cli_override_reaches_the_model():
    assert load_config(["model.threshold=7"]).model.threshold == 7


def test_config_loads_inside_an_existing_hydra_session():
    """@hydra.main leaves GlobalHydra initialized; initialize_config_dir would raise."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        assert GlobalHydra.instance().is_initialized()

        assert load_config().model.threshold == 4


def test_hydras_own_keys_do_not_leak_into_the_typed_config():
    assert set(Config.model_fields) == {"data_source", "features", "model", "training", "serving", "monitoring"}


def test_an_unknown_top_level_key_is_rejected():
    with pytest.raises(ValidationError):
        Config(**{**load_config().model_dump(), "extra": 1})
