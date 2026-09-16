import pytest

from src.features.spec import default_spec
from src.pipelines.prepare import prepare_features, prepare_labelled
from tests.pipelines.test_prepare import InMemorySource


def test_an_application_with_no_outcome_can_still_be_prepared(raw):
    sin_label = raw.drop(columns=[default_spec().target])

    prepared = prepare_features(InMemorySource(sin_label))

    assert len(prepared) == len(raw)
    assert default_spec().target not in prepared.columns


def test_preparing_for_training_still_requires_the_outcome(raw):
    with pytest.raises(Exception):
        prepare_labelled(InMemorySource(raw.drop(columns=[default_spec().target])))


def test_both_paths_agree_on_every_shared_column(sample_source):
    solo_features = prepare_features(sample_source)
    con_label = prepare_labelled(sample_source)

    compartidas = list(solo_features.columns)
    assert solo_features[compartidas].equals(con_label[compartidas])
