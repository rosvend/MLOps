import pandas as pd

from src.models.heuristic import RULES, explain, predict, score, score_frame


def test_the_score_is_the_sum_of_every_rule(neutral_record):
    record = neutral_record()

    assert score(record) == sum(rule(record) for rule in RULES)


def test_explain_reports_one_entry_per_rule(neutral_record):
    breakdown = explain(neutral_record())

    assert len(breakdown) == len(RULES)
    assert sum(breakdown.values()) == score(neutral_record())


def test_a_risky_profile_outranks_a_safe_one(neutral_record):
    risky = neutral_record(
        puntaje_datacredito=690,
        huella_consulta=11,
        rango_edad="18-25",
        edad_cliente=22,
        tipo_laboral="Independiente",
        ratio_ingreso_declarado_bureau=8.0,
        tendencia_ingresos="Decreciente",
        capital_prestado=4_000_000,
        plazo_meses=24,
    )
    safe = neutral_record(
        puntaje_datacredito=900,
        huella_consulta=1,
        rango_edad="46-55",
        edad_cliente=50,
        ratio_ingreso_declarado_bureau=0.6,
        tendencia_ingresos="Creciente",
    )

    assert score(risky) > score(safe)


def test_an_empty_record_scores_without_raising():
    assert isinstance(score({}), int)


def test_predict_flags_scores_at_or_above_the_threshold(neutral_record):
    record = neutral_record()
    points = score(record)

    assert predict(record, threshold=points)
    assert not predict(record, threshold=points + 1)


def test_score_frame_scores_every_row(sample_source):
    from src.pipelines.prepare import prepare

    scores = score_frame(prepare(sample_source))

    assert len(scores) == 14
    assert scores.dtype.kind == "i"


def test_a_row_scores_the_same_alone_as_inside_a_frame(sample_source):
    from src.pipelines.prepare import prepare

    prepared = prepare(sample_source)

    assert score(prepared.iloc[0]) == score_frame(prepared).iloc[0]


def test_scoring_one_row_does_not_depend_on_the_others(sample_source):
    from src.pipelines.prepare import prepare

    prepared = prepare(sample_source)
    alone = score_frame(prepared.head(1)).iloc[0]

    assert alone == score_frame(prepared).iloc[0]
