"""Tests for aggregate confidence (Phase 4). Run: pytest eval/tests -q"""

from __future__ import annotations

import random
from decimal import Decimal

from extractor.confidence import (
    ConfidenceModel,
    FieldSignals,
    group_of,
    pick_threshold,
    risk_coverage_curve,
    validate_arithmetic,
)
from extractor.schema import LineItem, Receipt
from eval.metrics import auc, ece


def test_validation_passes_when_arithmetic_closes():
    r = Receipt(subtotal=Decimal("90.00"), tax=Decimal("10.00"), total=Decimal("100.00"))
    v = validate_arithmetic(r)
    assert v["total"] == 1.0 and v["subtotal"] == 1.0


def test_validation_fails_when_arithmetic_breaks():
    r = Receipt(subtotal=Decimal("90.00"), tax=Decimal("10.00"), total=Decimal("150.00"))
    v = validate_arithmetic(r)
    assert v["total"] == 0.0


def test_validation_neutral_when_not_checkable():
    # total alone -> no identity to check, so the field is simply absent from the map
    assert "total" not in validate_arithmetic(Receipt(total=Decimal("50.00")))


def test_line_items_sum_checked_against_subtotal():
    r = Receipt(
        subtotal=Decimal("15.00"),
        line_items=[
            LineItem(description="a", line_total=Decimal("10.00")),
            LineItem(description="b", line_total=Decimal("5.00")),
        ],
    )
    assert validate_arithmetic(r)["line_items"] == 1.0


def test_currency_validation_rejects_non_iso():
    assert validate_arithmetic(Receipt(currency="DOLLARS"))["currency"] == 0.0
    assert validate_arithmetic(Receipt(currency="IDR"))["currency"] == 1.0


def test_field_groups():
    assert group_of("total") == "money"
    assert group_of("merchant_name") == "string"
    assert group_of("date") == "date"


def test_unfitted_model_scores_in_range_and_ranks_sensibly():
    m = ConfidenceModel()
    strong = m.score_field("total", FieldSignals(1.0, 1.0, 1.0, 1.0))
    weak = m.score_field("total", FieldSignals(0.0, 0.0, 0.0, 0.0))
    assert 0.0 <= weak < strong <= 1.0


def test_self_report_alone_cannot_dominate():
    """A field with ONLY high self-report must not reach high confidence -- the
    whole point of down-weighting the weakest signal."""
    m = ConfidenceModel()
    bluffing = m.score_field("total", FieldSignals(agreement=0.0, grounded=0.0,
                                                   validated=0.0, self_report=1.0))
    assert bluffing < 0.3


def test_fit_learns_to_favor_informative_signal():
    """Generate data where `validated` perfectly predicts correctness and
    `self_report` is pure noise; the fitted weights should reflect that."""
    rng = random.Random(0)
    samples = []
    for _ in range(200):
        ok = rng.random() < 0.5
        samples.append((
            "total",
            FieldSignals(
                agreement=1.0 if ok else 0.0,
                grounded=1.0 if ok else 0.0,
                validated=1.0 if ok else 0.0,
                self_report=rng.random(),  # noise
            ),
            ok,
        ))
    m = ConfidenceModel()
    report = m.fit(samples)
    assert report["money"]["fitted"] is True
    w = report["money"]["weights"]
    assert w["validated"] > w["self_report"]


def test_fit_skips_groups_with_too_few_samples():
    m = ConfidenceModel()
    report = m.fit([("date", FieldSignals(), True)] * 3)
    assert report["date"]["fitted"] is False


def test_fitted_model_improves_calibration():
    rng = random.Random(1)
    samples, pairs_before = [], []
    m_default = ConfidenceModel()
    for _ in range(200):
        ok = rng.random() < 0.4
        sig = FieldSignals(
            agreement=1.0 if ok else 0.0,
            grounded=1.0 if ok else 0.0,
            validated=1.0 if ok else 0.0,
            self_report=0.95,  # systematically overconfident
        )
        samples.append(("total", sig, ok))
        pairs_before.append((m_default.score_field("total", sig), ok))

    m_fitted = ConfidenceModel()
    m_fitted.fit(samples)
    pairs_after = [(m_fitted.score_field("total", s), ok) for _, s, ok in samples]

    assert ece(pairs_after) <= ece(pairs_before)
    assert auc(pairs_after) >= 0.9  # signals are informative, so ranking should hold


def test_risk_coverage_curve_is_monotone_in_coverage():
    pairs = [(0.9, True), (0.8, True), (0.4, False), (0.2, False)]
    curve = risk_coverage_curve(pairs)
    coverages = [p["coverage"] for p in curve]
    assert coverages == sorted(coverages, reverse=True)


def test_pick_threshold_meets_target_precision():
    pairs = [(0.95, True), (0.9, True), (0.85, True), (0.3, False), (0.1, False)]
    chosen = pick_threshold(pairs, target_precision=1.0)
    assert chosen["met_target"] is True
    assert chosen["achieved"]["precision"] == 1.0


def test_pick_threshold_reports_failure_honestly():
    pairs = [(0.9, False), (0.8, False), (0.7, True)]
    chosen = pick_threshold(pairs, target_precision=0.99)
    assert chosen["met_target"] is False


def test_save_load_roundtrip(tmp_path):
    m = ConfidenceModel()
    m.fit([("total", FieldSignals(1.0, 1.0, 1.0, 0.5), True)] * 20)
    p = tmp_path / "conf.json"
    m.save(p)
    loaded = ConfidenceModel.load(p)
    sig = FieldSignals(1.0, 1.0, 1.0, 0.5)
    assert abs(loaded.score_field("total", sig) - m.score_field("total", sig)) < 1e-9
