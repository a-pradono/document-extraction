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


class TestCalibrationRefusesAPartialFit:
    """Regression: a rate-limited run fitted the confidence model on 8 of 18 calibration
    documents, wrote the model and the threshold, and exited 0. Every downstream run
    would then have used a threshold read off whichever documents happened to run before
    the quota ran out."""

    def _run(self, tmp_path, monkeypatch, failures, allow_partial=False):
        import pytest

        from eval import run as eval_run
        from extractor.confidence import FieldSignals
        from extractor.schema import Receipt

        ids = [f"cord_{i:04d}" for i in range(1, 11)]
        monkeypatch.setattr(eval_run, "_doc_ids", lambda _: ids)
        monkeypatch.setattr(eval_run, "_find_doc", lambda d: tmp_path / f"{d}.png")
        gt = tmp_path / "gt.json"
        gt.write_text('{"total": "10", "subtotal": "10"}')
        monkeypatch.setattr(eval_run, "_find_gt", lambda d: gt)
        monkeypatch.setattr(eval_run, "MODEL_PATH", tmp_path / "model.json")
        monkeypatch.setattr(eval_run, "OUT_ROOT", tmp_path / "out")

        def extract(path, *a, **k):
            if path.stem in failures:
                raise RuntimeError("RateLimitError: 429 quota exceeded")
            return eval_run.Extraction(
                document=Receipt(total="10", subtotal="10"), latency_s=0.1,
                signals={"total": FieldSignals(1.0, 1.0, 1.0, 0.9),
                         "subtotal": FieldSignals(1.0, 1.0, 1.0, 0.9)})

        monkeypatch.setattr(eval_run, "extract_one", extract)
        return eval_run, pytest

    def test_it_refuses_and_writes_nothing_when_most_documents_failed(self, tmp_path, monkeypatch):
        eval_run, pytest = self._run(tmp_path, monkeypatch,
                                     failures={f"cord_{i:04d}" for i in range(3, 11)})
        with pytest.raises(SystemExit) as exc:
            eval_run.calibrate(0.95)
        assert "REFUSING" in str(exc.value) and "2 of 10" in str(exc.value)
        assert not (tmp_path / "model.json").exists()          # nothing downstream can pick it up
        assert not (tmp_path / "out" / "calibration.json").exists()

    def test_allow_partial_fits_but_marks_the_report(self, tmp_path, monkeypatch):
        import json

        eval_run, _ = self._run(tmp_path, monkeypatch,
                                failures={f"cord_{i:04d}" for i in range(3, 11)})
        out = eval_run.calibrate(0.95, allow_partial=True)
        report = json.loads(out.read_text())
        assert report["partial"] is True
        assert report["n_documents"] == 2 and report["n_documents_requested"] == 10
        assert len(report["failed_documents"]) == 8

    def test_a_full_run_is_not_marked_partial(self, tmp_path, monkeypatch):
        import json

        eval_run, _ = self._run(tmp_path, monkeypatch, failures=set())
        report = json.loads(eval_run.calibrate(0.95).read_text())
        assert report["partial"] is False and report["document_coverage"] == 1.0
        assert report["n_documents"] == 10
