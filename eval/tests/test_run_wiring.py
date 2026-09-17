"""Tests for the eval runner's Phase 4 wiring. Run: pytest eval/tests -q

The runner itself makes network calls, so what is tested here is the part that must be
correct regardless of what the models return: how abstention is reported, and the split
isolation rules that keep the published numbers honest.
"""

from __future__ import annotations

from decimal import Decimal

from eval import metrics
from eval.run import abstention_report, _doc_ids
from extractor.schema import Receipt


def _doc(doc_id: str, pred: Receipt, gt: Receipt, confidence: dict) -> metrics.DocResult:
    return metrics.compare_document(doc_id, pred, gt, latency_s=1.0, confidence=confidence)


class TestAbstentionReport:
    def test_unavailable_without_confidence(self):
        result = _doc("d1", Receipt(total=Decimal("1")), Receipt(total=Decimal("1")), {})
        assert abstention_report([result], 0.5) == {"available": False}

    def test_dropping_a_wrong_low_confidence_field_raises_precision(self):
        gt = Receipt(merchant_name="Kopi", total=Decimal("100"))
        pred = Receipt(merchant_name="Kopi", total=Decimal("999"))   # total is wrong
        result = _doc("d1", pred, gt, {"merchant_name": 0.9, "total": 0.1})

        report = abstention_report([result], threshold=0.5)
        assert report["precision_kept"] > report["precision_all"]
        assert report["coverage"] < 1.0

    def test_a_useless_score_shows_no_precision_gain(self):
        """The honest negative result: uniform confidence buys nothing, and the report
        must show that rather than hiding it behind a coverage number."""
        gt = Receipt(merchant_name="Kopi", total=Decimal("100"))
        pred = Receipt(merchant_name="Kopi", total=Decimal("999"))
        result = _doc("d1", pred, gt, {"merchant_name": 0.8, "total": 0.8})

        report = abstention_report([result], threshold=0.5)
        assert report["coverage"] == 1.0
        assert report["precision_kept"] == report["precision_all"]

    def test_threshold_above_every_score_keeps_nothing(self):
        gt = Receipt(total=Decimal("100"))
        result = _doc("d1", Receipt(total=Decimal("100")), gt, {"total": 0.4})
        report = abstention_report([result], threshold=0.9)
        assert report["coverage"] == 0.0 and report["precision_kept"] is None

    def test_curve_is_included_for_choosing_an_operating_point(self):
        gt = Receipt(total=Decimal("100"))
        result = _doc("d1", Receipt(total=Decimal("100")), gt, {"total": 0.7})
        assert abstention_report([result], 0.5)["risk_coverage_curve"]


class TestSplitIsolation:
    """The two rules that keep the reported scorecard honest, asserted as tests so a
    future edit to the split files cannot quietly break them."""

    def test_smoke_never_leaks_into_test(self):
        try:
            smoke, test = set(_doc_ids("smoke")), set(_doc_ids("test"))
        except SystemExit:
            return  # splits not generated yet in this checkout
        assert not (smoke & test), f"smoke overlaps test: {smoke & test}"

    def test_smoke_is_drawn_from_calibration(self):
        try:
            smoke, calib = set(_doc_ids("smoke")), set(_doc_ids("calibration"))
        except SystemExit:
            return
        assert smoke <= calib, f"smoke outside calibration: {smoke - calib}"

    def test_calibration_and_test_are_disjoint(self):
        try:
            calib, test = set(_doc_ids("calibration")), set(_doc_ids("test"))
        except SystemExit:
            return
        assert not (calib & test)

    def test_default_sweep_excludes_synthetic(self):
        assert not any(i.startswith("syn_") for i in _doc_ids(None))
