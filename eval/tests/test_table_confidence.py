"""Confidence and abstention on the line-item table. Run: pytest eval/tests -q"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from api.schemas import from_result
from eval import metrics
from extractor.confidence import ConfidenceModel, group_of, validate_arithmetic
from extractor.pipeline import PipelineResult, build_signals, table_agreement, table_grounding
from extractor.schema import LineItem, Receipt

TEXT = "KOPI\nEs Kopi Susu 2 18.000 36.000\nCroissant 1 12.000 12.000\nSUBTOTAL 48.000\nTOTAL 48.000"
ROWS = [LineItem(description="Es Kopi Susu", quantity="2", unit_price="18000", line_total="36000"),
        LineItem(description="Croissant", quantity="1", unit_price="12000", line_total="12000")]


class TestValidation:
    def test_rows_that_multiply_and_sum_pass(self):
        assert validate_arithmetic(Receipt(subtotal="48000", line_items=ROWS))["line_items"] == 1.0

    def test_row_arithmetic_is_checked(self):
        """Regression: the docstring promised qty x unit_price == line_total, but the code
        never checked it."""
        bad = [ROWS[0].model_copy(update={"line_total": Decimal("99000")}), ROWS[1]]
        assert validate_arithmetic(Receipt(line_items=bad))["line_items"] == 0.0

    def test_rows_that_multiply_but_do_not_sum_fail(self):
        """A dropped row: each surviving row is internally consistent, the table isn't."""
        assert validate_arithmetic(Receipt(subtotal="48000", line_items=ROWS[:1]))["line_items"] == 0.0

    def test_partial_rows_are_not_checkable_per_row(self):
        partial = [LineItem(description="Kopi", line_total="36000")]
        assert "line_items" not in validate_arithmetic(Receipt(line_items=partial))


class TestTableSignals:
    def test_grounding_is_the_fraction_of_readable_rows(self):
        invented = ROWS + [LineItem(description="Matcha Latte", line_total="25000")]
        assert table_grounding(ROWS, TEXT) == 1.0
        assert table_grounding(invented, TEXT) == pytest.approx(2 / 3, abs=1e-4)

    def test_grounding_is_neutral_for_an_empty_table(self):
        assert table_grounding([], TEXT) == 0.5

    def test_agreement_degrades_gradually_on_a_dropped_row(self):
        assert table_agreement(ROWS, ROWS) == 1.0
        partial = table_agreement(ROWS, ROWS[:1])
        assert 0.0 < partial < 1.0

    def test_both_empty_agree(self):
        assert table_agreement([], []) == 1.0

    def test_signals_include_the_table(self):
        signals = build_signals(Receipt(subtotal="48000", line_items=ROWS),
                                Receipt(line_items=ROWS), TEXT, {})
        t = signals["line_items"]
        assert (t.agreement, t.grounded, t.validated) == (1.0, 1.0, 1.0)

    def test_table_has_its_own_calibration_group(self):
        assert group_of("line_items") == "table"
        assert "table" in ConfidenceModel().groups


class TestScoring:
    def test_table_is_a_confidence_observation_judged_on_perfection(self):
        gt = Receipt(line_items=ROWS)
        perfect = metrics.compare_document("a", Receipt(line_items=ROWS), gt,
                                           confidence={"line_items": 0.9})
        dropped = metrics.compare_document("b", Receipt(line_items=ROWS[:1]), gt,
                                           confidence={"line_items": 0.2})
        obs = [o for o in metrics.confidence_observations([perfect, dropped]) if o[0] == "line_items"]
        assert obs == [("line_items", 0.9, True), ("line_items", 0.2, False)]

    def test_table_counts_toward_calibration_auc(self):
        gt = Receipt(line_items=ROWS)
        results = [
            metrics.compare_document("a", Receipt(line_items=ROWS), gt, confidence={"line_items": 0.9}),
            metrics.compare_document("b", Receipt(line_items=ROWS[:1]), gt, confidence={"line_items": 0.2}),
        ]
        cal = metrics.calibration(results)
        assert cal["per_field"]["line_items"]["auc"] == 1.0


class TestWithholding:
    def _result(self, abstained):
        return PipelineResult(receipt=Receipt(total="48000", line_items=ROWS), receipt_b=None,
                              transcription=TEXT, confidence={"total": 0.9, "line_items": 0.2},
                              abstained=abstained)

    def test_abstained_table_is_withheld_whole(self):
        out = from_result("d", self._result(["line_items"]), 0.5)
        assert out.receipt.line_items == []
        rows = json.loads(out.withheld["line_items"])
        assert [r["description"] for r in rows] == ["Es Kopi Susu", "Croissant"]
        assert out.receipt.total == Decimal("48000")

    def test_confident_table_is_returned(self):
        assert len(from_result("d", self._result([]), 0.5).receipt.line_items) == 2

    def test_withheld_table_round_trips_through_the_eval_client(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from PIL import Image

        from api import app as app_module
        from api import service
        from eval import run as eval_run

        for key in service.REQUIRED_ENV + service.RECOMMENDED_ENV:
            monkeypatch.setenv(key, "test")
        monkeypatch.setenv("EXTRACTION_MODE", "pipeline")
        monkeypatch.setattr(service, "extract_with_confidence",
                            lambda *a, **k: self._result(["line_items"]))
        monkeypatch.setattr(eval_run, "_api_client", TestClient(app_module.app))
        path = tmp_path / "r.jpg"
        Image.new("RGB", (20, 30), "white").save(path)

        ext = eval_run.extract_one(path, "api", None, 0.5, True)
        assert [i.description for i in ext.receipt.line_items] == ["Es Kopi Susu", "Croissant"]
        assert ext.receipt.line_items[0].line_total == Decimal("36000")
