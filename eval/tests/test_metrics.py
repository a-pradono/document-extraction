"""Correctness tests for the eval primitives. Run: pytest eval/tests -q"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from extractor.schema import LineItem, Receipt
from eval import metrics, taxonomy


def test_money_normalized_match_but_not_exact():
    c = metrics.compare_scalar("total", Decimal("1000"), Decimal("1000.00"))
    assert c.label == taxonomy.CORRECT
    assert c.normalized is True
    assert c.exact is False


def test_missing_and_spurious():
    assert metrics.compare_scalar("tax", Decimal("5"), None).label == taxonomy.MISSING
    assert metrics.compare_scalar("tax", None, Decimal("5")).label == taxonomy.SPURIOUS


def test_both_absent_is_true_negative():
    c = metrics.compare_scalar("payment_method", None, None)
    assert c.label == taxonomy.CORRECT
    assert c.gt_present is False and c.pred_present is False


def test_format_error_vs_wrong_value():
    fmt = metrics.compare_scalar("merchant_name", "ACME, Inc.", "ACME Inc")
    assert fmt.label == taxonomy.FORMAT_ERROR
    wrong = metrics.compare_scalar("merchant_name", "ACME", "Beta Corp")
    assert wrong.label == taxonomy.WRONG_VALUE


def test_currency_symbol_normalization():
    assert metrics.compare_scalar("currency", "IDR", "Rp").label == taxonomy.CORRECT


def test_date_iso_match():
    assert metrics.compare_scalar("date", date(2023, 2, 1), date(2023, 2, 1)).label == taxonomy.CORRECT


def test_swapped_field_detection():
    gt = Receipt(subtotal=Decimal("90"), total=Decimal("100"))
    pred = Receipt(subtotal=Decimal("90"), total=Decimal("90"))
    res = metrics.compare_document("d1", pred, gt)
    total_cmp = next(c for c in res.comparisons if c.field == "total")
    assert total_cmp.label == taxonomy.SWAPPED_FIELD


def test_line_item_matching_and_f1():
    gt = [LineItem(description="Coffee", line_total=Decimal("5.00")),
          LineItem(description="Muffin", line_total=Decimal("3.00"))]
    pred = [LineItem(description="coffee", line_total=Decimal("5")),
            LineItem(description="Muffin", line_total=Decimal("3.00"))]
    stats = metrics.score_line_items(gt, pred)
    assert stats["matched"] == 2
    assert stats["f1"] == 1.0
    assert stats["field_accuracy"]["line_total"] == 1.0


def test_aggregate_shapes():
    gt = Receipt(merchant_name="Acme", total=Decimal("10.00"))
    pred = Receipt(merchant_name="Acme", total=Decimal("10.00"))
    agg = metrics.aggregate([metrics.compare_document("d", pred, gt)])
    assert agg["document_accuracy_rate"] == 1.0
    assert "per_field" in agg and "error_taxonomy" in agg


def test_field_blank_everywhere_is_undefined_not_zero():
    """Regression: a field with no ground truth and no prediction in the whole set
    scored F1 0.000 and was averaged into macro F1, reading as total failure on a field
    the system got entirely right."""
    from decimal import Decimal
    from eval import metrics as m
    results = [m.compare_document(f"d{i}", Receipt(total=Decimal("1")), Receipt(total=Decimal("1")))
               for i in range(3)]
    agg = m.aggregate(results)
    assert agg["per_field"]["payment_method"]["f1"] is None
    assert agg["per_field"]["total"]["f1"] == 1.0
    assert agg["macro_f1"] == 1.0


def test_spurious_on_an_unlabelled_field_still_scores_zero():
    from decimal import Decimal
    from eval import metrics as m
    results = [m.compare_document("d", Receipt(total=Decimal("1"), payment_method="VISA"),
                                  Receipt(total=Decimal("1")))]
    assert m.aggregate(results)["per_field"]["payment_method"]["f1"] == 0.0
