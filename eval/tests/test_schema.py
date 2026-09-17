"""Schema guarantees the rest of the system assumes. Run: pytest eval/tests -q"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from extractor.schema import LineItem, Receipt


class TestTypesResolve:
    def test_date_is_a_real_date_not_nonetype(self):
        """Regression: the field name `date` shadows the imported `date` type in the
        class namespace, so a bare `Optional[date]` annotation resolves to NoneType and
        every date in the system silently becomes null — including in ground truth."""
        assert Receipt.model_fields["date"].annotation == datetime.date | None

    def test_date_string_parses(self):
        assert Receipt(date="2024-03-12").date == datetime.date(2024, 3, 12)

    def test_money_is_decimal_never_float(self):
        r = Receipt(total="53280.10")
        assert isinstance(r.total, Decimal) and r.total == Decimal("53280.10")

    def test_money_string_avoids_binary_float_error(self):
        assert Receipt(subtotal="0.1", tax="0.2").subtotal + Receipt(tax="0.2").tax \
            == Decimal("0.3")


class TestAbsence:
    def test_every_field_defaults_to_absent(self):
        r = Receipt()
        assert r.merchant_name is None and r.total is None and r.line_items == []

    def test_explicit_null_is_accepted(self):
        assert Receipt(date=None, total=None).total is None

    def test_line_items_accept_partial_rows(self):
        item = LineItem(description="Es Kopi Susu")
        assert item.quantity is None and item.line_total is None


class TestValidation:
    def test_unparseable_date_is_rejected_not_coerced(self):
        with pytest.raises(ValidationError):
            Receipt(date="not a date")

    def test_unparseable_money_is_rejected(self):
        with pytest.raises(ValidationError):
            Receipt(total="twelve dollars")

    def test_stray_keys_are_ignored_not_fatal(self):
        assert Receipt.model_validate({"total": "1", "_confidence": {"total": 0.9}}).total \
            == Decimal("1")


class TestGroundTruthFilesLoad:
    def test_every_committed_ground_truth_file_validates(self):
        from pathlib import Path

        gt_dir = Path("data/ground_truth")
        files = [p for p in gt_dir.glob("*.json") if not p.stem.startswith("_")]
        if not files:
            pytest.skip("no ground truth generated yet")
        for path in files:
            Receipt.model_validate_json(path.read_text(encoding="utf-8"))
