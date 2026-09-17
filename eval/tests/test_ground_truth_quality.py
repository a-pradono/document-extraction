"""Cross-section reconciliation and ground-truth quality. Run: pytest eval/tests -q

Every regression case here came from auditing the real 50 labels, not from imagination:
CORD service charges, WildReceipt tips, cash-rounded totals, tilted-photo row pairing,
and a dataset annotation that reads 5.90 as 5390.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import audit_gt  # noqa: E402
import gt_common  # noqa: E402
from convert_cord import _to_receipt  # noqa: E402
from convert_wildreceipt import _group_rows, align_column  # noqa: E402
from extractor.confidence import money_tolerance, reconciles, validate_arithmetic  # noqa: E402
from extractor.schema import Receipt  # noqa: E402

D = Decimal


class TestReconciliation:
    def test_service_charge_closes_the_arithmetic(self):
        """Regression: CORD cord_0002 is 238,000 + 17,850 service + 25,585 tax = 281,435.
        Without service_charge in the schema, a CORRECT extraction failed validation."""
        r = Receipt(subtotal="238000", service_charge="17850", tax="25585", total="281435")
        v = validate_arithmetic(r)
        assert v["total"] == v["service_charge"] == v["subtotal"] == 1.0

    def test_missing_service_charge_still_fails(self):
        r = Receipt(subtotal="238000", tax="25585", total="281435")
        assert validate_arithmetic(r)["total"] == 0.0

    def test_tip_reconciles(self):
        """WildReceipt wild_0024: 52.00 + 4.30 tax + 9.36 gratuity = 65.66."""
        r = Receipt(subtotal="52.00", tax="4.30", service_charge="9.36", total="65.66")
        assert validate_arithmetic(r)["total"] == 1.0

    def test_discount_is_subtracted_whatever_its_sign(self):
        for discount in ("10000", "-10000"):
            r = Receipt(subtotal="60000", discount=discount, total="50000")
            assert validate_arithmetic(r)["total"] == 1.0

    @pytest.mark.parametrize("expected,printed", [
        ("334011", "334000"),        # CORD: rounded to nearest 100
        ("680339", "680500"),        # CORD: rounded up to 500
        ("364999.68", "365000"),     # CORD cord_0011: sub-unit rounding
    ])
    def test_cash_rounding_reconciles(self, expected, printed):
        assert reconciles(D(expected), D(printed), money_tolerance(D(printed)))

    @pytest.mark.parametrize("expected,printed", [
        ("334011", "334300"),        # a misread hundreds digit is not a rounding
        ("680339", "681339"),        # nor is a misread thousands digit
        ("19.38", "19.45"),          # small currencies keep the 0.02 tolerance
    ])
    def test_misreads_do_not_hide_behind_rounding(self, expected, printed):
        assert not reconciles(D(expected), D(printed), money_tolerance(D(printed)))

    def test_rows_sum_to_total_less_charges_when_no_subtotal(self):
        r = Receipt(total="110", service_charge="10",
                    line_items=[{"description": "A", "line_total": "60"},
                                {"description": "B", "line_total": "40"}])
        assert validate_arithmetic(r)["line_items"] == 1.0


class TestCordConverter:
    def test_service_and_discount_are_converted(self):
        parse = {"menu": [{"nm": "X", "price": "60.000"}],
                 "sub_total": {"subtotal_price": "60.000", "service_price": "3.000",
                               "discount_price": "-5.000", "tax_price": "6.000"},
                 "total": {"total_price": "64.000"}}
        r = _to_receipt(parse)
        assert (r["service_charge"], r["discount"]) == ("3000", "5000")
        rec = Receipt.model_validate(r)
        assert validate_arithmetic(rec)["total"] == 1.0

    def test_nested_add_ons_become_their_own_rows(self):
        """Regression: cord_0001 prints 'Kupon 15 100,000' and 'ADD CHICKEN BOX 909'
        (subtotal 100,909). The add-on is nested under `sub` in CORD and was dropped, so
        the first real model run was scored as inventing a row the label lacked."""
        parse = {"menu": {"nm": "Kupon 15", "price": "100,000",
                          "sub": {"nm": "ADD CHICKEN BOX", "price": "909"}},
                 "sub_total": {"subtotal_price": "100,909", "tax_price": "10,091"},
                 "total": {"total_price": "111,000"}}
        r = _to_receipt(parse)
        assert [i["description"] for i in r["line_items"]] == ["Kupon 15", "ADD CHICKEN BOX"]
        assert validate_arithmetic(Receipt.model_validate(r))["line_items"] == 1.0

    def test_list_of_sub_items(self):
        parse = {"menu": [{"nm": "Tea", "price": "10.000",
                           "sub": [{"nm": "Jelly", "price": "4.000"}, {"nm": "Less sugar"}]}],
                 "total": {"total_price": "14.000"}}
        assert [i["description"] for i in _to_receipt(parse)["line_items"]] == ["Tea", "Jelly", "Less sugar"]

    def test_list_shaped_sub_total_does_not_crash(self):
        parse = {"menu": [{"nm": "X", "price": "1.000"}],
                 "sub_total": [{"subtotal_price": "1.000"}], "total": [{"total_price": "1.000"}]}
        assert _to_receipt(parse)["total"] == "1000"


class TestRowAlignment:
    def test_order_preserving_beats_nearest_neighbour_on_a_tilted_photo(self):
        """Regression, from wild_0018's real coordinates: every price sits ~11px below its
        item and the row pitch is ~14.5px, so nearest-neighbour shifted the whole table
        down one row and orphaned Kopi. Its six prices sum exactly to the 47.30 subtotal."""
        items = [187.5, 204.0, 217.5, 232.0, 246.5, 261.5]
        prices = [198.5, 211.5, 224.5, 238.5, 251.5, 266.0]
        assert align_column(items, prices, tol=13.0) == [(i, i) for i in range(6)]

    def test_matches_never_cross(self):
        pairs = align_column([10, 20, 30], [31, 19, 11], tol=15)
        firsts = [a for a, _ in pairs]
        seconds = [b for _, b in pairs]
        assert firsts == sorted(firsts) and seconds == sorted(seconds)

    def test_a_missing_price_leaves_that_item_unpriced_not_shifted(self):
        pairs = align_column([10, 30, 50], [11, 51], tol=8)
        assert pairs == [(0, 0), (2, 1)]

    def test_wild_0018_rows_end_to_end(self):
        names = ["Kopi", "Wanton", "SpyP", "PorkChop", "ChickenFloss", "Toast"]
        items = [187.5, 204.0, 217.5, 232.0, 246.5, 261.5]
        prices = ["1.70", "8.90", "7.80", "12.50", "10.50", "5.90"]
        price_ys = [198.5, 211.5, 224.5, 238.5, 251.5, 266.0]
        boxes = [(y, 260.0, 17.0, "description", n) for y, n in zip(items, names)]
        boxes += [(y, 375.0, 12.0, "line_total", p) for y, p in zip(price_ys, prices)]
        rows = _group_rows(boxes)
        assert [(r["description"], r["line_total"]) for r in rows] == list(zip(names, prices))
        assert sum(D(r["line_total"]) for r in rows) == D("47.30")


class TestAudit:
    def test_clean_label_has_no_flags(self):
        label = {"subtotal": "48000", "service_charge": "2000", "tax": "5000", "total": "55000",
                 "line_items": [{"description": "Kopi", "quantity": "2", "unit_price": "24000",
                                 "line_total": "48000"}], "date": "2024-03-12"}
        assert audit_gt.audit_receipt(label) == []

    def test_dataset_annotation_error_is_caught_by_the_row_sum(self):
        """wild_0018: the dataset annotates 5.90 as '5390'. Once rows pair correctly the
        sum check surfaces it."""
        label = {"subtotal": "47.30", "line_items": [
            {"description": "Kopi", "line_total": "1.70"},
            {"description": "Toast", "line_total": "5390"}]}
        assert any(f["field"] == "line_items" for f in audit_gt.audit_receipt(label))

    @pytest.mark.parametrize("label,field", [
        ({"subtotal": "10", "tax": "1", "total": "15"}, "total"),
        ({"total": "5", "tax": "9"}, "tax"),
        ({"date": "2099-01-01"}, "date"),
        ({"merchant_name": "CaltexCentralMyrtleford"}, "merchant_name"),
        ({"line_items": [{"description": "A", "quantity": "2", "unit_price": "5", "line_total": "11"}]},
         "line_items[0]"),
        ({"line_items": [{"line_total": "4.99"}]}, "line_items[0]"),
    ])
    def test_suspicious_labels_are_flagged(self, label, field):
        assert field in [f["field"] for f in audit_gt.audit_receipt(label)]

    def test_future_date_is_an_error_not_a_check(self):
        flags = audit_gt.audit_receipt({"date": "2099-01-01"})
        assert flags[0]["severity"] == audit_gt.ERROR


class TestSamplePinning:
    def test_pins_come_from_provenance(self, tmp_path, monkeypatch):
        prov = tmp_path / "_provenance.json"
        prov.write_text(json.dumps({"cord_0001": {"original_file": "a.png"},
                                    "wild_0001": {"original_file": "b.jpeg"}}))
        monkeypatch.setattr(gt_common, "PROVENANCE", prov)
        assert gt_common.pinned_selection("cord") == {"cord_0001": "a.png"}

    def test_verification_survives_only_identical_labels(self, tmp_path, monkeypatch):
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        (gt_dir / "d_0001.json").write_text('{"total": "1"}')
        (gt_dir / "d_0002.json").write_text('{"total": "2"}')
        import hashlib
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]  # noqa: E731
        prov = tmp_path / "_provenance.json"
        prov.write_text(json.dumps({
            "d_0001": {"verified": True, "label_sha": sha(gt_dir / "d_0001.json"), "verified_note": "ok"},
            "d_0002": {"verified": True, "label_sha": "stale"},
        }))
        monkeypatch.setattr(gt_common, "PROVENANCE", prov)
        monkeypatch.setattr(gt_common, "GT_DIR", gt_dir)
        entries = {"d_0001": {"verified": False}, "d_0002": {"verified": False}}
        gt_common.carry_verification(entries)
        assert entries["d_0001"]["verified"] is True
        assert entries["d_0002"]["verified"] is False
        assert "re-verify" in entries["d_0002"]["verified_note"]
