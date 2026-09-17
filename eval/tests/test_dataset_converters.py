"""Tests for the dataset converters' correctness-critical parts.

A bug here is worse than a bug in the extractor: it corrupts the ground truth silently,
and every metric downstream then measures the wrong thing while looking healthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from convert_cord import _to_receipt  # noqa: E402
from convert_wildreceipt import _group_rows, convert as wild_convert, parse_date  # noqa: E402
from gt_common import normalize_money, normalize_qty, stratified_sample  # noqa: E402


class TestNormalizeMoney:
    @pytest.mark.parametrize("raw,expected", [
        ("53280", "53280"),
        ("53.280", "53280"),        # IDR thousands separator (CORD)
        ("53,280", "53280"),
        ("1.234.567", "1234567"),
        ("53.28", "53.28"),         # US decimal (WildReceipt)
        ("$53.28", "53.28"),
        ("53,28", "53.28"),         # European decimal comma
        ("Rp 12.000", "12000"),
        ("  12,345.67  ", "12345.67"),
        ("0.50", "0.50"),
        ("007", "7"),
    ])
    def test_values(self, raw, expected):
        assert normalize_money(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "  ", "TOTAL", "N/A", "-"])
    def test_no_number_is_none(self, raw):
        assert normalize_money(raw) is None

    @pytest.mark.parametrize("raw,expected", [("-5.00", "-5.00"), ("(5.00)", "-5.00")])
    def test_negatives(self, raw, expected):
        assert normalize_money(raw) == expected

    def test_output_parses_as_decimal(self):
        from decimal import Decimal
        assert Decimal(normalize_money("1.234.567")) == Decimal(1234567)


class TestNormalizeQty:
    @pytest.mark.parametrize("raw,expected", [
        ("2", "2"), ("x2", "2"), ("2 pcs", "2"), ("1.0", "1"), ("1,000", "1000"),
        ("0.5", "0.5"),
    ])
    def test_values(self, raw, expected):
        assert normalize_qty(raw) == expected


class TestParseDate:
    @pytest.mark.parametrize("raw,expected", [
        ("2018-03-12", "2018-03-12"),
        ("03/12/2018", "2018-03-12"),     # US MM/DD
        ("3/12/18", "2018-03-12"),
        ("MAR 12, 2018", "2018-03-12"),
        ("March 12 2018", "2018-03-12"),
    ])
    def test_parsed(self, raw, expected):
        assert parse_date(raw) == expected

    @pytest.mark.parametrize("raw", [
        "25/03/2018",    # unambiguously DD/MM — locale unknown, refuse rather than guess
        "13/13/2018",    # impossible
        "02/30/2018",    # impossible day
        "1999-01-01",    # implausible for this dataset
        "no date here",
        "",
    ])
    def test_refused_rather_than_guessed(self, raw):
        assert parse_date(raw) is None


class TestRowGrouping:
    """Boxes exist for exactly one job: putting description/qty/price on one line item."""

    @staticmethod
    def _box(y, x, field, text, h=20.0):
        return (y, x, h, field, text)

    def test_groups_by_vertical_proximity(self):
        boxes = [
            self._box(100, 10, "description", "Coffee"),
            self._box(103, 200, "quantity", "2"),
            self._box(101, 300, "line_total", "5.00"),
            self._box(160, 10, "description", "Bagel"),
            self._box(162, 300, "line_total", "3.25"),
        ]
        items = _group_rows(boxes)
        assert items == [
            {"description": "Coffee", "quantity": "2", "line_total": "5.00"},
            {"description": "Bagel", "line_total": "3.25"},
        ]

    def test_rows_are_top_to_bottom_regardless_of_input_order(self):
        boxes = [
            self._box(300, 10, "description", "Third"),
            self._box(100, 10, "description", "First"),
            self._box(200, 10, "description", "Second"),
        ]
        assert [i["description"] for i in _group_rows(boxes)] == ["First", "Second", "Third"]

    def test_no_coordinate_survives_into_the_item(self):
        items = _group_rows([self._box(100, 10, "description", "Coffee"),
                             self._box(100, 300, "line_total", "5.00")])
        assert set(items[0]) <= {"description", "quantity", "unit_price", "line_total"}

    def test_empty_input(self):
        assert _group_rows([]) == []

    def test_stray_zero_price_box_is_not_a_line_item(self):
        boxes = [self._box(100, 10, "description", "Coffee"),
                 self._box(100, 300, "line_total", "5.00"),
                 self._box(400, 300, "line_total", "0.00")]
        assert _group_rows(boxes) == [{"description": "Coffee", "line_total": "5.00"}]

    def test_a_described_item_priced_zero_is_kept(self):
        boxes = [self._box(100, 10, "description", "Promo item"),
                 self._box(100, 300, "line_total", "0.00")]
        assert _group_rows(boxes) == [{"description": "Promo item", "line_total": "0.00"}]


class TestCordConversion:
    def test_maps_gt_parse_to_schema(self):
        parse = {
            "menu": [
                {"nm": "ES KOPI SUSU", "cnt": "2x", "unitprice": "18.000", "price": "36.000"},
                {"nm": "CROISSANT", "cnt": "1", "unitprice": "12.000", "price": "12.000"},
            ],
            "sub_total": {"subtotal_price": "48.000", "tax_price": "5.280"},
            "total": {"total_price": "53.280", "cashprice": "100.000"},
        }
        assert _to_receipt(parse) == {
            "subtotal": "48000", "tax": "5280", "total": "53280",
            "line_items": [
                {"description": "ES KOPI SUSU", "quantity": "2",
                 "unit_price": "18000", "line_total": "36000"},
                {"description": "CROISSANT", "quantity": "1",
                 "unit_price": "12000", "line_total": "12000"},
            ],
        }

    def test_single_item_menu_arrives_as_a_bare_dict(self):
        parse = {"menu": {"nm": "TEH", "cnt": "1", "price": "8.000"},
                 "total": {"total_price": "8.000"}}
        assert _to_receipt(parse)["line_items"] == [
            {"description": "TEH", "quantity": "1", "line_total": "8000"}
        ]

    def test_absent_subtotal_and_tax_stay_absent(self):
        receipt = _to_receipt({"menu": [{"nm": "X", "price": "1.000"}],
                               "total": {"total_price": "1.000"}})
        assert "subtotal" not in receipt and "tax" not in receipt

    def test_cashprice_is_not_mistaken_for_the_total(self):
        receipt = _to_receipt({"menu": [{"nm": "X", "price": "1.000"}],
                               "total": {"total_price": "1.000", "cashprice": "50.000"}})
        assert receipt["total"] == "1000"


class TestWildReceiptConversion:
    CLASSES = {0: "Ignore", 1: "Store_name_value", 3: "Date_value",
               7: "Prod_item_value", 8: "Prod_quantity_value", 9: "Prod_price_value",
               11: "Subtotal_value", 12: "Tax_value", 13: "Total_value"}

    @staticmethod
    def _ann(label, text, y, x=10.0):
        box = [x, y, x + 80, y, x + 80, y + 20, x, y + 20]
        return {"box": box, "text": text, "label": label}

    def test_end_to_end_row(self):
        row = {"annotations": [
            self._ann(1, "WALMART", 20),
            self._ann(3, "03/12/2018", 50),
            self._ann(7, "MILK 2%", 100),
            self._ann(9, "3.49", 100, x=300),
            self._ann(11, "3.49", 200),
            self._ann(12, "0.24", 230),
            self._ann(13, "3.73", 260),
            self._ann(0, "THANK YOU", 400),
        ]}
        assert wild_convert(row, self.CLASSES) == {
            "merchant_name": "WALMART",
            "date": "2018-03-12",
            "subtotal": "3.49", "tax": "0.24", "total": "3.73",
            "line_items": [{"description": "MILK 2%", "line_total": "3.49"}],
        }

    def test_ignore_class_contributes_nothing(self):
        row = {"annotations": [self._ann(0, "SURVEY CODE 1234", 20),
                               self._ann(13, "9.99", 100)]}
        assert wild_convert(row, self.CLASSES) == {"total": "9.99"}

    def test_unparseable_date_becomes_null_not_a_guess(self):
        row = {"annotations": [self._ann(3, "25/03/2018", 20), self._ann(13, "1.00", 60)]}
        assert "date" not in wild_convert(row, self.CLASSES)

    def test_repeated_total_annotations_collapse_to_one_value(self):
        """Regression: a receipt annotates Total_value once per copy — the line, the
        EFTPOS slip, the signature copy. Concatenating them produced a total of
        '77437743774377.43' from four boxes that all said 77.43."""
        row = {"annotations": [
            self._ann(13, "77.43", 100),
            self._ann(13, "77.43", 200),
            self._ann(13, "AU$77.43", 300),
            self._ann(13, "AU$77.43", 400),
        ]}
        assert wild_convert(row, self.CLASSES)["total"] == "77.43"

    def test_conflicting_repeats_take_the_most_frequent(self):
        row = {"annotations": [self._ann(13, "10.00", 100), self._ann(13, "10.00", 200),
                               self._ann(13, "99.99", 300)]}
        assert wild_convert(row, self.CLASSES)["total"] == "10.00"

    def test_ties_break_toward_the_bottom_of_the_receipt(self):
        row = {"annotations": [self._ann(13, "10.00", 100), self._ann(13, "12.00", 500)]}
        assert wild_convert(row, self.CLASSES)["total"] == "12.00"

    def test_repeated_store_name_is_not_duplicated(self):
        row = {"annotations": [self._ann(1, "CALTEX", 20), self._ann(1, "CALTEX", 300),
                               self._ann(13, "1.00", 60)]}
        assert wild_convert(row, self.CLASSES)["merchant_name"] == "CALTEX"

    def test_multi_box_store_name_joins_left_to_right(self):
        row = {"annotations": [self._ann(1, "SUPER", 20, x=10),
                               self._ann(1, "MART", 20, x=120),
                               self._ann(13, "1.00", 60)]}
        assert wild_convert(row, self.CLASSES)["merchant_name"] == "SUPER MART"


class TestStratifiedSample:
    def test_spans_the_line_item_range_instead_of_taking_the_head(self):
        # A pool ordered short-to-long: taking the first n would be all 2-item receipts.
        pool = [(f"d{i}", i, f"merchant{i % 7}") for i in range(2, 22)]
        picked = stratified_sample(pool, 8)
        sizes = [next(c for id_, c, _ in pool if id_ == p) for p in picked]
        assert len(picked) == 8
        assert min(sizes) <= 3 and max(sizes) >= 14

    def test_is_deterministic_for_a_fixed_seed(self):
        pool = [(f"d{i}", i % 17, f"m{i % 5}") for i in range(60)]
        assert stratified_sample(pool, 12) == stratified_sample(pool, 12)

    def test_prefers_distinct_merchants(self):
        pool = [(f"d{i}", 5, "same") for i in range(10)] + \
               [(f"u{i}", 5, f"uniq{i}") for i in range(10)]
        picked = set(stratified_sample(pool, 10))
        assert len(picked & {f"u{i}" for i in range(10)}) >= 8

    def test_never_returns_more_than_asked(self):
        pool = [(f"d{i}", i, f"m{i}") for i in range(5)]
        assert len(stratified_sample(pool, 25)) == 5
