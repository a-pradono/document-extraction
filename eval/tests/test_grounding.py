"""Type-aware groundedness — the anti-hallucination signal. Run: pytest eval/tests -q

Both directions matter. A false negative marks a correct value as hallucinated and
drags its confidence down; a false positive lets an invented value through.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from extractor.grounding import date_readings, is_grounded, money_readings
from extractor.pipeline import build_signals
from extractor.schema import Receipt

CORD = "KOPI KENANGAN\nEs Kopi Susu 2 36.000\nSUBTOTAL 48.000\nPB1 5.280\nTOTAL 53.280\nTgl 12/03/2024"
US = "LUNA GRILL\n691 Collection Blvd\nSubtotal $1,234.50\nTax 1.39\nTOTAL 19.38\nMar 7, 2017 VISA"


class TestMoney:
    @pytest.mark.parametrize("value,text", [
        (Decimal("53280"), CORD),          # regression: IDR thousands separator
        (Decimal("5280"), CORD),
        (Decimal("1234.50"), US),          # regression: comma thousands
        (Decimal("19.38"), US),
        (Decimal("19.5"), "TOTAL 19.50"),  # trailing zero
        (Decimal("7"), "QTY 7"),
    ])
    def test_correct_values_are_grounded(self, value, text):
        assert is_grounded(value, text, "money")

    @pytest.mark.parametrize("value,text", [
        (Decimal("99999"), CORD),          # invented
        (Decimal("5"), CORD),              # regression: substring of 53.280
        (Decimal("28"), CORD),             # substring of 5.280
        (Decimal("123450"), US),           # "1,234.50" is not 123450
        (Decimal("3"), "TOTAL 53.280"),
    ])
    def test_invented_values_are_not(self, value, text):
        assert not is_grounded(value, text, "money")

    @pytest.mark.parametrize("token,expected", [
        ("53.280", {Decimal("53280"), Decimal("53.280")}),
        ("1,234.50", {Decimal("1234.50")}),
        ("12,5", {Decimal("12.5")}),
        ("1.234.567", {Decimal("1234567")}),
        ("42", {Decimal("42")}),
    ])
    def test_readings(self, token, expected):
        assert money_readings(token) == expected

    def test_space_stripped_text_still_grounds(self):
        """WildReceipt transcriptions glue labels to amounts ("TOTAL19.38")."""
        assert is_grounded(Decimal("19.38"), "TOTAL19.38", "money")


class TestDate:
    @pytest.mark.parametrize("value,text", [
        (date(2024, 3, 12), CORD),                     # regression: DD/MM/YYYY printed
        (date(2017, 3, 7), US),                        # month name
        (date(2024, 3, 12), "DATE 2024-03-12"),
        (date(2018, 3, 12), "03/12/18"),               # US MM/DD/YY
        (date(2024, 5, 3), "3 Mei 2024"),              # Indonesian month
        (date(2012, 1, 24), "Date24/01/2012"),         # space-stripped
    ])
    def test_printed_formats_ground(self, value, text):
        assert is_grounded(value, text, "date")

    def test_ambiguous_date_grounds_both_readings(self):
        assert {date(2024, 4, 3), date(2024, 3, 4)} <= date_readings("03/04/2024")

    def test_invented_date_is_not_grounded(self):
        assert not is_grounded(date(2024, 3, 13), CORD, "date")

    def test_impossible_readings_are_dropped(self):
        assert date(2024, 12, 25) in date_readings("25/12/2024")
        assert len(date_readings("25/12/2024")) == 1   # 12/25 as D/M is not a date


class TestCurrency:
    @pytest.mark.parametrize("code,text", [
        ("IDR", CORD.replace("TOTAL", "TOTAL Rp")),    # regression: printed as symbol
        ("USD", US),
        ("AUD", "AU$77.43"),
        ("IDR", "Total IDR 53.280"),
    ])
    def test_grounded(self, code, text):
        assert is_grounded(code, text, "currency")

    def test_ungrounded_when_nothing_on_the_receipt_says_so(self):
        assert not is_grounded("IDR", CORD, "currency")

    def test_letters_inside_words_do_not_count(self):
        assert not is_grounded("IDR", "CIDRE apple", "currency")
        assert not is_grounded("MYR", "Rmx storm", "currency")


class TestString:
    def test_case_and_whitespace_insensitive(self):
        assert is_grounded("Kopi Kenangan", CORD, "string")

    def test_whole_tokens_only(self):
        assert not is_grounded("Kopi", "KOPITIAM", "string")
        assert not is_grounded("an", CORD, "string")

    def test_punctuation_ignored(self):
        assert is_grounded("691 Collection Blvd.", US, "string")

    @pytest.mark.parametrize("value", [None, ""])
    def test_empty_is_not_grounded(self, value):
        assert not is_grounded(value, CORD, "string")


class TestSignalAssembly:
    def test_correct_cord_money_now_scores_grounded(self):
        """End-to-end over the real signal path: the old substring check scored every
        one of these 0.0 — i.e. as hallucinated — on a correct extraction."""
        receipt = Receipt(subtotal="48000", tax="5280", total="53280", date="2024-03-12")
        signals = build_signals(receipt, None, CORD, {})
        assert all(signals[f].grounded == 1.0 for f in ("subtotal", "tax", "total", "date"))

    def test_hallucinated_total_scores_ungrounded(self):
        signals = build_signals(Receipt(total="5"), None, CORD, {})
        assert signals["total"].grounded == 0.0

    def test_absent_value_stays_neutral(self):
        assert build_signals(Receipt(), None, CORD, {})["total"].grounded == 0.5
