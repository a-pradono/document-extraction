"""Groundedness: could this extracted value have been read from the transcription?

The anti-hallucination confidence signal. It must be TYPE-AWARE, because the value we
check is the parsed schema value and the transcription is what was printed:

    parsed Decimal 53280        printed "53.280"      (IDR thousands separator)
    parsed date 2024-03-12      printed "12/03/2024"
    parsed "IDR"                printed "Rp"

A plain substring check got both directions wrong. Correct values read as ungrounded
(every CORD total), and hallucinated ones read as grounded ("5" is a substring of
"53.280"). That made the signal anti-informative on exactly the tier it is meant to
protect. Here every comparison is on whole tokens, after parsing the printed text the
same way the value was parsed.

Grounding asks "is there ANY reading of the text that yields this value", so ambiguity
is resolved permissively: "03/04/2024" grounds both 3 April and 4 March. Deciding which
one is right is the extractor's job; this signal only catches values with no source.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

_NUMBER = re.compile(r"(?<![\d])\d[\d.,]*\d(?![\d])|(?<![\d.,])\d(?![\d])")
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTHS.update({"mei": 5, "agu": 8, "agt": 8, "okt": 10, "des": 12})  # Indonesian (CORD)

# Symbols a receipt prints instead of the ISO code. "$" is genuinely ambiguous, so it
# grounds any dollar currency rather than guessing which one.
_SYMBOLS: dict[str, set[str]] = {
    "$": {"USD", "AUD", "CAD", "SGD", "NZD", "HKD"},
    "rp": {"IDR"},
    "€": {"EUR"},
    "£": {"GBP"},
    "¥": {"JPY", "CNY"},
    "rm": {"MYR"},
    "₹": {"INR"},
    "aed": {"AED"},
    "dhs": {"AED"},
}


def is_grounded(value: Any, text: str, kind: str = "string") -> bool:
    """`kind` is the field group: string | money | date | currency."""
    if value is None or not text:
        return False
    if kind == "money":
        return _money_grounded(value, text)
    if kind == "date":
        return _date_grounded(value, text)
    if kind == "currency":
        return _currency_grounded(value, text)
    return _string_grounded(value, text)


# --- money -------------------------------------------------------------------
def _decimal(s: str) -> Decimal | None:
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def money_readings(token: str) -> set[Decimal]:
    """Every value a printed number could denote under either separator convention.

    "53.280" -> {53.28, 53280}   "1,234.50" -> {1234.50}   "12,5" -> {12.5, 125}
    """
    readings: set[Decimal] = set()
    groups = re.split(r"[.,]", token)
    if not all(g.isdigit() for g in groups):
        return readings
    # Every separator is a thousands separator: only valid if each later group is 3 digits.
    if all(len(g) == 3 for g in groups[1:]):
        readings.add(Decimal("".join(groups)))
    # The last separator is the decimal point: earlier groups must still be thousands, and
    # written with the OTHER character — "1.234.567" has no decimal reading.
    separators = re.findall(r"[.,]", token)
    if (len(groups) > 1 and 1 <= len(groups[-1]) <= 3
            and all(len(g) == 3 for g in groups[1:-1])
            and separators[-1] not in separators[:-1]):
        readings.add(Decimal("".join(groups[:-1]) + "." + groups[-1]))
    return readings


def _money_grounded(value: Any, text: str) -> bool:
    target = _decimal(str(value))
    if target is None:
        return False
    return any(target == reading
               for token in _NUMBER.findall(text)
               for reading in money_readings(token))


# --- dates -------------------------------------------------------------------
def _valid(y: int, m: int, d: int) -> date | None:
    if y < 100:
        y += 2000
    try:
        return date(y, m, d)
    except ValueError:
        return None


def date_readings(text: str) -> set[date]:
    found: set[date] = set()
    low = text.lower()
    # Digit lookarounds, not \b: space-stripped transcriptions glue labels to values
    # ("Date24/01/2012"), and \b finds no boundary between "e" and "2".
    for a, b, c in re.findall(r"(?<!\d)(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})(?!\d)", low):
        x, y, z = int(a), int(b), int(c)
        candidates: Iterable = (
            [(x, y, z)] if len(a) == 4 else [(z, y, x), (z, x, y)]   # Y-M-D | D-M-Y, M-D-Y
        )
        for yy, mm, dd in candidates:
            if (d := _valid(yy, mm, dd)) is not None:
                found.add(d)
    month = r"([a-z]{3})[a-z]*\.?"
    for dd, mon, yy in re.findall(rf"\b(\d{{1,2}})\s*{month}\s*,?\s*(\d{{2,4}})\b", low):
        if mon in _MONTHS and (d := _valid(int(yy), _MONTHS[mon], int(dd))):
            found.add(d)
    for mon, dd, yy in re.findall(rf"\b{month}\s*(\d{{1,2}})\s*,?\s*(\d{{2,4}})\b", low):
        if mon in _MONTHS and (d := _valid(int(yy), _MONTHS[mon], int(dd))):
            found.add(d)
    return found


def _date_grounded(value: Any, text: str) -> bool:
    if not isinstance(value, date):
        try:
            value = date.fromisoformat(str(value))
        except ValueError:
            return False
    return value in date_readings(text)


# --- currency ----------------------------------------------------------------
def _currency_grounded(value: Any, text: str) -> bool:
    code = str(value).strip().upper()
    low = text.lower()
    if re.search(rf"(?<![a-z]){re.escape(code.lower())}(?![a-z])", low):
        return True
    for symbol, codes in _SYMBOLS.items():
        if code not in codes:
            continue
        pattern = re.escape(symbol) if not symbol.isalpha() else rf"(?<![a-z]){symbol}(?![a-z])"
        if re.search(pattern, low):
            return True
    return False


# --- strings -----------------------------------------------------------------
def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _string_grounded(value: Any, text: str) -> bool:
    """Token-sequence containment: whitespace, case and punctuation are ignored, but a
    value must match whole tokens ("Kopi" does not ground in "Kopitiam")."""
    needle, hay = _tokens(str(value)), _tokens(text)
    if not needle:
        return False
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))
