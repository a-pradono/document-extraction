"""Aggregate per-field confidence + abstention (Phase 4).

Design notes (put these in the README):

The score pools FOUR signals rather than diffing two models pairwise. Ranked by how
much they actually know:

  1. `validated`  -- deterministic arithmetic/format checks (subtotal+tax==total, date
                     parses, currency is plausible, line items sum). STRONGEST: it is
                     model-free, so it is genuinely independent of who read the digits.
                     This matters because in Option 1 the LLM reads the VLM's
                     transcription, so A and B are only PARTIALLY independent --
                     reading errors propagate. Validation is what breaks that
                     correlation.
  2. `grounded`   -- does the value literally appear in the transcription? Direct
                     anti-hallucination check.
  3. `agreement`  -- do Model A (VLM, image) and Model B (LLM, transcription) return
                     the same normalized value?
  4. `self_report`-- the model's stated certainty. WEAKEST: LLM self-reported
                     confidence is well known to be overconfident, because the model
                     emits a fluent-sounding number rather than estimating its own
                     error rate. Included, but never allowed to dominate.

Weights are FITTED on a held-out calibration split (see `fit`), not hand-set, so the
learned coefficients are themselves a reportable finding. Falls back to sane fixed
weights when unfitted, so the pipeline works before you have calibration data.

Calibration is per FIELD GROUP (string / money / date / currency), not global:
`merchant_name` and `total` have very different difficulty profiles, and one global
threshold over-abstains on easy fields while under-abstaining on hard ones.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path
from typing import Optional

from eval.metrics import (
    CURRENCY_FIELDS,
    DATE_FIELDS,
    MONEY_FIELDS,
    STRING_FIELDS,
    norm_money,
    present,
)

SIGNALS = ("agreement", "grounded", "validated", "self_report")

# Fallback weights, used until `fit()` has run. Deliberately ordered by the
# signal-strength reasoning above: deterministic checks outrank self-report.
DEFAULT_WEIGHTS = {"agreement": 0.30, "grounded": 0.30, "validated": 0.30, "self_report": 0.10}
DEFAULT_BIAS = 0.0

# Field groups for per-group calibration.
GROUPS = {
    "string": set(STRING_FIELDS),
    "money": set(MONEY_FIELDS),
    "date": set(DATE_FIELDS),
    "currency": set(CURRENCY_FIELDS),
    # The line-item table as one unit: its signals and its failure profile (a dropped or
    # merged row) look nothing like a scalar's, so it is calibrated on its own.
    "table": {"line_items"},
}


def group_of(field: str) -> str:
    for name, members in GROUPS.items():
        if field in members:
            return name
    return "string"


@dataclass
class FieldSignals:
    """Raw signal vector for one field, each in [0, 1]."""

    agreement: float = 0.0
    grounded: float = 0.0
    validated: float = 0.5   # 0.5 = "no applicable rule", neither pass nor fail
    self_report: float = 0.5

    def vector(self) -> list[float]:
        return [getattr(self, s) for s in SIGNALS]


# --- deterministic validation (the model-free signal) ------------------------
RECONCILIATION_FIELDS = ("subtotal", "service_charge", "discount", "tax", "total")


def money_tolerance(*amounts) -> Decimal:
    """How far a printed amount may be from the arithmetic and still reconcile.

    A flat 0.02 rejected legitimate cash rounding on large-denomination currencies: a CORD
    receipt prints 365,000 for 316,017 + 48,982.68 = 364,999.68. At or above 1,000 a
    receipt's smallest meaningful unit is 1, so the tolerance is 1. Below that it stays
    0.02, which still catches a misread cent.
    """
    largest = max((abs(a) for a in amounts if a is not None), default=Decimal(0))
    return Decimal("1") if largest >= 1000 else Decimal("0.02")


def expected_total(sub, service, discount, tax) -> Decimal:
    zero = Decimal(0)
    return sub + (service or zero) - abs(discount or zero) + (tax or zero)


CASH_ROUNDING_STEPS = (Decimal(100), Decimal(500), Decimal(1000))


def reconciles(expected: Decimal, printed: Decimal, tolerance: Decimal) -> bool:
    """Does a printed total agree with the arithmetic?

    Within tolerance, or equal to the computed amount rounded to a cash step. Receipts in
    large-denomination currencies round the payable total: CORD prints 334,000 for
    334,011 and 680,500 for 680,339. That rule is exact, unlike a looser tolerance, which
    would also accept a misread hundreds digit, since a misread almost never lands on a
    rounding of the true amount.
    """
    if abs(expected - printed) <= tolerance:
        return True
    if printed < 1000:
        return False
    for step in CASH_ROUNDING_STEPS:
        down = (expected / step).to_integral_value(rounding="ROUND_FLOOR") * step
        candidates = {down, down + step}                 # nearest either way, or rounded up
        if printed in candidates and abs(printed - expected) < step:
            return True
    return False


def validate_arithmetic(receipt, tolerance: Optional[str] = None) -> dict[str, float]:
    """Return per-field validation scores in [0,1]. A field that can't be checked is left out.

    Checks that need no model and cannot be bluffed:
      - subtotal + service_charge - discount + tax == total
      - sum(line_item.line_total) == subtotal (or the pre-charge amount implied by total)
      - quantity * unit_price == line_total, per row
    """
    sub, service, discount, tax, total = (
        norm_money(getattr(receipt, f, None)) for f in RECONCILIATION_FIELDS)
    tol = norm_money(tolerance) if tolerance else money_tolerance(sub, total)
    scores: dict[str, float] = {}

    if total is not None and sub is not None:
        ok = reconciles(expected_total(sub, service, discount, tax), total, tol)
        # An arithmetic identity implicates every field in it, so credit/blame all.
        for f in RECONCILIATION_FIELDS:
            if present(getattr(receipt, f, None)):
                scores[f] = 1.0 if ok else 0.0

    items = getattr(receipt, "line_items", []) or []

    # Per row: quantity x unit_price == line_total, wherever all three were read. This was
    # promised by the docstring above but never implemented.
    row_checks = []
    for item in items:
        q, u, t = norm_money(item.quantity), norm_money(item.unit_price), norm_money(item.line_total)
        if None not in (q, u, t):
            row_checks.append(abs(q * u - t) <= tol)
    if row_checks:
        scores["line_items"] = 1.0 if all(row_checks) else 0.0

    item_totals = [norm_money(i.line_total) for i in items]
    if items and all(t is not None for t in item_totals):
        line_sum = sum(item_totals)
        if sub is not None:
            anchor = sub
        elif total is not None:
            # No printed subtotal: the rows should sum to the total minus everything that
            # was added on top of them, plus whatever was taken off.
            anchor = total - (service or 0) + abs(discount or 0) - (tax or 0)
        else:
            anchor = None
        if anchor is not None:
            ok = abs(line_sum - anchor) <= tol
            # Both checks must pass: a table whose rows multiply out but don't sum to the
            # subtotal has lost or invented a row.
            scores["line_items"] = min(scores.get("line_items", 1.0), 1.0 if ok else 0.0)
            # Reinforce the anchor field: it now has two independent checks.
            anchor_field = "subtotal" if sub is not None else "total"
            if anchor_field in scores:
                scores[anchor_field] = min(scores[anchor_field], 1.0 if ok else 0.0)

    if present(getattr(receipt, "date", None)):
        scores["date"] = 1.0  # Pydantic already parsed it to a real date
    cur = getattr(receipt, "currency", None)
    if present(cur):
        scores["currency"] = 1.0 if len(str(cur).strip()) == 3 else 0.0

    scores.update(_invoice_checks(receipt))
    return scores


def _invoice_checks(doc) -> dict[str, float]:
    """Two model-free checks that only make sense for invoices.

    - due_date before issue_date is impossible on a real invoice, so it means a misread
      or swapped date. Both dates are blamed, since either could be the wrong one.
    - vendor and customer with the same name means role resolution failed: one party's
      details were copied into the other. Both names are blamed. Distinct names earn no
      credit, since two different names can still be the wrong way round.
    """
    scores: dict[str, float] = {}
    issue, due = getattr(doc, "issue_date", None), getattr(doc, "due_date", None)
    for name, value in (("issue_date", issue), ("due_date", due)):
        if value is not None:
            scores[name] = 1.0
    if issue is not None and due is not None and due < issue:
        scores["issue_date"] = scores["due_date"] = 0.0

    vendor, customer = getattr(doc, "vendor", None), getattr(doc, "customer", None)
    if vendor is not None and customer is not None and vendor.name and customer.name:
        if norm_party(vendor.name) == norm_party(customer.name):
            scores["vendor.name"] = scores["customer.name"] = 0.0
    return scores


def norm_party(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


# --- logistic model ----------------------------------------------------------
def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class GroupModel:
    weights: dict[str, float]
    bias: float
    n_train: int = 0
    fitted: bool = False

    def score(self, sig: FieldSignals) -> float:
        z = self.bias + sum(self.weights[s] * getattr(sig, s) for s in SIGNALS)
        # Unfitted fallback weights are a convex combination already in [0,1];
        # passing them through the sigmoid would squash everything toward 0.5.
        return z if not self.fitted else _sigmoid(z)


class ConfidenceModel:
    """Per-field-group logistic calibration of the four signals."""

    def __init__(self, groups: Optional[dict[str, GroupModel]] = None):
        self.groups = groups or {
            g: GroupModel(weights=dict(DEFAULT_WEIGHTS), bias=DEFAULT_BIAS) for g in GROUPS
        }

    def score_field(self, field: str, sig: FieldSignals) -> float:
        model = self.groups.get(group_of(field))
        if model is None:
            model = GroupModel(dict(DEFAULT_WEIGHTS), DEFAULT_BIAS)
        return max(0.0, min(1.0, model.score(sig)))

    # --- fitting -------------------------------------------------------------
    def fit(
        self,
        samples: list[tuple[str, FieldSignals, bool]],
        *,
        lr: float = 0.1,
        epochs: int = 600,
        l2: float = 0.01,
    ) -> dict:
        """Fit per-group logistic regression on (field, signals, was_correct).

        Plain gradient descent -- ~480 observations at n=60 docs is small, so a
        1-2 parameter-per-signal logistic model is the right capacity. Isotonic
        regression would overfit at this size. L2 keeps coefficients tame when a
        group has few samples.

        IMPORTANT: pass ONLY calibration-split samples here. Fitting on the
        reported test set inflates the numbers you publish.
        """
        report: dict = {}
        by_group: dict[str, list[tuple[FieldSignals, bool]]] = {g: [] for g in GROUPS}
        for field, sig, ok in samples:
            by_group[group_of(field)].append((sig, ok))

        for gname, rows in by_group.items():
            if len(rows) < 12:  # too few to fit responsibly -> keep defaults
                report[gname] = {"fitted": False, "n": len(rows), "reason": "insufficient samples"}
                continue

            w = {s: 0.0 for s in SIGNALS}
            b = 0.0
            n = len(rows)
            for _ in range(epochs):
                gw = {s: 0.0 for s in SIGNALS}
                gb = 0.0
                for sig, ok in rows:
                    p = _sigmoid(b + sum(w[s] * getattr(sig, s) for s in SIGNALS))
                    err = p - (1.0 if ok else 0.0)
                    for s in SIGNALS:
                        gw[s] += err * getattr(sig, s)
                    gb += err
                for s in SIGNALS:
                    w[s] -= lr * (gw[s] / n + l2 * w[s])
                b -= lr * (gb / n)

            self.groups[gname] = GroupModel(weights=w, bias=b, n_train=n, fitted=True)
            report[gname] = {
                "fitted": True,
                "n": n,
                "weights": {s: round(w[s], 4) for s in SIGNALS},
                "bias": round(b, 4),
            }
        return report

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({g: asdict(m) for g, m in self.groups.items()}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceModel":
        raw = json.loads(Path(path).read_text())
        return cls({g: GroupModel(**m) for g, m in raw.items()})


# --- threshold selection -----------------------------------------------------
def risk_coverage_curve(scored: list[tuple[float, bool]], steps: int = 41) -> list[dict]:
    """Coverage/precision at each threshold. Read the operating point off this
    curve against a target precision -- don't pick a threshold by intuition."""
    if not scored:
        return []
    out = []
    for i in range(steps):
        t = i / (steps - 1)
        kept = [ok for c, ok in scored if c >= t]
        if not kept:
            continue
        out.append({
            "threshold": round(t, 3),
            "coverage": round(len(kept) / len(scored), 4),
            "precision": round(sum(kept) / len(kept), 4),
        })
    return out


def pick_threshold(scored: list[tuple[float, bool]], target_precision: float = 0.95) -> dict:
    """Lowest threshold meeting target precision -> maximum coverage at that quality."""
    curve = risk_coverage_curve(scored)
    feasible = [p for p in curve if p["precision"] >= target_precision]
    if not feasible:
        best = max(curve, key=lambda p: p["precision"]) if curve else {}
        return {"threshold": best.get("threshold", 1.0), "achieved": best,
                "met_target": False, "target_precision": target_precision}
    best = max(feasible, key=lambda p: p["coverage"])
    return {"threshold": best["threshold"], "achieved": best,
            "met_target": True, "target_precision": target_precision}
