"""Metrics (Phase 3).

Pure, testable functions. The order that matters:
  1. NORMALIZE before comparing (dates->ISO, money->Decimal, strip currency/whitespace)
     so we score semantic correctness, not string noise.
  2. Compare at two levels: exact (raw) and normalized. The gap between them is
     format sensitivity.
  3. Label each comparison with the error taxonomy.
  4. Aggregate into per-field P/R/F1, Document Accuracy Rate, line-item F1, taxonomy
     counts, hallucination proxy, and (when confidence is present) calibration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from difflib import SequenceMatcher
from typing import Any, Optional

from pydantic import BaseModel

from extractor.schema import Invoice, LineItem, Receipt
from eval import taxonomy

# --- field groups ------------------------------------------------------------
# Scalar fields per document type. Nested invoice parties are flattened to dotted names
# ("vendor.name") so every comparison, taxonomy label, confidence signal and regression
# gate works on them unchanged. Crucially, the swapped_field detector sees
# vendor.name and customer.name as ordinary sibling fields, so a role swap is caught.
RECEIPT_STRING_FIELDS = ("merchant_name", "merchant_address", "payment_method")
INVOICE_STRING_FIELDS = ("invoice_number", "purchase_order",
                         "vendor.name", "vendor.address", "vendor.tax_id",
                         "customer.name", "customer.address", "customer.tax_id")
RECEIPT_DATE_FIELDS = ("date",)
INVOICE_DATE_FIELDS = ("issue_date", "due_date")
CURRENCY_FIELDS = ("currency",)
RECEIPT_MONEY_FIELDS = ("subtotal", "service_charge", "discount", "tax", "total")
INVOICE_MONEY_FIELDS = ("subtotal", "tax", "total")

FIELDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "receipt": RECEIPT_STRING_FIELDS + RECEIPT_DATE_FIELDS + CURRENCY_FIELDS + RECEIPT_MONEY_FIELDS,
    "invoice": INVOICE_STRING_FIELDS + INVOICE_DATE_FIELDS + CURRENCY_FIELDS + INVOICE_MONEY_FIELDS,
}

# Unions, for normalization and grouping, which depend only on what a field IS.
STRING_FIELDS = RECEIPT_STRING_FIELDS + INVOICE_STRING_FIELDS
DATE_FIELDS = RECEIPT_DATE_FIELDS + INVOICE_DATE_FIELDS
MONEY_FIELDS = RECEIPT_MONEY_FIELDS
SCALAR_FIELDS = FIELDS_BY_TYPE["receipt"]          # receipt default, kept for receipt callers


def doc_type_of_model(doc: BaseModel) -> str:
    return "invoice" if isinstance(doc, Invoice) else "receipt"


def scalar_fields(doc_or_type: BaseModel | str) -> tuple[str, ...]:
    doc_type = doc_or_type if isinstance(doc_or_type, str) else doc_type_of_model(doc_or_type)
    return FIELDS_BY_TYPE[doc_type]


def get_field(doc: BaseModel, name: str) -> Any:
    """Read a possibly dotted field ("vendor.name") from a document."""
    value: Any = doc
    for part in name.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value

CURRENCY_SYMBOLS = {"$": "USD", "rp": "IDR", "€": "EUR", "£": "GBP", "¥": "JPY"}


# --- normalization -----------------------------------------------------------
def norm_string(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower())


def aggr_string(v: Any) -> str:
    """Aggressive: alphanumerics only. Used to separate format_error from wrong_value."""
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def norm_money(v: Any) -> Optional[Decimal]:
    try:
        d = Decimal(str(v).replace(",", "").replace(" ", ""))
        return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def norm_qty(v: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(v).replace(",", "").strip()).normalize()
    except (InvalidOperation, ValueError):
        return None


def norm_currency(v: Any) -> str:
    s = str(v).strip().lower()
    return CURRENCY_SYMBOLS.get(s, s.upper())


def norm_date(v: Any) -> str:
    return v.isoformat() if isinstance(v, date) else str(v).strip()


def _normalize(field_name: str, v: Any):
    if field_name in MONEY_FIELDS:
        return norm_money(v)
    if field_name in CURRENCY_FIELDS:
        return norm_currency(v)
    if field_name in DATE_FIELDS:
        return norm_date(v)
    return norm_string(v)


def present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    return True


# --- comparison primitives ---------------------------------------------------
@dataclass
class FieldComparison:
    field: str
    gt: Any
    pred: Any
    label: str
    exact: bool
    normalized: bool
    gt_present: bool
    pred_present: bool


def compare_scalar(field_name: str, gt_v: Any, pred_v: Any) -> FieldComparison:
    gp, pp = present(gt_v), present(pred_v)
    exact = gp and pp and str(gt_v) == str(pred_v)
    normalized = False
    aggressive = False
    if gp and pp:
        gn, pn = _normalize(field_name, gt_v), _normalize(field_name, pred_v)
        normalized = gn is not None and gn == pn
        aggressive = normalized or (aggr_string(gt_v) == aggr_string(pred_v))
    label = taxonomy.classify(gp, pp, normalized, aggressive)
    return FieldComparison(field_name, gt_v, pred_v, label, exact, normalized, gp, pp)


# --- line items --------------------------------------------------------------
def _item_similarity(a: LineItem, b: LineItem) -> float:
    da, db = norm_string(a.description or ""), norm_string(b.description or "")
    desc = SequenceMatcher(None, da, db).ratio() if (da or db) else 0.0
    la, lb = norm_money(a.line_total), norm_money(b.line_total)
    amount = 1.0 if (la is not None and la == lb) else 0.0
    return 0.7 * desc + 0.3 * amount


def match_line_items(gt: list[LineItem], pred: list[LineItem], threshold: float = 0.5):
    """Greedy one-to-one matching by similarity. Returns list of (gt_idx, pred_idx)."""
    scored = [
        (_item_similarity(g, p), i, j)
        for i, g in enumerate(gt)
        for j, p in enumerate(pred)
    ]
    scored.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[tuple[int, int]] = []
    for score, i, j in scored:
        if score < threshold or i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matches.append((i, j))
    return matches


def score_line_items(gt: list[LineItem], pred: list[LineItem]) -> dict:
    matches = match_line_items(gt, pred)
    n_gt, n_pred, n_match = len(gt), len(pred), len(matches)
    precision = n_match / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = n_match / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    f1 = _f1(precision, recall)

    sub = ("description", "quantity", "unit_price", "line_total")
    correct = {k: 0 for k in sub}
    for gi, pj in matches:
        g, p = gt[gi], pred[pj]
        for k in sub:
            gv, pv = getattr(g, k), getattr(p, k)
            if k == "description":
                ok = present(gv) == present(pv) and (
                    not present(gv) or norm_string(gv) == norm_string(pv)
                )
            else:
                gn = norm_money(gv) if k != "quantity" else norm_qty(gv)
                pn = norm_money(pv) if k != "quantity" else norm_qty(pv)
                ok = (gn is None and pn is None) or (gn is not None and gn == pn)
            correct[k] += int(ok)
    field_accuracy = {k: (correct[k] / n_match if n_match else None) for k in sub}
    perfect = n_gt == n_pred == n_match and all(
        (field_accuracy[k] == 1.0) for k in sub if field_accuracy[k] is not None
    )
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "matched": n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "field_accuracy": field_accuracy,
        "perfect": perfect,
    }


# --- document-level ----------------------------------------------------------
@dataclass
class DocResult:
    doc_id: str
    comparisons: list[FieldComparison]
    line_items: dict
    document_correct: bool
    latency_s: Optional[float] = None
    tokens: Optional[dict] = None          # filled in Phase 5
    cost_usd: Optional[float] = None       # filled in Phase 5
    confidence: dict = dc_field(default_factory=dict)  # field -> conf, Phase 4


def _apply_swapped(comparisons: list[FieldComparison], gt: BaseModel) -> None:
    """Relabel a wrong/missing field as swapped_field when the predicted value
    actually belongs to a different field (a common, diagnosable error).

    Works on flattened party fields, so an invoice whose vendor and customer names came
    back reversed is labeled swapped_field on both, which is role resolution failing."""
    gt_norm = {
        f: _normalize(f, get_field(gt, f))
        for f in scalar_fields(gt)
        if present(get_field(gt, f))
    }
    for c in comparisons:
        if c.label not in (taxonomy.WRONG_VALUE, taxonomy.SPURIOUS):
            continue
        if not c.pred_present:
            continue
        pv = _normalize(c.field, c.pred)
        for other, gv in gt_norm.items():
            if other != c.field and pv is not None and pv == gv:
                c.label = taxonomy.SWAPPED_FIELD
                break


def compare_document(
    doc_id: str,
    pred: BaseModel,
    gt: BaseModel,
    latency_s: Optional[float] = None,
    confidence: Optional[dict] = None,
) -> DocResult:
    if type(pred) is not type(gt):
        raise TypeError(f"{doc_id}: prediction is {type(pred).__name__}, ground truth is "
                        f"{type(gt).__name__}; a document must be scored against its own type")
    comparisons = [compare_scalar(f, get_field(gt, f), get_field(pred, f)) for f in scalar_fields(gt)]
    _apply_swapped(comparisons, gt)
    li = score_line_items(gt.line_items, pred.line_items)
    scalars_ok = all(c.label == taxonomy.CORRECT for c in comparisons)
    document_correct = scalars_ok and li["perfect"]
    return DocResult(
        doc_id=doc_id,
        comparisons=comparisons,
        line_items=li,
        document_correct=document_correct,
        latency_s=latency_s,
        confidence=confidence or {},
    )


# --- aggregation -------------------------------------------------------------
def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def aggregate(results: list[DocResult]) -> dict:
    """Aggregate over whatever fields the documents were scored on.

    Callers must not pool document types into one headline (receipts and invoices have
    different fields and difficulty); eval/run.py aggregates each type separately."""
    fields: list[str] = []
    for r in results:
        for c in r.comparisons:
            if c.field not in fields:
                fields.append(c.field)
    per_field: dict[str, dict] = {}
    for f in fields:
        tp_n = tp_e = pred_pos = gt_pos = 0
        for r in results:
            c = next((c for c in r.comparisons if c.field == f), None)
            if c is None:
                continue
            pred_pos += int(c.pred_present)
            gt_pos += int(c.gt_present)
            if c.gt_present and c.pred_present:
                tp_n += int(c.label == taxonomy.CORRECT)
                tp_e += int(c.exact)
        if gt_pos == 0 and pred_pos == 0:
            # Nothing in the ground truth and nothing predicted: the field is correctly
            # blank everywhere, and P/R/F1 are undefined, not zero. Reporting 0.000 here
            # would read as total failure and drag macro F1 down for a field the system
            # got entirely right (e.g. payment_method on a set that never labels it).
            per_field[f] = {"support": 0, "predicted": 0, "precision": None,
                            "recall": None, "f1": None, "exact_f1": None}
            continue
        p_n = tp_n / pred_pos if pred_pos else 0.0
        r_n = tp_n / gt_pos if gt_pos else 0.0
        p_e = tp_e / pred_pos if pred_pos else 0.0
        r_e = tp_e / gt_pos if gt_pos else 0.0
        per_field[f] = {
            "support": gt_pos,
            "predicted": pred_pos,
            "precision": round(p_n, 4),
            "recall": round(r_n, 4),
            "f1": round(_f1(p_n, r_n), 4),
            "exact_f1": round(_f1(p_e, r_e), 4),
        }

    # micro over all scalar fields
    micro_tp = sum(
        int(c.label == taxonomy.CORRECT and c.gt_present and c.pred_present)
        for r in results for c in r.comparisons
    )
    micro_pred = sum(int(c.pred_present) for r in results for c in r.comparisons)
    micro_gt = sum(int(c.gt_present) for r in results for c in r.comparisons)
    micro_p = micro_tp / micro_pred if micro_pred else 0.0
    micro_r = micro_tp / micro_gt if micro_gt else 0.0
    measurable = [v["f1"] for v in per_field.values() if v["f1"] is not None]
    macro_f1 = sum(measurable) / len(measurable) if measurable else 0.0

    labels = [c.label for r in results for c in r.comparisons]
    tax_counts = taxonomy.rollup(labels)
    hallucination_rate = (tax_counts[taxonomy.SPURIOUS] / micro_pred) if micro_pred else 0.0

    dar = sum(int(r.document_correct) for r in results) / len(results) if results else 0.0
    li_f1 = (
        sum(r.line_items["f1"] for r in results) / len(results) if results else 0.0
    )

    latencies = [r.latency_s for r in results if r.latency_s is not None]
    latency = {}
    if latencies:
        latencies.sort()
        latency = {
            "p50_s": round(latencies[len(latencies) // 2], 3),
            "p95_s": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3),
            "mean_s": round(sum(latencies) / len(latencies), 3),
        }

    return {
        "n_docs": len(results),
        "document_accuracy_rate": round(dar, 4),
        "micro_f1": round(_f1(micro_p, micro_r), 4),
        "macro_f1": round(macro_f1, 4),
        "line_item_f1": round(li_f1, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "per_field": per_field,
        "error_taxonomy": tax_counts,
        "latency": latency,
        "calibration": calibration(results),
    }


def auc(pairs: list[tuple[float, bool]]) -> Optional[float]:
    """ROC AUC via the Mann-Whitney statistic, with tie handling.

    DISCRIMINATION: does the score rank correct fields above incorrect ones?
    ~0.5 means the score carries no information -- and no amount of rescaling can
    fix that, you have to change the signals. Distinct from calibration below.
    """
    pos = [c for c, ok in pairs if ok]
    neg = [c for c, ok in pairs if not ok]
    if not pos or not neg:
        return None  # undefined when one class is absent
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(wins / (len(pos) * len(neg)), 4)


def ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> Optional[float]:
    """Expected Calibration Error: mean |confidence - accuracy| weighted by bin size.

    CALIBRATION: do the numbers mean what they say -- is 0.9 right ~90% of the time?
    High ECE with good AUC is the fixable case (rescale on a held-out split).
    """
    if not pairs:
        return None
    total = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        chunk = [(c, ok) for c, ok in pairs if (lo <= c < hi or (i == n_bins - 1 and c == 1.0))]
        if not chunk:
            continue
        acc = sum(ok for _, ok in chunk) / len(chunk)
        conf = sum(c for c, _ in chunk) / len(chunk)
        total += (len(chunk) / len(pairs)) * abs(conf - acc)
    return round(total, 4)


def confidence_observations(results: list[DocResult]) -> list[tuple[str, float, bool]]:
    """Every (field, confidence, was_correct) the scorecard can judge confidence on.

    The single definition used by calibration, the abstention report and fitting, so the
    three can never disagree about what counts. The line-item table is one observation,
    correct only if the whole table is perfect: strict, but an abstention on a table with
    one dropped row is exactly the behaviour wanted.
    """
    obs: list[tuple[str, float, bool]] = []
    for r in results:
        for c in r.comparisons:
            if c.field in r.confidence:
                obs.append((c.field, r.confidence[c.field], c.label == taxonomy.CORRECT))
        if "line_items" in r.confidence:
            obs.append(("line_items", r.confidence["line_items"], bool(r.line_items.get("perfect"))))
    return obs


def calibration(results: list[DocResult], bins=(0.0, 0.5, 0.7, 0.9, 1.01)) -> dict:
    """Confidence quality: discrimination (AUC), calibration (ECE), a reliability
    diagram, and the risk-coverage tradeoff. Dormant until the pipeline supplies
    per-field confidence.

    Reported per field GROUP as well as overall, because a single global threshold
    over-abstains on easy fields and under-abstains on hard ones.
    """
    pairs: list[tuple[float, bool]] = []
    by_field: dict[str, list[tuple[float, bool]]] = {}
    for field, conf, ok in confidence_observations(results):
        pairs.append((conf, ok))
        by_field.setdefault(field, []).append((conf, ok))
    if not pairs:
        return {"available": False}

    binned = []
    for lo, hi in zip(bins, bins[1:]):
        chunk = [ok for conf, ok in pairs if lo <= conf < hi]
        if chunk:
            binned.append({"range": f"[{lo},{hi})", "n": len(chunk),
                           "accuracy": round(sum(chunk) / len(chunk), 4)})

    coverage = []
    for t in (0.5, 0.7, 0.9):
        kept = [ok for conf, ok in pairs if conf >= t]
        if kept:
            coverage.append({"threshold": t, "coverage": round(len(kept) / len(pairs), 4),
                             "accuracy": round(sum(kept) / len(kept), 4)})

    # Reliability diagram: 10 equal-width bins, mean confidence vs observed accuracy.
    reliability = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        chunk = [(c, ok) for c, ok in pairs if (lo <= c < hi or (i == 9 and c == 1.0))]
        if chunk:
            reliability.append({
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": len(chunk),
                "mean_confidence": round(sum(c for c, _ in chunk) / len(chunk), 4),
                "accuracy": round(sum(ok for _, ok in chunk) / len(chunk), 4),
            })

    per_field = {
        f: {"n": len(ps), "auc": auc(ps), "ece": ece(ps)}
        for f, ps in sorted(by_field.items())
    }

    return {
        "available": True,
        "n_observations": len(pairs),
        "auc": auc(pairs),
        "ece": ece(pairs),
        "bins": binned,
        "reliability": reliability,
        "coverage_accuracy": coverage,
        "per_field": per_field,
    }
