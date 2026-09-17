"""Confidence cascade (Phase 6 optimization).

    cheap:     Flash-Lite VLM, one call -> fields + transcription
               + model-free checks (arithmetic, grounding) + self-report
    accept     if nothing below is triggered
    escalate:  the full two-model pipeline (Flash VLM + different-family LLM)

Why this shape: most receipts are easy, and the full path pays for a stronger VLM AND a
second model on every one of them. The cascade spends that only where the cheap read
gives a concrete reason for doubt.

Escalation triggers, strongest first. The first two are model-free, so they don't depend
on the cheap model grading its own work:
  arithmetic_failed    subtotal + tax != total, or the line items don't sum
  ungrounded:<field>   a present value that cannot be read from the transcription
  low_confidence:<f>   the fitted confidence model scores a field below the bar
  empty_extraction     no total and no line items: nothing to validate against
  cheap_path_failed    the cheap call errored or returned an invalid receipt

HONEST COSTS, not hidden ones:
  - An escalated document pays for BOTH paths. The cheap call is sunk, so it is counted
    on that document's bill. If the escalation rate is high, the cascade costs MORE than
    the baseline, and the before/after table has to show that.
  - The cheap path has no second model, so the agreement signal is neutral (0.5) when
    it is scored. The confidence model was fitted on full-pipeline signals, so these
    cheap-path scores are only a gate. Whether the gate is any good is decided by the
    accuracy delta on the eval set, not by assumption.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from extractor.confidence import ConfidenceModel, FieldSignals
from extractor.cost import stage_usage
from extractor.extract import prepare_pages
from extractor.output import salvage_receipt
from extractor.pipeline import (
    PipelineResult,
    _self_report,
    build_signals,
    extract_with_confidence,
    fields_of,
    run_vlm,
)
from extractor.provider import Provider
from extractor.schema import DOCUMENT_MODELS
from eval.metrics import MONEY_FIELDS, get_field, present

CONSISTENCY_FIELDS = ("issue_date", "due_date", "vendor.name", "customer.name")


def escalation_reasons(
    receipt: BaseModel,
    signals: dict[str, FieldSignals],
    confidence: dict[str, float],
    escalate_below: float,
) -> list[str]:
    """Why the cheap result can't be trusted as-is. Empty list means accept it."""
    reasons: list[str] = []

    def has(f: str) -> bool:
        value = get_field(receipt, f)
        return present(value) and value != []          # an empty table is absent too

    if any(has(f) and signals[f].validated == 0.0
           for f in (*MONEY_FIELDS, "line_items") if f in signals):
        reasons.append("arithmetic_failed")

    # Invoice-only: due date before issue date, or the same name on both parties.
    if any(has(f) and signals[f].validated == 0.0 for f in CONSISTENCY_FIELDS if f in signals):
        reasons.append("consistency_failed")

    for f, sig in signals.items():
        if has(f) and sig.grounded == 0.0:
            reasons.append(f"ungrounded:{f}")

    # Absent fields are skipped: their confidence reflects "is null the right answer",
    # and escalating a receipt because it has no payment_method would escalate almost
    # everything while buying nothing.
    for f, c in confidence.items():
        if has(f) and c < escalate_below:
            reasons.append(f"low_confidence:{f}")

    if not present(getattr(receipt, "total", None)) and not receipt.line_items:
        reasons.append("empty_extraction")
    return reasons


def _cheap_provider() -> Provider:
    return Provider(
        base_url=os.environ.get("CASCADE_BASE_URL") or os.environ["VLM_A_BASE_URL"],
        api_key=os.environ.get("CASCADE_API_KEY") or os.environ.get("VLM_A_API_KEY", ""),
        model=os.environ["CASCADE_MODEL"],
    )


def extract_cascade(
    path: str | Path,
    *,
    threshold: float = 0.5,
    model: Optional[ConfidenceModel] = None,
    use_cache: bool = True,
    escalate_below: Optional[float] = None,
    doc_type: str = "receipt",
) -> PipelineResult:
    started = time.perf_counter()
    conf_model = model or ConfidenceModel()
    if escalate_below is None:
        # `or`, not a .get() default: .env.example ships the key empty, and float("")
        # would fail every document with ValueError.
        escalate_below = float(os.environ.get("CASCADE_ESCALATE_BELOW") or threshold)

    reasons: list[str]
    # Set before the try: a cheap call that SUCCEEDED but whose output is unusable was
    # still paid for, and must stay on the escalated document's bill.
    cheap_stage = None
    try:
        pages = prepare_pages(path)
        raw, usage, cached = run_vlm(pages, _cheap_provider(), use_cache, doc_type=doc_type)
        cheap_stage = stage_usage("cascade", usage, cached)
        salvaged = salvage_receipt(fields_of(raw, doc_type), DOCUMENT_MODELS[doc_type])
        receipt = salvaged.document
        transcription = str(raw.get("transcription", "") or "")

        signals = build_signals(receipt, None, transcription, _self_report(raw))
        confidence: dict[str, float] = {}
        for f, sig in signals.items():
            sig.agreement = 0.5            # no second model on the cheap path: unknown
            confidence[f] = round(conf_model.score_field(f, sig), 4)
        reasons = escalation_reasons(receipt, signals, confidence, escalate_below)
        # Any invalid value in the cheap read is a reason to let the stronger path try.
        reasons += [f"invalid_output:{f}" for f in salvaged.invalid]
    except Exception as exc:  # noqa: BLE001 — a failed cheap read escalates, never fails the doc
        reasons = [f"cheap_path_failed: {type(exc).__name__}: {exc}"]

    if not reasons:
        return PipelineResult(
            document=receipt,
            document_b=None,
            transcription=transcription,
            confidence=confidence,
            abstained=[f for f, c in confidence.items() if c < threshold],
            signals=signals,
            latency_s=round(cheap_stage.latency_s, 3),
            wall_s=round(time.perf_counter() - started, 3),
            tokens={"cascade": usage},
            stages=[cheap_stage],
            cost_usd=round(cheap_stage.cost_usd, 8),
            cascade={"escalated": False, "reasons": []},
        )

    full = extract_with_confidence(path, threshold=threshold, model=model, use_cache=use_cache,
                                   doc_type=doc_type)
    if cheap_stage is not None:
        full.stages.insert(0, cheap_stage)        # sunk cost: the escalated doc pays both
    full.latency_s = round(sum(s.latency_s for s in full.stages), 3)
    full.cost_usd = round(sum(s.cost_usd for s in full.stages), 8)
    full.wall_s = round(time.perf_counter() - started, 3)
    full.cascade = {"escalated": True, "reasons": reasons}
    return full
