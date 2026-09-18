"""Option 1 pipeline: VLM reader + text-LLM reasoner (Phase 4).

    image --> Model A (VLM)  --> {transcription, fields}
    transcription --> Model B (LLM) --> fields (independent re-extraction)
    + deterministic validation (Python, model-free)
    --> aggregate per-field confidence --> abstention

Why this shape (README material):
  - The VLM preserves layout, which is what receipts need: column-aligned line items
    and "which of these five numbers is the total" are layout questions. Flat OCR text
    loses that.
  - Its transcription REPLACES a separate OCR stage, so we keep a groundedness source
    without a slow local OCR dependency.
  - The text LLM does what a VLM is weakest at: cross-section reasoning and arithmetic
    reconciliation. It is a cheap text-only call.
  - KNOWN LIMITATION, stated honestly: Model B reads Model A's transcription, so the
    two are only PARTIALLY independent -- a misread digit propagates and inflates
    their agreement. This is why deterministic validation is weighted heavily: it is
    model-free and breaks that correlation. True reading independence would need a
    second VLM on the raw image (documented as an upgrade path).

Model A returns BOTH transcription and fields in one call: a second call for the
transcript would double vision-token cost for no accuracy gain.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from extractor.confidence import (
    ConfidenceModel,
    FieldSignals,
    group_of,
    validate_arithmetic,
)
from extractor.cost import StageUsage, stage_usage
from extractor.extract import (
    CURRENCY_RULE,
    INVOICE_RULES,
    NUMBER_RULES,
    page_note,
    prepare_pages,
)
from extractor.output import is_invalid, parse_or_repair, salvage_receipt
from extractor.grounding import is_grounded
from extractor.provider import Provider
from extractor.schema import DOCUMENT_MODELS
from eval.metrics import _normalize, get_field, present, scalar_fields, score_line_items

load_dotenv()

CACHE_DIR = Path(os.environ.get("CACHE_DIR", ".cache/extractions"))

VLM_SYSTEM = (
    "You are a precise document reader for receipts and invoices. Extract only what is "
    "visibly present. If a field is absent, return null. Never guess or invent values. "
    "Return a single JSON object and nothing else."
)
LLM_SYSTEM = (
    "You are a careful receipt and invoice data extractor working from a transcription. "
    "Extract only values present in the transcription. If a field is absent, return "
    "null. Never invent values. Return a single JSON object and nothing else."
)


@dataclass
class PipelineResult:
    document: BaseModel                      # Model A's fields (Receipt or Invoice): the answer
    document_b: Optional[BaseModel]          # Model B's independent re-extraction
    transcription: str
    confidence: dict[str, float] = dc_field(default_factory=dict)
    abstained: list[str] = dc_field(default_factory=list)
    signals: dict[str, FieldSignals] = dc_field(default_factory=dict)
    latency_s: float = 0.0                # model-call latency: a property of the system
    wall_s: float = 0.0                   # this run's wall clock, near 0 on cache hits
    tokens: dict = dc_field(default_factory=dict)
    stages: list[StageUsage] = dc_field(default_factory=list)
    cost_usd: float = 0.0
    errors: list[str] = dc_field(default_factory=list)
    cascade: dict = dc_field(default_factory=dict)  # Phase 6: {"escalated", "reasons"}


def _schema_prompt(doc_type: str = "receipt") -> str:
    return json.dumps(DOCUMENT_MODELS[doc_type].model_json_schema(), indent=2)


def _cache_path(tag: str, *parts: bytes | str) -> Path:
    """Key on EVERYTHING that changes the answer: input, model, and prompt.

    Keying on the image alone meant a different model (the Phase 6 cascade) or an
    edited prompt silently got the old model's cached output — so a before/after
    comparison would have measured nothing while reporting a result.
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else part.encode("utf-8"))
        h.update(b"\x00")
    return CACHE_DIR / f"{h.hexdigest()[:24]}_{tag}.json"


def _usage(resp) -> dict:
    return {
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "latency_s": resp.latency_s,
        "model": resp.model,
    }


def _read_cache(path: Path, use_cache: bool):
    if use_cache and path.exists():
        payload = json.loads(path.read_text())
        return payload["raw"], payload["usage"]
    return None


def _write_cache(path: Path, use_cache: bool, raw: dict, usage: dict) -> None:
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"raw": raw, "usage": usage}))
        tmp.replace(path)  # atomic: concurrent batch workers never read a half-written file


def _self_report(raw: dict) -> dict[str, float]:
    """Pull optional per-field self-reported certainty, if the model supplied it.
    Weakest signal -- LLM self-confidence is typically overconfident -- so a missing
    value defaults to a neutral 0.5 rather than to 1.0."""
    block = raw.get("_confidence") or raw.get("confidence") or {}
    out: dict[str, float] = {}
    if isinstance(block, dict):
        for k, v in block.items():
            try:
                out[k] = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                continue
    return out


# --- Model A: VLM reads the image -------------------------------------------
def _vlm_prompt(doc_type: str = "receipt") -> str:
    return (
        f"Read this {doc_type} and return ONE JSON object with exactly these keys:\n"
        '  "transcription": every line of text you can read, in reading order, '
        "preserving line breaks.\n"
        '  "fields": an object matching the schema below.\n'
        '  "_confidence": an object mapping each field name to your certainty 0.0-1.0 '
        '(nested fields as dotted names, e.g. "vendor.name").\n'
        "Money as numbers, dates as YYYY-MM-DD, currency as ISO-4217. Include every "
        "line item, including add-on lines printed under an item. Use null for anything "
        "not present.\n" + NUMBER_RULES + "\n" + CURRENCY_RULE + "\n"
        + (INVOICE_RULES + "\n" if doc_type == "invoice" else "")
        + "\nSchema for \"fields\":\n" + _schema_prompt(doc_type)
    )


def _vlm_max_tokens(n_pages: int) -> int:
    """3000 was too small: a 27-item receipt's transcription plus its items as JSON runs
    past it, and thinking models spend reasoning tokens from the same budget."""
    base = int(os.environ.get("VLM_MAX_TOKENS", "8192"))
    return base + (n_pages - 1) * int(os.environ.get("VLM_MAX_TOKENS_PER_EXTRA_PAGE", "4096"))


def run_vlm(pages: Path | list[Path], provider: Provider, use_cache: bool = True,
            doc_type: str = "receipt"):
    """Read every page in ONE call. Returns (raw, usage, cached)."""
    pages = [pages] if isinstance(pages, Path) else list(pages)
    prompt = page_note(len(pages)) + _vlm_prompt(doc_type)
    cache = _cache_path("vlm", *(p.read_bytes() for p in pages), provider.model, VLM_SYSTEM, prompt)
    hit = _read_cache(cache, use_cache)
    if hit:
        return (*hit, True)

    resp = provider.complete(
        [
            {"role": "system", "content": VLM_SYSTEM},
            {"role": "user", "content": [Provider.text_block(prompt),
                                         *(Provider.image_block(p) for p in pages)]},
        ],
        json_mode=True,
        max_tokens=_vlm_max_tokens(len(pages)),
    )
    usage = _usage(resp)
    raw = parse_or_repair(resp, provider, usage)
    _write_cache(cache, use_cache, raw, usage)
    return raw, usage, False


# --- Model B: text LLM re-extracts from the transcription --------------------
def _llm_prompt(transcription: str, doc_type: str = "receipt") -> str:
    return (
        f"Extract {doc_type} fields from this transcription into a JSON object matching "
        "the schema. Reason about totals: if subtotal and tax are present they should "
        "sum to the total, and line item totals should sum to the subtotal. Do not "
        "alter values to force arithmetic to balance -- report what is written. Use "
        "null for anything absent.\n" + NUMBER_RULES + "\n" + CURRENCY_RULE + "\n"
        + (INVOICE_RULES + "\n" if doc_type == "invoice" else "")
        + "\nSchema:\n" + _schema_prompt(doc_type)
        + "\n\nTranscription:\n"
        + transcription
    )


def run_llm(transcription: str, provider: Provider, use_cache: bool = True,
            doc_type: str = "receipt"):
    """Returns (raw, usage, cached)."""
    prompt = _llm_prompt(transcription, doc_type)
    cache = _cache_path("llm", provider.model, LLM_SYSTEM, prompt)
    hit = _read_cache(cache, use_cache)
    if hit:
        return (*hit, True)

    resp = provider.complete(
        [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": prompt}],
        json_mode=True,
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
    )
    usage = _usage(resp)
    raw = parse_or_repair(resp, provider, usage)
    _write_cache(cache, use_cache, raw, usage)
    return raw, usage, False


# --- signal assembly ---------------------------------------------------------
def build_signals(
    a: BaseModel,
    b: Optional[BaseModel],
    transcription: str,
    self_report: dict[str, float],
) -> dict[str, FieldSignals]:
    validation = validate_arithmetic(a)
    signals: dict[str, FieldSignals] = {}

    for f in scalar_fields(a):
        av = get_field(a, f)

        # agreement: same normalized value from the independent re-extraction
        agreement = 0.0
        if b is not None:
            bv = get_field(b, f)
            if not present(av) and not present(bv):
                agreement = 1.0                      # both say absent -> agree
            elif present(av) and present(bv):
                agreement = 1.0 if _normalize(f, av) == _normalize(f, bv) else 0.0

        # grounded: the value can be read from the transcription, compared by type —
        # a parsed Decimal against printed "53.280", a date against "12/03/2024".
        grounded = 1.0 if (present(av) and is_grounded(av, transcription, group_of(f))) else 0.0
        if not present(av):
            # Can't ground an absent value; absence is corroborated by agreement
            # instead, so stay neutral rather than penalizing.
            grounded = 0.5

        signals[f] = FieldSignals(
            agreement=agreement,
            grounded=grounded,
            validated=validation.get(f, 0.5),
            self_report=self_report.get(f, 0.5),
        )

    signals["line_items"] = FieldSignals(
        agreement=table_agreement(a.line_items, b.line_items) if b is not None else 0.0,
        grounded=table_grounding(a.line_items, transcription),
        validated=validation.get("line_items", 0.5),
        self_report=self_report.get("line_items", 0.5),
    )
    return signals


def table_agreement(a_items: list, b_items: list) -> float:
    """Row-level agreement between the two models: the line-item F1 of one table scored
    against the other (the same matcher the eval uses), so a single dropped row lowers
    agreement instead of zeroing it."""
    if not a_items and not b_items:
        return 1.0
    return round(score_line_items(a_items, b_items)["f1"], 4)


def table_grounding(items: list, transcription: str) -> float:
    """Fraction of rows whose amount AND description can both be read from the text.
    An invented row, the table's version of a hallucination, pulls this down."""
    if not items:
        return 0.5                                   # nothing to ground: neutral
    grounded = 0
    for item in items:
        amount_ok = item.line_total is None or is_grounded(item.line_total, transcription, "money")
        desc_ok = not item.description or is_grounded(item.description, transcription, "string")
        grounded += int(amount_ok and desc_ok)
    return round(grounded / len(items), 4)


def fields_of(raw: dict, doc_type: str = "receipt") -> Any:
    """The extracted fields from a model response. Models asked for {"fields": {...}}
    sometimes return the fields at the top level instead; reading only "fields" would
    turn that into an all-null document that looks like a legitimately empty one."""
    if "fields" in raw:
        return raw["fields"]
    top_level = {k: v for k, v in raw.items() if k in DOCUMENT_MODELS[doc_type].model_fields}
    return top_level or None


def extract_with_confidence(
    path: str | Path,
    *,
    threshold: float = 0.5,
    model: Optional[ConfidenceModel] = None,
    use_cache: bool = True,
    doc_type: str = "receipt",
) -> PipelineResult:
    started = time.perf_counter()
    pages = prepare_pages(path)
    conf_model = model or ConfidenceModel()
    schema = DOCUMENT_MODELS[doc_type]
    result = PipelineResult(document=schema(), document_b=None, transcription="")

    vlm = Provider(
        base_url=os.environ["VLM_A_BASE_URL"],
        api_key=os.environ.get("VLM_A_API_KEY", ""),
        model=os.environ["VLM_A_MODEL"],
    )
    raw_a, usage_a, cached_a = run_vlm(pages, vlm, use_cache, doc_type=doc_type)
    result.stages.append(stage_usage("model_a", usage_a, cached_a))
    if usage_a.get("repaired"):
        result.errors.append("model_a_output_repaired: invalid JSON fixed by one text-only call")
    result.transcription = str(raw_a.get("transcription", "") or "")
    salvaged_a = salvage_receipt(fields_of(raw_a, doc_type), schema)
    result.document = salvaged_a.document
    result.errors.extend(f"model_a_{note}" for note in salvaged_a.notes)
    self_report = _self_report(raw_a)

    # Model B is best-effort: if it fails, we degrade to VLM-only with the
    # agreement signal neutralized rather than failing the whole document.
    usage_b: dict = {}
    try:
        llm = Provider(
            base_url=os.environ["LLM_B_BASE_URL"],
            api_key=os.environ.get("LLM_B_API_KEY", ""),
            model=os.environ["LLM_B_MODEL"],
        )
        raw_b, usage_b, cached_b = run_llm(result.transcription, llm, use_cache, doc_type=doc_type)
        result.stages.append(stage_usage("model_b", usage_b, cached_b))
        # B's invalid values just become null for comparison: A's answer is what ships,
        # so a garbled cross-check lowers agreement without forcing an abstention.
        salvaged_b = salvage_receipt(fields_of(raw_b, doc_type), schema)
        result.document_b = salvaged_b.document
        result.errors.extend(f"model_b_{note}" for note in salvaged_b.notes)
    except Exception as exc:  # noqa: BLE001 - resilience is a graded requirement
        result.errors.append(f"model_b_failed: {type(exc).__name__}: {exc}")

    result.signals = build_signals(
        result.document, result.document_b, result.transcription, self_report
    )
    for f, sig in result.signals.items():
        if result.document_b is None:
            sig.agreement = 0.5  # unknown, not disagreement
        c = conf_model.score_field(f, sig)
        if is_invalid(f, salvaged_a.invalid):
            c = 0.0              # the model produced garbage here: never ship it, abstain
        result.confidence[f] = round(c, 4)
        if c < threshold:
            result.abstained.append(f)

    result.tokens = {"model_a": usage_a, "model_b": usage_b}
    result.latency_s = round(sum(s.latency_s for s in result.stages), 3)
    result.cost_usd = round(sum(s.cost_usd for s in result.stages), 8)
    result.wall_s = round(time.perf_counter() - started, 3)
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m extractor.pipeline <path-to-receipt>", file=sys.stderr)
        sys.exit(1)
    r = extract_with_confidence(sys.argv[1])
    print(r.document.model_dump_json(indent=2))
    print("\nconfidence:", json.dumps(r.confidence, indent=2))
    print("abstained:", r.abstained)
    if r.errors:
        print("errors:", r.errors, file=sys.stderr)
