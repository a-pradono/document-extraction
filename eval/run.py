"""Eval runner (Phase 3, extended in Phase 4).

  python -m eval.run --set smoke|calibration|test|degraded|synthetic
                     [--source pipeline|cascade|vlm-only|api] [--version v1] [--no-cache]
  python -m eval.run --calibrate                 # fit the confidence model, pick a threshold

Loads ground truth from data/ground_truth/<id>.json, finds the matching document in
data/docs/<id>.*, extracts it, compares against ground truth, and writes a timestamped
scorecard to outputs/.

Two extraction sources exist on purpose:
  vlm-only   the Phase 3 baseline — one VLM call, no confidence.
  pipeline   Phase 4 — VLM + text-LLM + deterministic validation, with per-field
             confidence and abstention. Default.
Keeping the baseline runnable is what makes "abstention raised precision" a measured
claim rather than an assertion.

--calibrate fits the confidence model on the calibration split ONLY and reads the
abstention threshold off the risk-coverage curve. Nothing is ever fitted on `test`;
`smoke` is drawn from calibration so heavy iteration cannot overfit the reported numbers.

--source api (Phase 8) uploads each document to the running extraction service
(EVAL_API_URL, default http://127.0.0.1:8000) so the same harness scores the deployed
container. The server's models, mode and threshold are what gets measured, so the
scorecard's meta records them from /health rather than from this process's .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

from extractor.confidence import (
    ConfidenceModel,
    FieldSignals,
    pick_threshold,
    risk_coverage_curve,
)
from extractor.cascade import extract_cascade
from extractor.cost import StageUsage, summarize as summarize_cost
from extractor.extract import extract as extract_vlm_only
from extractor.pipeline import PipelineResult, extract_with_confidence
from extractor.schema import document_type_of, model_for
from api.schemas import from_result
from eval import fingerprint, metrics, taxonomy
from eval.render_doc import write_card
from eval.scorecard import write_scorecard

load_dotenv()

GT_DIR = Path("data/ground_truth")
DOCS_DIR = Path("data/docs")
SETS_DIR = Path("data/sets")
OUT_ROOT = Path("outputs")

CALIBRATION_SET = "calibration"
MODEL_PATH = Path(os.environ.get("CONFIDENCE_MODEL_PATH", ".cache/confidence_model.json"))
THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))


@dataclass
class Extraction:
    """One document's extraction, whatever source produced it."""

    document: BaseModel
    latency_s: float
    transcription: str = ""
    confidence: dict[str, float] = dc_field(default_factory=dict)
    signals: dict[str, FieldSignals] = dc_field(default_factory=dict)
    abstained: list[str] = dc_field(default_factory=list)
    tokens: dict = dc_field(default_factory=dict)
    stages: list[StageUsage] = dc_field(default_factory=list)
    cost_usd: float = 0.0
    errors: list[str] = dc_field(default_factory=list)
    cascade: dict = dc_field(default_factory=dict)


# --- document discovery ------------------------------------------------------
def _doc_ids(set_name: str | None) -> list[str]:
    if set_name:
        manifest = SETS_DIR / f"{set_name}.txt"
        if not manifest.exists():
            raise SystemExit(f"no manifest {manifest} — run scripts/assign_splits.py")
        return [ln.strip() for ln in manifest.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    # Non-recursive on purpose: ground_truth/synthetic/ holds augmented copies, which
    # must never land in the headline scorecard unless asked for by --set synthetic.
    return sorted(p.stem for p in GT_DIR.glob("*.json") if not p.stem.startswith("_"))


def _find_gt(doc_id: str) -> Path:
    for candidate in (GT_DIR / f"{doc_id}.json", GT_DIR / "synthetic" / f"{doc_id}.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no ground truth in {GT_DIR} for id '{doc_id}'")


def _find_doc(doc_id: str) -> Path:
    hits = sorted(DOCS_DIR.glob(f"{doc_id}.*")) or sorted(DOCS_DIR.rglob(f"{doc_id}.*"))
    if not hits:
        raise FileNotFoundError(f"no document in {DOCS_DIR} for id '{doc_id}'")
    return hits[0]


# --- extraction sources ------------------------------------------------------
API_URL = os.environ.get("EVAL_API_URL", "http://127.0.0.1:8000")
_api_client: Optional[httpx.Client] = None


def api_client() -> httpx.Client:
    """One pooled client per run. Replaceable, so tests can hand in a FastAPI TestClient
    and exercise the real HTTP contract without a server."""
    global _api_client
    if _api_client is None:
        timeout = float(os.environ.get("EVAL_API_TIMEOUT_S", "600"))
        _api_client = httpx.Client(base_url=API_URL, timeout=timeout)
    return _api_client


def api_preflight() -> dict:
    """Fail once, up front, rather than recording the same connection error against
    every document — and capture what the server is actually running, because with
    --source api the SERVER's models and threshold are what gets measured."""
    try:
        resp = api_client().get("/health")
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"extraction API not reachable at {API_URL}: {exc}\n"
                         f"Start it (docker compose up) or set EVAL_API_URL.") from None
    health = resp.json()
    if health.get("status") == "misconfigured":
        raise SystemExit(f"extraction API at {API_URL} is misconfigured: "
                         f"{', '.join(health.get('missing_config', []))} unset")
    return health


def _extract_via_api(doc_path: Path, doc_type: str = "receipt") -> Extraction:
    with doc_path.open("rb") as fh:
        resp = api_client().post("/extract", files={"file": (doc_path.name, fh)},
                                 data={"document_type": doc_type})
    if resp.status_code == 502:
        detail = resp.json().get("detail", {})
        raise RuntimeError(f"API extraction failed: {detail.get('error', detail)}")
    resp.raise_for_status()
    body = resp.json()

    # The API WITHHOLDS abstained values from `receipt` (a consumer must never see them).
    # The eval needs them back: abstention is measured as precision on kept vs all
    # fields, and scoring a withheld-but-correct value as "missing" would make an API run
    # disagree with a local run of the very same system.
    raw = json.loads(json.dumps(body["document"] or {}))
    for field, value in (body.get("withheld") or {}).items():
        restored = json.loads(value) if field == "line_items" and value else value
        *parents, leaf = field.split(".")                  # "vendor.name" -> raw["vendor"]["name"]
        target = raw
        for parent in parents:
            target = target.setdefault(parent, {}) or {}
        target[leaf] = restored
    usage = body.get("usage") or {}
    stages = [StageUsage(**s) for s in usage.get("stages", [])]
    return Extraction(
        document=model_for(doc_type).model_validate(raw),
        latency_s=float(usage.get("latency_s", 0.0)),
        confidence=body.get("confidence") or {},
        abstained=body.get("abstained") or [],
        signals={f: FieldSignals(**s) for f, s in (body.get("signals") or {}).items()},
        transcription=body.get("transcription") or "",
        stages=stages,
        cost_usd=float(usage.get("cost_usd", 0.0)),
        errors=body.get("warnings") or [],
        cascade=body.get("cascade") or {},
    )


def extract_one(
    doc_path: Path,
    source: str,
    conf_model: Optional[ConfidenceModel],
    threshold: float,
    use_cache: bool,
    doc_type: str = "receipt",
) -> Extraction:
    if source == "api":
        return _extract_via_api(doc_path, doc_type)
    if source == "vlm-only":
        start = time.perf_counter()
        document = extract_vlm_only(doc_path, doc_type=doc_type)
        return Extraction(document=document, latency_s=time.perf_counter() - start)

    extractor = extract_cascade if source == "cascade" else extract_with_confidence
    result = extractor(doc_path, threshold=threshold, model=conf_model, use_cache=use_cache,
                       doc_type=doc_type)
    return Extraction(
        document=result.document,
        latency_s=result.latency_s,
        transcription=result.transcription,
        confidence=result.confidence,
        signals=result.signals,
        abstained=result.abstained,
        tokens=result.tokens,
        stages=result.stages,
        cost_usd=result.cost_usd,
        errors=result.errors,
        cascade=result.cascade,
    )


def _load_model() -> tuple[Optional[ConfidenceModel], bool]:
    if MODEL_PATH.exists():
        return ConfidenceModel.load(MODEL_PATH), True
    return None, False


# --- the run -----------------------------------------------------------------
def run(set_name: str | None, source: str, version: str, use_cache: bool = True) -> Path:
    ids = _doc_ids(set_name)
    if not ids:
        print(f"No ground truth found in {GT_DIR}. Add <id>.json files first.", file=sys.stderr)
        sys.exit(1)

    conf_model, fitted = _load_model()
    threshold = THRESHOLD
    server: dict = {}
    if source == "api":
        server = api_preflight()
        # The server abstained at ITS threshold; measuring abstention at ours would
        # score a system that never ran.
        threshold = float(server.get("threshold", threshold))
        fitted = server.get("confidence_model") == "fitted"
    if source in ("pipeline", "cascade", "api") and not fitted:
        print(f"! no fitted confidence model at {MODEL_PATH} - using fallback weights.\n"
              f"  Run `python -m eval.run --calibrate` first for meaningful confidence.",
              file=sys.stderr)

    results: list[metrics.DocResult] = []
    cards: list[dict] = []
    failures: list[dict] = []
    stage_rows: list[list[StageUsage]] = []
    cascades: dict[str, dict] = {}
    for doc_id in ids:
        doc_type = document_type_of(doc_id)
        gt = model_for(doc_type).model_validate_json(_find_gt(doc_id).read_text())
        try:
            doc_path = _find_doc(doc_id)
            ext = extract_one(doc_path, source, conf_model, threshold, use_cache, doc_type)
        except Exception as exc:  # noqa: BLE001 — one bad document must not kill the batch
            failures.append({"doc_id": doc_id, "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc(limit=3)})
            print(f"  FAILED {doc_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            # Score it as an empty extraction: a document we could not read is a recall
            # failure, not a document that quietly disappears from the denominator.
            ext = Extraction(document=model_for(doc_type)(), latency_s=0.0)

        doc = metrics.compare_document(
            doc_id, ext.document, gt, latency_s=ext.latency_s, confidence=ext.confidence
        )
        cards.append(json.loads(from_result(doc_id, PipelineResult(
            document=ext.document, document_b=None, transcription=ext.transcription,
            confidence=ext.confidence, abstained=ext.abstained, signals=ext.signals,
            latency_s=ext.latency_s, tokens=ext.tokens, stages=ext.stages,
            cost_usd=ext.cost_usd, errors=ext.errors, cascade=ext.cascade,
        ), threshold, doc_type).model_dump_json()))
        doc.tokens = ext.tokens or None
        doc.cost_usd = ext.cost_usd or None
        results.append(doc)
        if ext.stages:
            stage_rows.append(ext.stages)
        if ext.cascade:
            cascades[doc_id] = ext.cascade
        flag = f" | abstained: {len(ext.abstained)}" if ext.abstained else ""
        if ext.cascade:
            flag += " | escalated" if ext.cascade.get("escalated") else " | cheap path"
        print(f"  scored {doc_id} ({ext.latency_s:.2f}s){flag}", file=sys.stderr)

    # Receipts are the headline. Other document types are reported alongside, never pooled
    # in: different fields and difficulty, and at n=10 invoices would distort every number.
    by_type: dict[str, list[metrics.DocResult]] = {}
    for r in results:
        by_type.setdefault(document_type_of(r.doc_id), []).append(r)
    headline = by_type.get("receipt") or results
    scorecard = metrics.aggregate(headline)
    scorecard["headline_document_type"] = "receipt" if "receipt" in by_type else document_type_of(results[0].doc_id)
    scorecard["by_document_type"] = {
        t: {**metrics.aggregate(docs), "abstention": abstention_report(docs, threshold)}
        for t, docs in sorted(by_type.items()) if docs is not headline
    }
    scorecard["abstention"] = abstention_report(headline, threshold)
    if stage_rows:
        # Failed documents are excluded from latency (they have none) but are already
        # counted against accuracy above, so they cannot flatter either number.
        scorecard["cost_latency"] = summarize_cost(
            stage_rows, [sum(s.latency_s for s in row) for row in stage_rows])
    scorecard["breakdown"] = breakdown(results)
    if cascades:
        scorecard["cascade"] = cascade_report(results, cascades)
    if failures:
        scorecard["failed_documents"] = failures

    meta = {
        "version": version,
        "set": set_name or "all",
        "source": source,
        "confidence_model": ("fitted" if fitted else "unfitted (fallback weights)")
                            if source == "api" else
                            (str(MODEL_PATH) if fitted else "unfitted (fallback weights)"),
        "threshold": threshold,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if server:
        meta["api"] = {"url": API_URL, "mode": server.get("mode"),
                       "model_a": server.get("model_a"), "model_b": server.get("model_b"),
                       "cascade_model": server.get("cascade_model")}
    meta["fingerprint"] = fingerprint.build(
        ids, _find_gt, _find_doc, source=source, threshold=threshold,
        confidence_model_path=MODEL_PATH, server=server or None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = OUT_ROOT / f"{stamp}_{version}"
    for payload in cards:                      # one human-readable card per document
        write_card(payload, out_dir / "documents")
    path = write_scorecard(out_dir, scorecard, results, meta)

    print(f"\nDocument accuracy: {scorecard['document_accuracy_rate']:.1%} | "
          f"micro F1: {scorecard['micro_f1']:.3f} | line-item F1: {scorecard['line_item_f1']:.3f}")
    ab = scorecard["abstention"]
    if ab.get("available"):
        print(f"Abstention @ {threshold}: coverage {ab['coverage']:.1%}, "
              f"precision {ab['precision_all']:.3f} -> {ab['precision_kept']:.3f}")
    if failures:
        print(f"{len(failures)} document(s) failed - see scorecard.json", file=sys.stderr)
    print(f"Scorecard: {path}")
    return path


def abstention_report(results: list[metrics.DocResult], threshold: float) -> dict:
    """What abstaining actually bought: precision on kept fields vs on everything.

    This is the Phase 4 gate. If precision_kept is not meaningfully above
    precision_all, the confidence score is not doing its job and the fix is the
    SIGNALS, not the threshold.
    """
    scored = [(conf, ok) for _, conf, ok in metrics.confidence_observations(results)]
    if not scored:
        return {"available": False}

    kept = [ok for conf, ok in scored if conf >= threshold]
    return {
        "available": True,
        "threshold": threshold,
        "n_fields": len(scored),
        "coverage": round(len(kept) / len(scored), 4),
        "precision_all": round(sum(ok for _, ok in scored) / len(scored), 4),
        "precision_kept": round(sum(kept) / len(kept), 4) if kept else None,
        "risk_coverage_curve": risk_coverage_curve(scored),
    }


def breakdown(results: list[metrics.DocResult]) -> dict:
    """The same headline metrics per subset: each source, and the degraded tag.

    One blended number hides the thing a reader most needs to know about messy
    documents. "F1 0.91 on clean CORD, 0.68 on photographed retail" is a finding;
    "F1 0.80" averages it away. Subsets under 3 documents are omitted as noise.
    """
    groups: dict[str, list[metrics.DocResult]] = {}
    for r in results:
        groups.setdefault(f"source:{r.doc_id.split('_')[0]}", []).append(r)
    degraded_path = SETS_DIR / "degraded.txt"
    if degraded_path.exists():
        tagged = {ln.strip() for ln in degraded_path.read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")}
        subset = [r for r in results if r.doc_id in tagged]
        if subset and len(subset) < len(results):
            groups["degraded"] = subset
            groups["not degraded"] = [r for r in results if r.doc_id not in tagged]

    out = {}
    for name, docs in sorted(groups.items()):
        if len(docs) < 3 or len(docs) == len(results) and name.startswith("source:"):
            continue
        agg = metrics.aggregate(docs)
        out[name] = {k: agg[k] for k in ("n_docs", "document_accuracy_rate", "micro_f1",
                                         "line_item_f1", "hallucination_rate")}
        out[name]["error_taxonomy"] = {k: v for k, v in agg["error_taxonomy"].items()
                                       if k != taxonomy.CORRECT and v}
    return out


def cascade_report(results: list[metrics.DocResult], cascades: dict[str, dict]) -> dict:
    """How often the cascade escalated, why, and whether the gate worked.

    The key diagnostic is accuracy on ACCEPTED documents. The cheap path is only safe if
    what it keeps is about as accurate as what the full path would produce. If accepted
    documents score well below escalated ones, the gate is letting bad reads through and
    the cost saving is being paid for in accuracy.
    """
    def field_accuracy(docs: list[metrics.DocResult]) -> Optional[float]:
        labels = [c.label for r in docs for c in r.comparisons]
        return round(sum(l == taxonomy.CORRECT for l in labels) / len(labels), 4) if labels else None

    by_id = {r.doc_id: r for r in results}
    escalated = [d for d, c in cascades.items() if c.get("escalated")]
    accepted = [d for d, c in cascades.items() if not c.get("escalated")]

    reasons: dict[str, int] = {}
    for c in cascades.values():
        # Collapse "low_confidence:total" to "low_confidence" so the counts stay readable;
        # per-field detail stays in per_document.json.
        for kind in {r.split(":", 1)[0] for r in c.get("reasons", [])}:
            reasons[kind] = reasons.get(kind, 0) + 1

    return {
        "n_docs": len(cascades),
        "escalated": len(escalated),
        "accepted_cheap": len(accepted),
        "escalation_rate": round(len(escalated) / len(cascades), 4) if cascades else 0.0,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "field_accuracy_accepted": field_accuracy([by_id[d] for d in accepted if d in by_id]),
        "field_accuracy_escalated": field_accuracy([by_id[d] for d in escalated if d in by_id]),
        "escalated_docs": sorted(escalated),
    }


# --- calibration -------------------------------------------------------------
# A fit is only as good as its coverage. Below this share of the calibration split
# actually extracted, the weights and the threshold describe a handful of documents
# that happened to run before the quota ran out, not the split.
MIN_CALIBRATION_COVERAGE = 0.8


def calibrate(target_precision: float, use_cache: bool = True,
              allow_partial: bool = False) -> Path:
    """Fit the confidence model on the calibration split and pick a threshold.

    The split is held out from `test` by construction (scripts/assign_splits.py), so
    nothing fitted here has seen a document that appears in the reported scorecard.
    """
    # Receipts only. Three calibration invoices are far too few to fit on, and pooling them
    # would let a different document type nudge the receipt weights. Invoices reuse the
    # receipt-fitted model, and their confidence is reported as provisional, not calibrated.
    all_ids = _doc_ids(CALIBRATION_SET)
    ids = [d for d in all_ids if document_type_of(d) == "receipt"]
    skipped = len(all_ids) - len(ids)
    print(f"Calibrating on {len(ids)} receipts from '{CALIBRATION_SET}'"
          + (f" ({skipped} non-receipt document(s) excluded)" if skipped else "") + "...",
          file=sys.stderr)

    samples: list[tuple[str, FieldSignals, bool]] = []
    collected: list[str] = []
    failed: list[dict] = []
    for doc_id in ids:
        gt = model_for(doc_id).model_validate_json(_find_gt(doc_id).read_text())
        try:
            ext = extract_one(_find_doc(doc_id), "pipeline", None, 0.0, use_cache)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {doc_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append({"doc_id": doc_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        # Correctness comes from the same compare_document the scorecard uses, so the
        # model is fitted on exactly the definition of "correct" it is later judged by.
        judged = {f: ok for f, _, ok in metrics.confidence_observations([
            metrics.compare_document(doc_id, ext.document, gt,
                                     confidence={f: 0.0 for f in ext.signals})])}
        for field, sig in ext.signals.items():
            if field in judged:
                samples.append((field, sig, judged[field]))
        collected.append(doc_id)
        print(f"  collected {doc_id}", file=sys.stderr)

    if not samples:
        raise SystemExit("no calibration samples — check the calibration split and the providers")

    # A partial fit that reports a threshold is worse than no fit: the number looks
    # authoritative and nothing downstream knows it came from a fraction of the split.
    # This is how a rate-limited run quietly produced a "calibrated" model from 8 of 18
    # documents, exit code 0.
    coverage = len(collected) / len(ids)
    if coverage < MIN_CALIBRATION_COVERAGE and not allow_partial:
        reasons = {f["error"].split(":")[0] for f in failed}
        raise SystemExit(
            f"\nREFUSING to fit: only {len(collected)} of {len(ids)} documents extracted "
            f"({coverage:.0%}, need {MIN_CALIBRATION_COVERAGE:.0%}).\n"
            f"  {len(failed)} failed: {', '.join(sorted(reasons))}\n"
            f"  Nothing was written, so no run can pick up a threshold fitted on a "
            f"fraction of the split.\n"
            f"  Fix the cause (a daily quota resets; see the errors above), then re-run. "
            f"Cached documents cost no further calls.\n"
            f"  --allow-partial overrides this, and marks the report partial.")

    model = ConfidenceModel()
    fit_report = model.fit(samples)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    # Score the same samples with the fitted model to read the operating point off the
    # curve. These are in-sample numbers — they set the threshold, they are NOT the
    # reported result. The reported result comes from `--set test`.
    scored = [(model.score_field(f, s), ok) for f, s, ok in samples]
    chosen = pick_threshold(scored, target_precision=target_precision)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": CALIBRATION_SET,
        "n_documents": len(collected),          # what the fit SAW, not what was asked for
        "n_documents_requested": len(ids),
        "document_coverage": round(coverage, 4),
        "partial": coverage < MIN_CALIBRATION_COVERAGE,
        "failed_documents": failed,
        "n_field_observations": len(samples),
        "fit": fit_report,
        "chosen_threshold": chosen,
        "auc_in_sample": metrics.auc(scored),
        "ece_in_sample": metrics.ece(scored),
        "risk_coverage_curve": risk_coverage_curve(scored),
        "note": "In-sample; used to SET the threshold. Reported numbers come from --set test.",
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFitted weights per field group:")
    for group, info in fit_report.items():
        if info.get("fitted"):
            weights = " ".join(f"{k}={v:+.2f}" for k, v in info["weights"].items())
            print(f"  {group:<9} n={info['n']:<4} {weights} bias={info['bias']:+.2f}")
        else:
            print(f"  {group:<9} n={info['n']:<4} not fitted ({info['reason']}) - keeping defaults")
    print(f"\nAUC {report['auc_in_sample']} | ECE {report['ece_in_sample']} (in-sample)")
    print(f"Fitted on {len(collected)} of {len(ids)} documents ({coverage:.0%} coverage)"
          + ("  ** PARTIAL — the threshold is provisional **" if report["partial"] else ""))
    print(f"\nModel: {MODEL_PATH}\nReport: {out}")
    if chosen["met_target"]:
        a = chosen["achieved"]
        print(f"\nThreshold {chosen['threshold']} -> precision {a['precision']:.3f} "
              f"at {a['coverage']:.1%} coverage (target {target_precision})")
        print(f"Set CONFIDENCE_THRESHOLD={chosen['threshold']} in .env to use it.")
    else:
        # No threshold recommendation here: suggesting one right after saying the target is
        # unreachable invites adopting a number that was just shown to be inadequate.
        print(f"\n! target precision {target_precision} unreachable at any threshold - "
              f"best is {chosen['achieved'].get('precision')}. The signals need work, "
              f"not the threshold. No threshold recommended.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", default=None,
                    help="smoke|calibration|test|degraded|synthetic")
    ap.add_argument("--source", default="pipeline",
                    choices=["pipeline", "cascade", "vlm-only", "api"])
    ap.add_argument("--version", default="dev")
    ap.add_argument("--no-cache", action="store_true", help="bypass the image-hash cache")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit the confidence model on the calibration split and pick a threshold")
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--allow-partial", action="store_true",
                    help="fit even if part of the calibration split failed to extract "
                         "(marks the report partial; the threshold is then provisional)")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.target_precision, use_cache=not args.no_cache,
                  allow_partial=args.allow_partial)
    else:
        run(args.set_name, args.source, args.version, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
