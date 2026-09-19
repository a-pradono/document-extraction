"""What does the confidence model lose without the second model?

    python -m eval.ablation

The VLM-only baseline (report §4.8) shows the second model adds no field-level accuracy,
which raises the obvious question: why pay for it? This answers that in the only place it
can pay off - confidence. It fits the four-signal model on the calibration split, refits
with `agreement` neutralised (the state a one-model pipeline is in, having nothing to
agree with), and compares discrimination and the operating point.

Runs entirely from the extraction cache: no API calls.

Measured 2026-09-19, Gemini 3.6 Flash + GPT-OSS 120B, 198 field observations:
    four signals      AUC 0.908  keeps 86.4% at 95.3% precision
    without agreement AUC 0.852  keeps 75.8% at 95.3% precision
so the second model buys 10.6 points of coverage at the same precision target.
"""
import dataclasses
import sys
from pathlib import Path

from eval import metrics, run as eval_run          # noqa: E402
from extractor.confidence import ConfidenceModel, pick_threshold, risk_coverage_curve  # noqa: E402
from extractor.schema import document_type_of, model_for  # noqa: E402

samples = []
ids = [d for d in eval_run._doc_ids("calibration") if document_type_of(d) == "receipt"]
for doc_id in ids:
    gt = model_for(doc_id).model_validate_json(eval_run._find_gt(doc_id).read_text())
    ext = eval_run.extract_one(eval_run._find_doc(doc_id), "pipeline", None, 0.0, True)
    judged = {f: ok for f, _, ok in metrics.confidence_observations([
        metrics.compare_document(doc_id, ext.document, gt,
                                 confidence={f: 0.0 for f in ext.signals},
                                 unscored=eval_run.unscored_fields(doc_id))])}
    for field, sig in ext.signals.items():
        if field in judged:
            samples.append((field, sig, judged[field]))

print(f"{len(samples)} field observations over {len(ids)} receipts\n")


def evaluate(rows, label):
    model = ConfidenceModel()
    model.fit(rows)
    scored = [(model.score_field(f, s), ok) for f, s, ok in rows]
    chosen = pick_threshold(scored, target_precision=0.95)
    achieved = chosen["achieved"]
    print(f"{label:28} AUC {metrics.auc(scored):.3f}  ECE {metrics.ece(scored):.3f}  "
          f"threshold {chosen['threshold']}  keeps {achieved['coverage']:.1%} "
          f"at precision {achieved['precision']:.3f}  (target met: {chosen['met_target']})")
    return metrics.auc(scored), achieved


full = evaluate(samples, "four signals (pipeline)")
# A VLM-only path has no second reader, so `agreement` carries no information: neutral 0.5
neutral = [(f, dataclasses.replace(s, agreement=0.5), ok) for f, s, ok in samples]
without = evaluate(neutral, "without agreement (VLM only)")

print(f"\nDelta: AUC {without[0] - full[0]:+.3f}, "
      f"coverage at the 95% target {without[1]['coverage'] - full[1]['coverage']:+.1%}")
