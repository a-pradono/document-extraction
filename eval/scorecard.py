"""Scorecard writer (Phase 3).

Turns aggregated metrics + per-document comparisons into three artifacts:
  - scorecard.json      machine-readable, used by the regression diff (Phase 7)
  - scorecard.md        embeddable in the README
  - scorecard.html      human drill-down
  - per_document.json   field-by-field results with taxonomy labels
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from eval import taxonomy
from eval.metrics import DocResult


def _default(o):
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"not serializable: {type(o)}")


def _num(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=_default)


def per_document_payload(results: list[DocResult]) -> list[dict]:
    out = []
    for r in results:
        out.append({
            "doc_id": r.doc_id,
            "document_correct": r.document_correct,
            "latency_s": r.latency_s,
            "cost_usd": r.cost_usd,
            "tokens": r.tokens,
            "fields": [
                {
                    "field": c.field,
                    "gt": c.gt,
                    "pred": c.pred,
                    "label": c.label,
                    "exact": c.exact,
                    "normalized": c.normalized,
                    "confidence": r.confidence.get(c.field),
                }
                for c in r.comparisons
            ],
            "line_items": r.line_items,
        })
    return out


def to_markdown(scorecard: dict, meta: dict) -> str:
    m = scorecard
    lines = [
        f"# Extraction Scorecard — {meta.get('version', 'run')}",
        "",
        f"- Docs: **{m['n_docs']}**  ·  Set: **{meta.get('set', 'all')}**  ·  "
        f"Generated: {meta.get('generated_at', '')}",
        f"- Document Accuracy Rate: **{m['document_accuracy_rate']:.1%}**",
        f"- Micro F1: **{m['micro_f1']:.3f}**  ·  Macro F1: **{m['macro_f1']:.3f}**  ·  "
        f"Line-item F1: **{m['line_item_f1']:.3f}**",
        f"- Hallucination rate (spurious/predicted): **{m['hallucination_rate']:.1%}**",
    ]
    if m["latency"]:
        lat = m["latency"]
        lines.append(f"- Latency: p50 {lat['p50_s']}s · p95 {lat['p95_s']}s · mean {lat['mean_s']}s")
    lines += ["", "## Per-field (normalized)", "",
              "| field | support | precision | recall | F1 | exact F1 |",
              "|---|--:|--:|--:|--:|--:|"]
    for f, v in m["per_field"].items():
        lines.append(f"| {f} | {v['support']} | {_num(v['precision'])} | {_num(v['recall'])} "
                     f"| {_num(v['f1'])} | {_num(v['exact_f1'])} |")
    if any(v["f1"] is None for v in m["per_field"].values()):
        lines += ["", "n/a: no ground-truth value and no prediction anywhere in the set, so the "
                      "field is undefined rather than 0 and is excluded from macro F1."]
    lines += ["", "## Failures by kind", "", "| kind | count |", "|---|--:|"]
    for label in taxonomy.ALL_LABELS:
        lines.append(f"| {label} | {m['error_taxonomy'].get(label, 0)} |")

    types = m.get("by_document_type") or {}
    if types:
        lines += ["", "## Other document types (not pooled into the headline)", "",
                  f"The headline above is {m.get('headline_document_type', 'receipt')}s only. "
                  "Each type below has its own fields; at small n these are indicative, and "
                  "their confidence reuses the receipt-fitted model (not calibrated for the type).", "",
                  "| type | n | doc accuracy | micro F1 | line-item F1 | hallucination | swapped fields | abstention precision kept |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for name, t in types.items():
            ab = t.get("abstention") or {}
            kept = ab.get("precision_kept")
            lines.append(f"| {name} | {t['n_docs']} | {t['document_accuracy_rate']:.1%} | "
                         f"{t['micro_f1']:.3f} | {t['line_item_f1']:.3f} | {t['hallucination_rate']:.1%} | "
                         f"{t['error_taxonomy'].get('swapped_field', 0)} | "
                         f"{'-' if kept is None else f'{kept:.3f}'} |")
            lines += ["", f"Per-field ({name}):", "", "| field | support | F1 |", "|---|--:|--:|"]
            lines += [f"| {f} | {v['support']} | {_num(v['f1'])} |" for f, v in t["per_field"].items()]

    bd = m.get("breakdown") or {}
    if bd:
        lines += ["", "## By subset", "",
                  "Blended numbers hide where the system fails. Small subsets are noisy; "
                  "read each against its n.", "",
                  "| subset | n | doc accuracy | micro F1 | line-item F1 | hallucination | top failure kinds |",
                  "|---|--:|--:|--:|--:|--:|---|"]
        for name, s in bd.items():
            kinds = ", ".join(f"{k} {v}" for k, v in sorted(s["error_taxonomy"].items(),
                                                           key=lambda kv: -kv[1])[:3]) or "-"
            lines.append(f"| {name} | {s['n_docs']} | {s['document_accuracy_rate']:.1%} | "
                         f"{s['micro_f1']:.3f} | {s['line_item_f1']:.3f} | "
                         f"{s['hallucination_rate']:.1%} | {kinds} |")

    cal = m.get("calibration", {})
    if cal.get("available"):
        lines += ["", "## Confidence calibration", "",
                  f"- AUC (discrimination): **{cal.get('auc')}** — "
                  "~0.5 means the score carries no information, and rescaling cannot fix that.",
                  f"- ECE (calibration): **{cal.get('ece')}** — "
                  "high ECE with good AUC is the fixable case: refit on the calibration split.",
                  "", "| conf range | n | accuracy |", "|---|--:|--:|"]
        for b in cal["bins"]:
            lines.append(f"| {b['range']} | {b['n']} | {b['accuracy']:.3f} |")

    ab = m.get("abstention", {})
    if ab.get("available") and ab.get("precision_kept") is not None:
        delta = ab["precision_kept"] - ab["precision_all"]
        lines += ["", "## Abstention", "",
                  f"At threshold **{ab['threshold']}**, keeping {ab['coverage']:.1%} of "
                  f"{ab['n_fields']} fields moves precision "
                  f"**{ab['precision_all']:.3f} -> {ab['precision_kept']:.3f}** "
                  f"({delta:+.3f}).",
                  "", "| threshold | coverage | precision |", "|--:|--:|--:|"]
        for point in ab["risk_coverage_curve"][::4]:
            lines.append(f"| {point['threshold']} | {point['coverage']:.3f} "
                         f"| {point['precision']:.3f} |")

    cl = m.get("cost_latency")
    if cl:
        lat = cl.get("latency") or {}
        lines += ["", "## Cost and latency", "",
                  f"Cost basis: {cl['cost_basis']}.", "",
                  "| metric | value |", "|---|--:|",
                  f"| cost / doc | ${cl['cost_per_doc_usd']:.5f} |",
                  f"| tokens / doc | {cl['tokens_per_doc']:.0f} |",
                  f"| latency p50 / p95 (n={lat.get('n', 0)}) | "
                  f"{lat.get('p50_s', '-')}s / {lat.get('p95_s', '-')}s |",
                  "", "| stage | calls | tokens in | tokens out | $ | mean latency |",
                  "|---|--:|--:|--:|--:|--:|"]
        for name, a in cl["stages"].items():
            lines.append(f"| {name} | {a['calls']} | {a['prompt_tokens']} | "
                         f"{a['completion_tokens']} | {a['cost_usd']:.5f} | {a['mean_latency_s']}s |")
        if not cl.get("fully_priced", True):
            lines += ["", "> Some stages have no price in `.env`; their cost counts as $0."]

    cas = m.get("cascade")
    if cas:
        acc_a, acc_e = cas["field_accuracy_accepted"], cas["field_accuracy_escalated"]
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        lines += ["", "## Cascade", "",
                  f"Escalated **{cas['escalated']} of {cas['n_docs']}** documents "
                  f"({cas['escalation_rate']:.1%}) to the two-model path.", "",
                  "| path | docs | field accuracy |", "|---|--:|--:|",
                  f"| accepted on cheap path | {cas['accepted_cheap']} | {fmt(acc_a)} |",
                  f"| escalated | {cas['escalated']} | {fmt(acc_e)} |", "",
                  "Accepted documents scoring well below escalated ones means the gate is",
                  "letting bad reads through: the saving is being paid for in accuracy.", "",
                  "| escalation reason | docs |", "|---|--:|"]
        lines += [f"| {k} | {v} |" for k, v in cas["reasons"].items()]

    failed = m.get("failed_documents") or []
    if failed:
        lines += ["", "## Failed documents", "",
                  "Scored as empty extractions — a document we could not read is a recall",
                  "failure, not one that quietly leaves the denominator.", "",
                  "| doc | error |", "|---|---|"]
        for f in failed:
            lines.append(f"| {f['doc_id']} | {f['error']} |")
    lines.append("")
    return "\n".join(lines)


_HTML = """<!doctype html><meta charset="utf-8"><title>Scorecard {version}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#111}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.k{{color:#555}} .big{{font-size:1.4rem;font-weight:700}}
.miss{{color:#b45309}} .spur{{color:#b91c1c}} .wrong{{color:#b91c1c}}
</style>
<h1>Extraction Scorecard — {version}</h1>
<p class=k>{n_docs} docs · set: {set} · {generated_at}</p>
<p><span class=big>{dar:.1%}</span> document accuracy ·
micro F1 {micro:.3f} · macro F1 {macro:.3f} · line-item F1 {li:.3f} ·
hallucination {halluc:.1%}</p>
{tables}
"""


def to_html(scorecard: dict, meta: dict) -> str:
    m = scorecard
    field_rows = "".join(
        f"<tr><td>{f}</td><td>{v['support']}</td><td>{_num(v['precision'])}</td>"
        f"<td>{_num(v['recall'])}</td><td>{_num(v['f1'])}</td><td>{_num(v['exact_f1'])}</td></tr>"
        for f, v in m["per_field"].items()
    )
    tax_rows = "".join(
        f"<tr><td>{label}</td><td>{m['error_taxonomy'].get(label, 0)}</td></tr>"
        for label in taxonomy.ALL_LABELS
    )
    tables = (
        "<h2>Per-field (normalized)</h2><table>"
        "<tr><th>field</th><th>support</th><th>precision</th><th>recall</th>"
        f"<th>F1</th><th>exact F1</th></tr>{field_rows}</table>"
        "<h2>Failures by kind</h2><table>"
        f"<tr><th>kind</th><th>count</th></tr>{tax_rows}</table>"
    )
    return _HTML.format(
        version=meta.get("version", "run"), n_docs=m["n_docs"], set=meta.get("set", "all"),
        generated_at=meta.get("generated_at", ""), dar=m["document_accuracy_rate"],
        micro=m["micro_f1"], macro=m["macro_f1"], li=m["line_item_f1"],
        halluc=m["hallucination_rate"], tables=tables,
    )


def write_scorecard(out_dir: Path, scorecard: dict, results: list[DocResult], meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(_dumps({"meta": meta, "metrics": scorecard}))
    (out_dir / "scorecard.md").write_text(to_markdown(scorecard, meta))
    (out_dir / "scorecard.html").write_text(to_html(scorecard, meta))
    (out_dir / "per_document.json").write_text(_dumps(per_document_payload(results)))
    return out_dir / "scorecard.json"
