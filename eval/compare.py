"""Before/after trade-off table for an optimization (Phase 6).

    python -m eval.compare outputs/<baseline_run> outputs/<optimized_run> [--out table.md]

Puts cost, latency and accuracy side by side for two runs on the SAME document set.
An optimization only counts as measured if all three move in one table. "-40% $/doc"
means nothing without the accuracy column beside it.

Refuses to compare runs over different documents, since a delta across different sets
measures the sets. Phase 7's regression diff goes deeper (field by field, newly failing
documents); this is the headline trade-off.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class Row:
    label: str
    get: Callable[[dict], Optional[float]]
    better: str            # "higher" | "lower" | "info"
    kind: str              # "rate" (pp delta) | "ratio" (relative % delta) | "score" (raw delta)


def _dig(d: dict, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


ROWS = [
    Row("Document accuracy rate", lambda m: _dig(m, "document_accuracy_rate"), "higher", "rate"),
    Row("Micro F1", lambda m: _dig(m, "micro_f1"), "higher", "rate"),
    Row("Macro F1", lambda m: _dig(m, "macro_f1"), "higher", "rate"),
    Row("Line-item F1", lambda m: _dig(m, "line_item_f1"), "higher", "rate"),
    Row("Hallucination rate", lambda m: _dig(m, "hallucination_rate"), "lower", "rate"),
    Row("Precision on kept fields", lambda m: _dig(m, "abstention", "precision_kept"), "higher", "rate"),
    Row("Coverage (fields kept)", lambda m: _dig(m, "abstention", "coverage"), "higher", "rate"),
    Row("Confidence AUC", lambda m: _dig(m, "calibration", "auc"), "higher", "score"),
    Row("Confidence ECE", lambda m: _dig(m, "calibration", "ece"), "lower", "score"),
    Row("$ / doc (list price)", lambda m: _dig(m, "cost_latency", "cost_per_doc_usd"), "lower", "ratio"),
    Row("Tokens / doc", lambda m: _dig(m, "cost_latency", "tokens_per_doc"), "lower", "ratio"),
    Row("Latency p50 (s)", lambda m: _dig(m, "cost_latency", "latency", "p50_s"), "lower", "ratio"),
    Row("Latency p95 (s)", lambda m: _dig(m, "cost_latency", "latency", "p95_s"), "lower", "ratio"),
    Row("Escalation rate", lambda m: _dig(m, "cascade", "escalation_rate"), "info", "rate"),
]

# Differences smaller than this are reported as unchanged rather than as a direction.
_EPSILON = 1e-9


def load_run(path: Path) -> tuple[dict, dict, list[str]]:
    """Accept a run directory or its scorecard.json. Returns (meta, metrics, doc_ids)."""
    run_dir = path if path.is_dir() else path.parent
    scorecard = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
    per_doc_path = run_dir / "per_document.json"
    doc_ids = []
    if per_doc_path.exists():
        doc_ids = sorted(d["doc_id"] for d in json.loads(per_doc_path.read_text(encoding="utf-8")))
    return scorecard.get("meta", {}), scorecard["metrics"], doc_ids


def _fmt(value: Optional[float], row: Row) -> str:
    if value is None:
        return "-"
    if row.label.startswith("$"):
        return f"${value:.5f}"
    if row.label.startswith("Tokens"):
        return f"{value:,.0f}"
    if row.kind == "ratio":
        return f"{value:.2f}"
    return f"{value:.3f}"


def _delta(a: Optional[float], b: Optional[float], row: Row) -> tuple[str, str]:
    if a is None or b is None:
        return "-", ""
    diff = b - a
    if abs(diff) <= _EPSILON:
        return "0", "same"
    if row.kind == "ratio":
        text = f"{diff / a:+.1%}" if a else f"{diff:+.3f}"
    elif row.kind == "score":
        text = f"{diff:+.3f}"                  # AUC/ECE are not percentages
    else:
        text = f"{diff * 100:+.1f} pp"
    if row.better == "info":
        return text, ""
    improved = diff > 0 if row.better == "higher" else diff < 0
    return text, "better" if improved else "worse"


def compare(before: tuple[dict, dict, list[str]], after: tuple[dict, dict, list[str]]) -> dict:
    (meta_a, m_a, ids_a), (meta_b, m_b, ids_b) = before, after
    rows = []
    for row in ROWS:
        a, b = row.get(m_a), row.get(m_b)
        if a is None and b is None:
            continue
        delta, verdict = _delta(a, b, row)
        rows.append({"metric": row.label, "before": a, "after": b, "delta": delta,
                     "verdict": verdict, "_row": row})
    return {"meta_before": meta_a, "meta_after": meta_b, "n_docs": len(ids_a) or m_a.get("n_docs"),
            "rows": rows}


def to_markdown(result: dict) -> str:
    a, b = result["meta_before"], result["meta_after"]
    n = result["n_docs"] or 0
    lines = [
        "# Before / after",
        "",
        f"- **Before:** `{a.get('version', '?')}` (source `{a.get('source', '?')}`)",
        f"- **After:** `{b.get('version', '?')}` (source `{b.get('source', '?')}`)",
        f"- Same {n} documents, set `{a.get('set', '?')}`.",
        "",
        "| metric | before | after | delta | |",
        "|---|--:|--:|--:|---|",
    ]
    for r in result["rows"]:
        row = r["_row"]
        lines.append(f"| {r['metric']} | {_fmt(r['before'], row)} | {_fmt(r['after'], row)} "
                     f"| {r['delta']} | {r['verdict']} |")
    if n:
        # The single most useful sentence for reading an accuracy delta at this size.
        lines += ["", f"At n={n}, one document flips document accuracy by "
                      f"{100 / n:.1f} pp. Treat accuracy deltas smaller than that as noise."]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-different-sets", action="store_true",
                    help="compare anyway (the deltas will then partly measure the sets)")
    args = ap.parse_args()

    before, after = load_run(args.before), load_run(args.after)
    if before[2] != after[2] and not args.allow_different_sets:
        only_a = sorted(set(before[2]) - set(after[2]))
        only_b = sorted(set(after[2]) - set(before[2]))
        raise SystemExit(
            "runs cover different documents - a delta would measure the sets, not the change.\n"
            f"  only in before: {only_a[:10]}{' ...' if len(only_a) > 10 else ''}\n"
            f"  only in after:  {only_b[:10]}{' ...' if len(only_b) > 10 else ''}\n"
            "Re-run both on the same --set, or pass --allow-different-sets.")

    table = to_markdown(compare(before, after))
    print(table)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
