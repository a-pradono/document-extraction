"""Regression gate: compare two runs and fail if the candidate got worse (Phase 7).

    python -m eval.regression outputs/<baseline_run> outputs/<candidate_run>
           [--max-field-drop 0.05] [--min-docs 2] [--max-dar-drop 0.05]
           [--allow-new-hallucinations 0] [--max-cost-rise 0.25] [--max-p95-rise 0.5]
           [--out report.md] [--json report.json]

Exit codes, so CI can block a merge:
    0  no regression
    1  regression: at least one gate failed
    2  the comparison itself is invalid (different documents, changed ground-truth labels
       or document bytes between the runs, missing files)

What it checks, and why each gate is shaped the way it is:

  field F1        A per-field F1 drop only counts when it's past --max-field-drop AND at
                  least --min-docs documents actually flipped from correct to incorrect on
                  that field. At n~43 one document moves a field's F1 by several points,
                  so a pure threshold would fail on noise, and a pure count would miss a
                  real drop on a rare field. Requiring both is what makes the gate usable.
  hallucinations  Any NEW spurious value (null in ground truth, filled by the candidate)
                  fails by default. Inventing a value is the one failure this system
                  promises never to make, so it gets no tolerance unless asked for.
  documents       Newly failing documents are listed with the exact label transition
                  (total: correct -> wrong_value). "DAR dropped 4.7 pp" is not
                  actionable; "these two receipts now misread the total" is.
  cost, latency   Gated as well, because an optimization that quietly doubles p95
                  should not ship either. Latency is noisier (network, cache), so its
                  default tolerance is looser.
  confidence      An AUC drop is reported as a warning. Confidence quality is real, but
                  on a few hundred field observations it is too noisy to block on.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

from eval import fingerprint, taxonomy
from eval.compare import compare as headline_compare, load_run, to_markdown as headline_markdown

EXIT_OK, EXIT_REGRESSION, EXIT_INVALID = 0, 1, 2


@dataclass
class Gates:
    max_field_drop: float = 0.05
    min_docs: int = 2
    max_dar_drop: float = 0.05
    allow_new_hallucinations: int = 0
    max_cost_rise: float = 0.25
    max_p95_rise: float = 0.50
    max_auc_drop_warn: float = 0.05
    allow_label_drift: bool = False


@dataclass
class Report:
    baseline: dict
    candidate: dict
    n_docs: int
    failures: list[str] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)
    field_changes: list[dict] = dc_field(default_factory=list)
    newly_failing: list[dict] = dc_field(default_factory=list)
    newly_fixed: list[dict] = dc_field(default_factory=list)
    new_hallucinations: list[dict] = dc_field(default_factory=list)
    taxonomy_delta: dict = dc_field(default_factory=dict)
    headline: dict = dc_field(default_factory=dict)
    config_changes: dict = dc_field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures


# --- loading -------------------------------------------------------------------
def _per_document(run_dir: Path) -> dict[str, dict]:
    run_dir = run_dir if run_dir.is_dir() else run_dir.parent
    path = run_dir / "per_document.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - is this a run directory from eval.run?")
    return {d["doc_id"]: d for d in json.loads(path.read_text(encoding="utf-8"))}


def _labels(doc: dict) -> dict[str, str]:
    return {f["field"]: f["label"] for f in doc["fields"]}


# --- analysis ------------------------------------------------------------------
def analyse(baseline_dir: Path, candidate_dir: Path, gates: Gates) -> Report:
    base_run, cand_run = load_run(baseline_dir), load_run(candidate_dir)
    base_docs, cand_docs = _per_document(baseline_dir), _per_document(candidate_dir)
    if set(base_docs) != set(cand_docs):
        raise ValueError(
            "runs cover different documents - a regression check across different sets "
            f"measures the sets. only in baseline: {sorted(set(base_docs) - set(cand_docs))[:10]}; "
            f"only in candidate: {sorted(set(cand_docs) - set(base_docs))[:10]}")

    (meta_b, m_b, _), (meta_c, m_c, _) = base_run, cand_run
    report = Report(baseline=meta_b, candidate=meta_c, n_docs=len(base_docs),
                    headline=headline_compare(base_run, cand_run))

    fp_b, fp_c = meta_b.get("fingerprint"), meta_c.get("fingerprint")
    if not fp_b or not fp_c:
        report.warnings.append("a run predates fingerprinting: label or document drift between "
                               "the runs cannot be ruled out")
    else:
        changes = fingerprint.differences(fp_b, fp_c)
        drifted = [k for k in ("ground_truth", "documents") if k in changes]
        if drifted and not gates.allow_label_drift:
            raise ValueError(
                f"{' and '.join(drifted)} changed between the runs, so a delta would measure the "
                f"relabeling, not the system. Re-run the baseline on the current labels, or pass "
                f"--allow-label-drift to compare anyway.")
        if drifted:
            report.warnings.append(f"{' and '.join(drifted)} changed between runs (allowed): part "
                                   f"of every delta below is the relabeling")
        report.config_changes = {k: v for k, v in changes.items()
                                 if k not in ("ground_truth", "documents")}

    _field_gate(report, m_b, m_c, base_docs, cand_docs, gates)
    _document_changes(report, base_docs, cand_docs)
    _hallucination_gate(report, base_docs, cand_docs, gates)
    _headline_gates(report, m_b, m_c, gates)
    report.taxonomy_delta = {
        label: m_c["error_taxonomy"].get(label, 0) - m_b["error_taxonomy"].get(label, 0)
        for label in taxonomy.ALL_LABELS
    }
    return report


def _field_gate(report: Report, m_b: dict, m_c: dict,
                base_docs: dict, cand_docs: dict, gates: Gates) -> None:
    for field_name, before in m_b["per_field"].items():
        after = m_c["per_field"].get(field_name)
        if after is None or (before["f1"] is None and after["f1"] is None):
            continue
        # Undefined F1 (no ground truth, no prediction) is "nothing to get wrong"; if the
        # candidate starts predicting there, that is spurious, and scores as a drop from 1.
        before = dict(before, f1=1.0 if before["f1"] is None else before["f1"])
        after = dict(after, f1=1.0 if after["f1"] is None else after["f1"])
        flipped_bad = sorted(d for d in base_docs
                             if _labels(base_docs[d]).get(field_name) == taxonomy.CORRECT
                             and _labels(cand_docs[d]).get(field_name) != taxonomy.CORRECT)
        flipped_good = sorted(d for d in base_docs
                              if _labels(base_docs[d]).get(field_name) != taxonomy.CORRECT
                              and _labels(cand_docs[d]).get(field_name) == taxonomy.CORRECT)
        drop = before["f1"] - after["f1"]
        regressed = drop > gates.max_field_drop and len(flipped_bad) >= gates.min_docs
        report.field_changes.append({
            "field": field_name, "f1_before": before["f1"], "f1_after": after["f1"],
            "delta": round(after["f1"] - before["f1"], 4) + 0.0,   # + 0.0 turns -0.0 into 0.0
            "newly_wrong": flipped_bad, "newly_right": flipped_good, "regressed": regressed,
        })
        if regressed:
            report.failures.append(
                f"{field_name}: F1 {before['f1']:.3f} -> {after['f1']:.3f} "
                f"({-drop:+.3f}), {len(flipped_bad)} document(s) newly wrong")
        elif drop > gates.max_field_drop:
            report.warnings.append(
                f"{field_name}: F1 {-drop:+.3f} but only {len(flipped_bad)} document(s) flipped "
                f"(< --min-docs {gates.min_docs}); treated as noise")

    li_b = sum(1 for d in base_docs.values() if d["line_items"].get("perfect"))
    li_c = sum(1 for d in cand_docs.values() if d["line_items"].get("perfect"))
    li_flipped = sorted(d for d in base_docs if base_docs[d]["line_items"].get("perfect")
                        and not cand_docs[d]["line_items"].get("perfect"))
    li_drop = m_b["line_item_f1"] - m_c["line_item_f1"]
    report.field_changes.append({
        "field": "line_items", "f1_before": m_b["line_item_f1"], "f1_after": m_c["line_item_f1"],
        "delta": round(-li_drop, 4) + 0.0, "newly_wrong": li_flipped, "newly_right": [],
        "regressed": li_drop > gates.max_field_drop and len(li_flipped) >= gates.min_docs,
        "perfect_tables": (li_b, li_c),
    })
    if report.field_changes[-1]["regressed"]:
        report.failures.append(
            f"line_items: F1 {m_b['line_item_f1']:.3f} -> {m_c['line_item_f1']:.3f}, "
            f"{len(li_flipped)} table(s) no longer extracted perfectly")


def _document_changes(report: Report, base_docs: dict, cand_docs: dict) -> None:
    for doc_id in sorted(base_docs):
        before, after = base_docs[doc_id], cand_docs[doc_id]
        lb, lc = _labels(before), _labels(after)
        transitions = [
            {"field": f, "from": lb[f], "to": lc.get(f),
             "gt": _value(before, f), "before": _value(before, f, "pred"),
             "after": _value(after, f, "pred")}
            for f in lb if lb[f] != lc.get(f)
        ]
        if before["line_items"].get("perfect") != after["line_items"].get("perfect"):
            transitions.append({"field": "line_items",
                                "from": "perfect" if before["line_items"].get("perfect") else "imperfect",
                                "to": "perfect" if after["line_items"].get("perfect") else "imperfect"})
        entry = {"doc_id": doc_id, "changes": transitions}
        if before["document_correct"] and not after["document_correct"]:
            report.newly_failing.append(entry)
        elif not before["document_correct"] and after["document_correct"]:
            report.newly_fixed.append(entry)


def _value(doc: dict, field_name: str, key: str = "gt"):
    return next((f.get(key) for f in doc["fields"] if f["field"] == field_name), None)


def _hallucination_gate(report: Report, base_docs: dict, cand_docs: dict, gates: Gates) -> None:
    for doc_id in sorted(base_docs):
        lb, lc = _labels(base_docs[doc_id]), _labels(cand_docs[doc_id])
        for f, label in lc.items():
            if label == taxonomy.SPURIOUS and lb.get(f) != taxonomy.SPURIOUS:
                report.new_hallucinations.append(
                    {"doc_id": doc_id, "field": f, "value": _value(cand_docs[doc_id], f, "pred")})
    n = len(report.new_hallucinations)
    if n > gates.allow_new_hallucinations:
        report.failures.append(
            f"{n} new hallucinated value(s) (allowed: {gates.allow_new_hallucinations}): "
            + ", ".join(f"{h['doc_id']}.{h['field']}={h['value']!r}"
                        for h in report.new_hallucinations[:5])
            + (" ..." if n > 5 else ""))


def _pct_rise(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None or before == 0:
        return None
    return (after - before) / before


def _headline_gates(report: Report, m_b: dict, m_c: dict, gates: Gates) -> None:
    dar_drop = m_b["document_accuracy_rate"] - m_c["document_accuracy_rate"]
    one_doc = 1 / report.n_docs if report.n_docs else 0
    # Same two-condition logic as fields: past tolerance AND more than one document's worth.
    if dar_drop > gates.max_dar_drop and dar_drop > one_doc + 1e-9:
        report.failures.append(
            f"document accuracy {m_b['document_accuracy_rate']:.1%} -> "
            f"{m_c['document_accuracy_rate']:.1%} ({-dar_drop * 100:+.1f} pp)")

    cost = _pct_rise(_get(m_b, "cost_latency", "cost_per_doc_usd"),
                     _get(m_c, "cost_latency", "cost_per_doc_usd"))
    if cost is not None and cost > gates.max_cost_rise:
        report.failures.append(f"$/doc rose {cost:+.1%} (allowed {gates.max_cost_rise:+.0%})")

    p95 = _pct_rise(_get(m_b, "cost_latency", "latency", "p95_s"),
                    _get(m_c, "cost_latency", "latency", "p95_s"))
    if p95 is not None and p95 > gates.max_p95_rise:
        report.failures.append(f"latency p95 rose {p95:+.1%} (allowed {gates.max_p95_rise:+.0%})")

    auc_b, auc_c = _get(m_b, "calibration", "auc"), _get(m_c, "calibration", "auc")
    if auc_b is not None and auc_c is not None and auc_b - auc_c > gates.max_auc_drop_warn:
        report.warnings.append(f"confidence AUC {auc_b:.3f} -> {auc_c:.3f}: confidence ranks "
                               f"errors less well (warning only: noisy at this size)")


def _get(d: dict, *keys: str):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


# --- rendering -----------------------------------------------------------------
def to_markdown(r: Report) -> str:
    verdict = "PASS - no regression" if r.passed else f"FAIL - {len(r.failures)} regression(s)"
    lines = [f"# Regression check: {verdict}", "",
             f"`{r.baseline.get('version', '?')}` -> `{r.candidate.get('version', '?')}` "
             f"on the same {r.n_docs} documents (set `{r.baseline.get('set', '?')}`).", ""]
    if r.config_changes:
        lines += ["## What changed between the runs", "",
                  "Attribute the deltas below to these, and nothing else changed "
                  "(ground truth and documents are identical).", "",
                  "| setting | baseline | candidate |", "|---|---|---|"]
        lines += [f"| {k} | `{a}` | `{b}` |" for k, (a, b) in r.config_changes.items()]
        lines.append("")
    if r.failures:
        lines += ["## Failed gates", ""] + [f"- {f}" for f in r.failures] + [""]
    if r.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in r.warnings] + [""]

    lines += ["## Per field", "", "| field | F1 before | F1 after | delta | newly wrong | newly right | |",
              "|---|--:|--:|--:|---|---|---|"]
    for c in r.field_changes:
        fmt = lambda ids: ", ".join(ids[:4]) + (" ..." if len(ids) > 4 else "") if ids else "-"  # noqa: E731
        lines.append(f"| {c['field']} | {c['f1_before']:.3f} | {c['f1_after']:.3f} | {c['delta']:+.3f} "
                     f"| {fmt(c['newly_wrong'])} | {fmt(c['newly_right'])} "
                     f"| {'REGRESSED' if c['regressed'] else ''} |")

    lines += ["", "## Failures by kind (candidate minus baseline)", "", "| kind | delta |", "|---|--:|"]
    lines += [f"| {k} | {v:+d} |" for k, v in r.taxonomy_delta.items() if v]
    if not any(r.taxonomy_delta.values()):
        lines.append("| (no change) | 0 |")

    for title, docs in (("Newly failing documents", r.newly_failing),
                        ("Newly fixed documents", r.newly_fixed)):
        if not docs:
            continue
        lines += ["", f"## {title}", ""]
        for d in docs:
            changes = "; ".join(
                f"{c['field']}: {c['from']} -> {c['to']}"
                + (f" (gt {c['gt']!r}, was {c['before']!r}, now {c['after']!r})"
                   if "gt" in c and c.get("to") != taxonomy.CORRECT else "")
                for c in d["changes"])
            lines.append(f"- **{d['doc_id']}**: {changes}")

    if r.new_hallucinations:
        lines += ["", "## New hallucinations", ""]
        lines += [f"- {h['doc_id']}.{h['field']} = {h['value']!r} (ground truth: null)"
                  for h in r.new_hallucinations]

    headline = headline_markdown(r.headline).split("\n", 2)[2]   # drop its title line
    lines += ["", "## Headline", headline]
    return "\n".join(lines)


def to_json(r: Report) -> dict:
    return {
        "passed": r.passed, "baseline": r.baseline, "candidate": r.candidate, "n_docs": r.n_docs,
        "failures": r.failures, "warnings": r.warnings, "field_changes": r.field_changes,
        "newly_failing": r.newly_failing, "newly_fixed": r.newly_fixed,
        "new_hallucinations": r.new_hallucinations, "taxonomy_delta": r.taxonomy_delta,
        "config_changes": {k: list(v) for k, v in r.config_changes.items()},
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fail if the candidate run regressed.")
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    d = Gates()
    ap.add_argument("--max-field-drop", type=float, default=d.max_field_drop)
    ap.add_argument("--min-docs", type=int, default=d.min_docs)
    ap.add_argument("--max-dar-drop", type=float, default=d.max_dar_drop)
    ap.add_argument("--allow-new-hallucinations", type=int, default=d.allow_new_hallucinations)
    ap.add_argument("--max-cost-rise", type=float, default=d.max_cost_rise)
    ap.add_argument("--max-p95-rise", type=float, default=d.max_p95_rise)
    ap.add_argument("--allow-label-drift", action="store_true",
                    help="compare even though ground truth or documents changed between runs")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", type=Path, dest="json_out")
    args = ap.parse_args(argv)

    gates = Gates(args.max_field_drop, args.min_docs, args.max_dar_drop,
                  args.allow_new_hallucinations, args.max_cost_rise, args.max_p95_rise,
                  allow_label_drift=args.allow_label_drift)
    try:
        report = analyse(args.baseline, args.candidate, gates)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"invalid comparison: {exc}", file=sys.stderr)
        return EXIT_INVALID

    text = to_markdown(report)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(to_json(report), indent=2, default=str), encoding="utf-8")
    return EXIT_OK if report.passed else EXIT_REGRESSION


if __name__ == "__main__":
    sys.exit(main())
