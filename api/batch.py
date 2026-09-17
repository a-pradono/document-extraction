"""Folder batch runner (Phase 5).

    python -m api.batch <folder> [--out outputs/batch_<stamp>] [--concurrency 3]
                                 [--recursive] [--no-cache]

Folder in -> per-document results + a cost/latency summary out:

    results.jsonl   one ExtractionOut per line, appended as each document finishes, so
                    an interrupted batch still leaves every completed result on disk
    summary.json    counts, failures, abstention, tokens, $/doc, latency p50/p95
    summary.md      the same, readable

Concurrency is bounded: the free tier allows ~10 requests/minute, so fanning out wider
only turns throughput into 429 retries. One bad document never kills the batch — it is
recorded as status="error" and counted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from api import service
from api.schemas import ExtractionOut, from_error
from extractor.cost import StageUsage, latency_percentiles, summarize

log = logging.getLogger("api.batch")

DocFn = Callable[[Path, str], ExtractionOut]


def discover(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern)
                  if p.is_file() and p.suffix.lower() in service.SUPPORTED_SUFFIXES)


def _doc_id(path: Path, folder: Path) -> str:
    """Relative path without suffix, so a/receipt.jpg and b/receipt.jpg stay distinct."""
    return path.relative_to(folder).with_suffix("").as_posix()


def run_batch(
    folder: Path,
    out_dir: Path,
    *,
    concurrency: int = 3,
    recursive: bool = False,
    doc_fn: DocFn | None = None,
) -> dict:
    files = discover(folder, recursive)
    if not files:
        raise SystemExit(f"no supported documents in {folder} "
                         f"({', '.join(sorted(service.SUPPORTED_SUFFIXES))})")
    doc_fn = doc_fn or (lambda p, d: service.extract_document(p, d))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results_path.write_text("")

    write_lock = threading.Lock()
    results: list[ExtractionOut] = []
    started = time.perf_counter()

    def work(path: Path) -> ExtractionOut:
        doc_id = _doc_id(path, folder)
        try:
            return doc_fn(path, doc_id)
        except Exception as exc:  # noqa: BLE001 — doc_fn should not raise, but never trust it
            log.exception("unhandled failure on %s", doc_id)
            return from_error(doc_id, exc)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(work, p): p for p in files}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with write_lock:
                results.append(result)
                with results_path.open("a", encoding="utf-8") as fh:
                    fh.write(result.model_dump_json() + "\n")
            mark = "ok " if result.status == "ok" else "ERR"
            extra = f"abstained {len(result.abstained)}" if result.status == "ok" else result.error
            print(f"  [{i}/{len(files)}] {mark} {result.doc_id}  {extra}", file=sys.stderr)

    wall = time.perf_counter() - started
    summary = summarize_results(results, wall_s=wall, concurrency=concurrency)
    summary["folder"] = str(folder)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def summarize_results(results: list[ExtractionOut], *, wall_s: float, concurrency: int) -> dict:
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    stage_rows = [[StageUsage(**s.model_dump()) for s in r.usage.stages] for r in ok if r.usage]
    cost = summarize(stage_rows, [r.usage.latency_s for r in ok if r.usage])
    cost["wall_latency"] = latency_percentiles([r.usage.wall_s for r in ok if r.usage])

    abstained_fields = sum(len(r.abstained) for r in ok)
    scored_fields = sum(len(r.confidence) for r in ok)
    by_field: dict[str, int] = {}
    for r in ok:
        for f in r.abstained:
            by_field[f] = by_field.get(f, 0) + 1

    cascaded = [r for r in ok if r.cascade]
    escalated = sum(1 for r in cascaded if r.cascade.get("escalated"))
    return {
        "mode": service.mode(),
        "cascade": {
            "documents": len(cascaded),
            "escalated": escalated,
            "escalation_rate": round(escalated / len(cascaded), 4) if cascaded else None,
        } if cascaded else None,
        "n_documents": len(results),
        "succeeded": len(ok),
        "failed": len(failed),
        "failures": [{"doc_id": r.doc_id, "error": r.error} for r in failed],
        "degraded_to_single_model": sum(
            1 for r in ok if any("model_b_failed" in w for w in r.warnings)),
        "abstention": {
            "fields_scored": scored_fields,
            "fields_abstained": abstained_fields,
            "rate": round(abstained_fields / scored_fields, 4) if scored_fields else 0.0,
            "by_field": dict(sorted(by_field.items(), key=lambda kv: -kv[1])),
        },
        "cost_latency": cost,
        "throughput": {
            "concurrency": concurrency,
            "wall_s": round(wall_s, 2),
            "docs_per_min": round(len(results) / wall_s * 60, 2) if wall_s else None,
        },
    }


def to_markdown(s: dict) -> str:
    c = s["cost_latency"]
    lat, wall = c.get("latency") or {}, c.get("wall_latency") or {}
    lines = [
        "# Batch summary", "",
        f"- Folder: `{s.get('folder', '')}` · {s.get('generated_at', '')}",
        f"- Documents: **{s['n_documents']}** · succeeded **{s['succeeded']}** · "
        f"failed **{s['failed']}** · degraded to single model {s['degraded_to_single_model']}",
        f"- Abstained: {s['abstention']['fields_abstained']} of "
        f"{s['abstention']['fields_scored']} fields ({s['abstention']['rate']:.1%})",
        f"- Mode: `{s.get('mode', 'pipeline')}`"
        + (f" · escalated {s['cascade']['escalated']} of {s['cascade']['documents']} "
           f"({s['cascade']['escalation_rate']:.1%})" if s.get("cascade") else ""),
        "", "## Cost and latency", "",
        f"Cost basis: {c['cost_basis']}.",
        "",
        "| metric | value |", "|---|--:|",
        f"| cost / doc | ${c['cost_per_doc_usd']:.5f} |",
        f"| total cost | ${c['total_cost_usd']:.4f} |",
        f"| tokens / doc | {c['tokens_per_doc']:.0f} |",
        f"| model latency p50 / p95 | {lat.get('p50_s', '-')}s / {lat.get('p95_s', '-')}s |",
        f"| wall latency p50 / p95 | {wall.get('p50_s', '-')}s / {wall.get('p95_s', '-')}s |",
        f"| throughput | {s['throughput']['docs_per_min']} docs/min "
        f"@ concurrency {s['throughput']['concurrency']} |",
    ]
    if not c.get("fully_priced", True):
        lines += ["", "> Some calls had no price in `.env`; their cost is counted as $0."]
    if c["stages"]:
        lines += ["", "| stage | calls | tokens in | tokens out | $ | mean latency | cache hits |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for name, a in c["stages"].items():
            lines.append(f"| {name} | {a['calls']} | {a['prompt_tokens']} | "
                         f"{a['completion_tokens']} | {a['cost_usd']:.5f} | "
                         f"{a['mean_latency_s']}s | {a['cache_hits']} |")
    if s["failures"]:
        lines += ["", "## Failures", "", "| doc | error |", "|---|---|"]
        lines += [f"| {f['doc_id']} | {f['error']} |" for f in s["failures"]]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--concurrency", type=int,
                    default=int(os.environ.get("BATCH_CONCURRENCY", "3")))
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"not a folder: {args.folder}")
    problems = service.config_problems()
    if problems["missing"]:
        raise SystemExit(f"not configured: set {', '.join(problems['missing'])} in .env "
                         f"(see .env.example). Nothing was processed.")
    if problems["degraded"]:
        print(f"! {', '.join(problems['degraded'])} unset: running single-model, "
              f"agreement signal neutralized", file=sys.stderr)
    service.confidence_model()  # load once here, not racily inside the worker pool
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = args.out or Path("outputs") / f"batch_{stamp}"
    use_cache = not args.no_cache
    doc_fn = lambda p, d: service.extract_document(p, d, use_cache=use_cache)  # noqa: E731

    s = run_batch(args.folder, out, concurrency=args.concurrency,
                  recursive=args.recursive, doc_fn=doc_fn)
    c, lat = s["cost_latency"], s["cost_latency"]["latency"]
    timing = f"p50 {lat['p50_s']}s p95 {lat['p95_s']}s" if lat else "no latency (nothing succeeded)"
    # ASCII separators: this line goes to a console, and cp1252 Windows terminals mangle
    # the middle dot the markdown summary uses.
    print(f"\n{s['succeeded']}/{s['n_documents']} succeeded | "
          f"${c['cost_per_doc_usd']:.5f}/doc (list-price) | {timing}")
    print(f"Results: {out}")
    sys.exit(1 if s["succeeded"] == 0 else 0)


if __name__ == "__main__":
    main()
