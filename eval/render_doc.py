"""One document's extraction, rendered for a human.

    python -m eval.render_doc <extraction.json> [-o out.html]

`scorecard.html` answers "how good is the system"; this answers "what did it read off
THIS document, and why should I believe it". It takes the exact payload `POST /extract`
returns (or a line of a batch `results.jsonl`) and emits self-contained HTML — inline
CSS, no CDN, because it is opened over `file://`.

Two rules keep it honest:
  - It recomputes NOTHING. The arithmetic verdicts come from the payload's `validation`
    block, so the page and the API can never disagree about whether a receipt added up.
  - It shows no ground truth. "What the system extracted" and "was it right" are
    different questions, and mixing them hides the case where a confident, internally
    consistent extraction is simply wrong.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Optional

from eval.scorecard import _CSS

DASH = "&mdash;"


def _flatten(document: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Scalar fields in payload order, nested parties as dotted names — the same names
    `confidence`, `abstained` and `withheld` use, so every block lines up."""
    rows: list[tuple[str, Any]] = []
    for key, value in document.items():
        if key == "line_items":
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(_flatten(value, f"{name}."))
        else:
            rows.append((name, value))
    return rows


def _signal_title(signals: dict, field: str) -> str:
    s = signals.get(field)
    if not s:
        return ""
    text = " | ".join(f"{k}: {v:.2f}" for k, v in s.items())
    return f' title="{html.escape(text)}"'


def _conf(confidence: dict, field: str) -> str:
    v = confidence.get(field)
    return DASH if v is None else f"{v:.2f}"


def _field_rows(payload: dict) -> str:
    document = payload.get("document") or {}
    abstained = set(payload.get("abstained") or [])
    withheld = payload.get("withheld") or {}
    confidence = payload.get("confidence") or {}
    signals = payload.get("signals") or {}
    out = []
    for field, value in _flatten(document):
        if field in abstained:
            # The value was read but withheld at the API boundary. Show what was read,
            # marked as not asserted: hiding it entirely would make abstention look like
            # a field that is simply absent, which is the one distinction that matters.
            raw = withheld.get(field)
            shown = DASH if raw is None else html.escape(str(raw))
            cls, status = "abst", "abstained"
            value_html = f'<span class=abst>{shown}</span>'
        elif value is None or value == "":
            cls, status, value_html = "absent", "absent", DASH
        else:
            cls, status, value_html = "ok", "ok", html.escape(str(value))
        out.append(
            f'<tr{_signal_title(signals, field)}><td>{html.escape(field)}</td>'
            f"<td>{value_html}</td><td>{_conf(confidence, field)}</td>"
            f'<td class={cls}>{status}</td></tr>')
    return "".join(out)


def _line_items(payload: dict) -> str:
    document = payload.get("document") or {}
    abstained = set(payload.get("abstained") or [])
    items = document.get("line_items") or []
    withheld_table = False
    if "line_items" in abstained:
        raw = (payload.get("withheld") or {}).get("line_items")
        items = json.loads(raw) if raw else []
        withheld_table = True
    if not items:
        return "<h2>Line items</h2><p class=k>none read.</p>"

    rows = "".join(
        "<tr>"
        f"<td>{n}</td><td>{html.escape(str(i.get('description') or ''))}</td>"
        f"<td>{i.get('quantity') if i.get('quantity') is not None else DASH}</td>"
        f"<td>{i.get('unit_price') if i.get('unit_price') is not None else DASH}</td>"
        f"<td>{i.get('line_total') if i.get('line_total') is not None else DASH}</td></tr>"
        for n, i in enumerate(items, 1))

    # The most persuasive element on the page: a model-free check that cannot be bluffed.
    # The VERDICT is the payload's, never recomputed here.
    validation = payload.get("validation") or {}
    notes = payload.get("validation_notes") or {}
    verdict = validation.get("line_items_sum_to_subtotal")
    totals = [i.get("line_total") for i in items]
    shown_sum = DASH
    if all(t is not None for t in totals):
        try:
            shown_sum = f"{sum(float(t) for t in totals):,.2f}"
        except (TypeError, ValueError):
            shown_sum = DASH
    if verdict is True:
        footer = f"<b>{shown_sum}</b> &#10003; matches subtotal"
    elif verdict is False:
        footer = f'<b class=wrong>{html.escape(notes.get("line_items_sum_to_subtotal", shown_sum))}</b>'
    else:
        footer = f"<b>{shown_sum}</b> <span class=k>(no subtotal to check against)</span>"

    banner = ("<p class=abst>This table was withheld: confidence fell below the "
              "threshold, so it is shown for review but not asserted.</p>"
              if withheld_table else "")
    return (f"<h2>Line items</h2>{banner}<table>"
            "<tr><th>#</th><th>description</th><th>qty</th><th>unit price</th>"
            f"<th>line total</th></tr>{rows}"
            f'<tr class=sum><td colspan=4>sum</td><td>{footer}</td></tr></table>')


def _validation(payload: dict) -> str:
    validation = payload.get("validation") or {}
    notes = payload.get("validation_notes") or {}
    if not validation:
        return ""
    chips = []
    for name, value in validation.items():
        cls = "pass" if value is True else "fail" if value is False else "na"
        mark = "&#10003;" if value is True else "&#10007;" if value is False else "n/a"
        note = notes.get(name)
        title = f' title="{html.escape(note)}"' if note else ""
        chips.append(f'<span class="chip {cls}"{title}>{html.escape(name)} {mark}</span>')
    failed = "".join(f"<li class=wrong>{html.escape(k)}: {html.escape(v)}</li>"
                     for k, v in notes.items())
    detail = f"<ul>{failed}</ul>" if failed else ""
    return ("<h2>Model-free validation</h2>"
            "<p class=k>Deterministic checks, independent of either model. "
            "<b>n/a</b> means the check did not apply &mdash; not that it passed.</p>"
            f"{''.join(chips)}{detail}")


def _footer(payload: dict) -> str:
    usage = payload.get("usage") or {}
    stages = usage.get("stages") or []
    prompt = sum(s.get("prompt_tokens", 0) for s in stages)
    completion = sum(s.get("completion_tokens", 0) for s in stages)
    stage_rows = "".join(
        f"<tr><td>{html.escape(s.get('stage', ''))}</td><td>{html.escape(s.get('model', ''))}</td>"
        f"<td>{s.get('prompt_tokens', 0)}</td><td>{s.get('completion_tokens', 0)}</td>"
        f"<td>{s.get('latency_s', 0):.2f}</td><td>${s.get('cost_usd', 0):.5f}</td>"
        f"<td>{'cached' if s.get('cached') else ''}</td></tr>" for s in stages)
    table = ("<table><tr><th>stage</th><th>model</th><th>prompt</th><th>completion</th>"
             f"<th>s</th><th>cost</th><th></th></tr>{stage_rows}</table>" if stage_rows else "")
    return (f"<h2>Cost and latency</h2><p class=k>{usage.get('latency_s', 0):.2f}s model time "
            f"&middot; {prompt} prompt + {completion} completion tokens &middot; "
            f"${usage.get('cost_usd', 0):.5f} list-price equivalent</p>{table}")


def _problems(payload: dict) -> str:
    items = list(payload.get("warnings") or [])
    if payload.get("error"):
        items.insert(0, payload["error"])
    if not items:
        return ""
    lis = "".join(f"<li class=wrong>{html.escape(str(i))}</li>" for i in items)
    return f"<h2>Problems</h2><ul>{lis}</ul>"


def render(payload: dict) -> str:
    doc_id = payload.get("doc_id", "document")
    doc_type = payload.get("document_type", "receipt")
    threshold = payload.get("threshold")
    abstained = payload.get("abstained") or []
    transcription = payload.get("transcription") or ""
    head = (f"<p class=k>{html.escape(doc_type)} &middot; status: "
            f"{html.escape(payload.get('status', 'ok'))}"
            + (f" &middot; threshold {threshold:.2f}" if isinstance(threshold, (int, float)) else "")
            + f" &middot; {len(abstained)} field(s) abstained</p>")
    transcript = (f"<details><summary>Transcription ({len(transcription)} chars) &mdash; "
                  f"the grounding source</summary><pre>{html.escape(transcription)}</pre>"
                  "</details>" if transcription else "")
    return (f'<!doctype html><meta charset="utf-8"><title>{html.escape(doc_id)}</title>'
            f"<style>{_CSS}</style>"
            f"<h1>{html.escape(doc_id)}</h1>{head}"
            f"{_problems(payload)}"
            "<h2>Fields</h2><table><tr><th>field</th><th>value</th><th>conf</th>"
            f"<th>status</th></tr>{_field_rows(payload)}</table>"
            f"{_line_items(payload)}{_validation(payload)}{_footer(payload)}{transcript}")


def write_card(payload: dict, out_dir: Path) -> Path:
    """Write the card and the payload it was rendered from.

    Keeping the JSON beside the HTML means a card can be re-rendered after a change to
    this renderer without re-running the models, and it is the exact object `/extract`
    returned, so the run is auditable field by field."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id = payload.get("doc_id", "document")
    (out_dir / f"{doc_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path = out_dir / f"{doc_id}.html"
    path.write_text(render(payload), encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("payload", type=Path, help="an ExtractionOut JSON file")
    ap.add_argument("-o", "--out", type=Path, help="default: alongside the input, .html")
    args = ap.parse_args(argv)

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    out = args.out or args.payload.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
