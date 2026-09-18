"""Extraction review dashboard: every source document beside what the system read.

    python -m eval.dashboard --set calibration            # writes outputs/dashboard.html
    python -m eval.dashboard --set smoke --open

One self-contained HTML file: images are embedded as data URIs, so it can be opened
over file:// or handed to someone with nothing else attached. It is built from the
SAME payload `POST /extract` returns (`api/schemas.ExtractionOut`), so what a reviewer
reads here is what a caller gets, field for field.

Like the per-document card (eval/render_doc.py) it shows NO ground truth and recomputes
nothing: the arithmetic verdicts come from the payload's `validation` block. Accuracy
questions belong in the scorecard; this answers "what did it read, and why believe it".

Cached documents cost nothing to re-render, so iterating on the layout is free.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from api.schemas import from_result
from eval import run as eval_run
from extractor.pipeline import PipelineResult
from extractor.schema import document_type_of

OUT_DEFAULT = Path("outputs/dashboard.html")
THUMB_WIDTH = 900          # enough to read a receipt on a laptop, small enough to embed
JPEG_QUALITY = 72

# Invoice party fields are shown as one block per role, the way a human reads them.
PARTY_FIELDS = ("name", "address", "tax_id")
LABELS = {
    "merchant_name": "Merchant", "merchant_address": "Address", "date": "Date",
    "currency": "Currency", "subtotal": "Subtotal", "service_charge": "Service",
    "discount": "Discount", "tax": "Tax", "total": "Total",
    "payment_method": "Payment method", "invoice_number": "Invoice #",
    "purchase_order": "PO", "issue_date": "Issued", "due_date": "Due",
    "vendor": "Vendor", "customer": "Customer",
}
DASH = "&mdash;"


# --- source images -----------------------------------------------------------
def thumbnail(path: Path) -> tuple[str, int]:
    """Return a data URI for the document, plus its page count.

    Multi-page invoices are PDFs: the first page is rendered, and the page count is
    shown as a badge so a reviewer knows there is more behind it.
    """
    pages = 1
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        pages = len(pdf)
        image = pdf[0].render(scale=2).to_pil()
    else:
        image = ImageOps.exif_transpose(Image.open(path))
    image = image.convert("RGB")
    if image.width > THUMB_WIDTH:
        height = round(image.height * THUMB_WIDTH / image.width)
        image = image.resize((THUMB_WIDTH, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
    uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return uri, pages


# --- payload -> rows ---------------------------------------------------------
def _money(value) -> str:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return html.escape(str(value))
    text = f"{number:,.2f}"
    return text[:-3] if text.endswith(".00") else text


MONEY_FIELDS = {"subtotal", "service_charge", "discount", "tax", "total"}


def _rows(payload: dict) -> list[dict]:
    """One display row per field, in document order, parties kept together."""
    document = payload.get("document") or {}
    abstained = set(payload.get("abstained") or [])
    withheld = payload.get("withheld") or {}
    confidence = payload.get("confidence") or {}
    signals = payload.get("signals") or {}
    rows: list[dict] = []

    def value_of(field: str):
        if field in abstained:
            return withheld.get(field), "abstained"
        raw = document
        for part in field.split("."):
            raw = (raw or {}).get(part) if isinstance(raw, dict) else None
        if raw is None or raw == "":
            return None, "absent"
        return raw, "ok"

    for key, value in document.items():
        if key == "line_items":
            continue
        if isinstance(value, dict):                       # vendor / customer
            parts, statuses, confs = [], [], []
            for sub in PARTY_FIELDS:
                field = f"{key}.{sub}"
                shown, status = value_of(field)
                statuses.append(status)
                if field in confidence:
                    confs.append(confidence[field])
                if shown:
                    parts.append(html.escape(str(shown)))
            status = ("abstained" if "abstained" in statuses
                      else "absent" if not parts else "ok")
            rows.append({
                "label": LABELS.get(key, key), "field": key,
                "html": "<br>".join(parts) if parts else DASH,
                "conf": min(confs) if confs else None, "status": status,
                "signals": signals.get(f"{key}.name"),
            })
            continue
        shown, status = value_of(key)
        text = DASH if shown is None else (
            _money(shown) if key in MONEY_FIELDS else html.escape(str(shown)))
        rows.append({
            "label": LABELS.get(key, key), "field": key, "html": text,
            "conf": confidence.get(key), "status": status, "signals": signals.get(key),
        })
    return rows


def _line_items(payload: dict) -> str:
    document = payload.get("document") or {}
    abstained = set(payload.get("abstained") or [])
    items = document.get("line_items") or []
    withheld_table = "line_items" in abstained
    if withheld_table:
        raw = (payload.get("withheld") or {}).get("line_items")
        items = json.loads(raw) if raw else []
    if not items:
        return "<h4>Line items</h4><p class=sm>none read</p>"

    body = "".join(
        "<tr>"
        f"<td class=n>{n}</td><td>{html.escape(str(i.get('description') or ''))}</td>"
        f"<td class=n>{'' if i.get('quantity') is None else _money(i['quantity'])}</td>"
        f"<td class=n>{'' if i.get('unit_price') is None else _money(i['unit_price'])}</td>"
        f"<td class=n>{DASH if i.get('line_total') is None else _money(i['line_total'])}</td>"
        "</tr>" for n, i in enumerate(items, 1))

    validation = payload.get("validation") or {}
    notes = payload.get("validation_notes") or {}
    verdict = validation.get("line_items_sum_to_subtotal")
    totals = [i.get("line_total") for i in items]
    shown_sum = DASH
    if all(t is not None for t in totals):
        try:
            shown_sum = _money(sum(float(t) for t in totals))
        except (TypeError, ValueError):
            pass
    if verdict is True:
        cls, foot = "sum-ok", f"<b>{shown_sum}</b> &#10003; <span class=sm>matches subtotal</span>"
    elif verdict is False:
        cls = "sum-bad"
        foot = (f"<b>{shown_sum}</b> <span class=sm>"
                f"{html.escape(notes.get('line_items_sum_to_subtotal', ''))}</span>")
    else:
        cls, foot = "sum-na", f"<b>{shown_sum}</b> <span class=sm>nothing to check against</span>"

    banner = ("<p class='sm warn-note'>Table withheld: confidence below threshold, shown "
              "for review but not asserted.</p>" if withheld_table else "")
    return (f"<h4>Line items</h4>{banner}<table class=li>"
            "<tr><th>#</th><th>description</th><th>qty</th><th>unit</th><th>line total</th></tr>"
            f"{body}<tr class='{cls}'><td colspan=3></td><td class=n><b>sum</b></td>"
            f"<td class=n>{foot}</td></tr></table>")


def _chips(payload: dict) -> str:
    validation = payload.get("validation") or {}
    notes = payload.get("validation_notes") or {}
    chips = []
    for name, value in validation.items():
        if value is None:
            continue                                   # not applicable: say nothing
        cls = "ch-ok" if value else "ch-bad"
        chips.append(f'<span class="chip {cls}">{html.escape(name.replace("_", " "))}</span>')
    detail = "".join(f'<span class="sm bad-detail">{html.escape(v)}</span>'
                     for v in notes.values())
    na = [k for k, v in validation.items() if v is None]
    if na:
        detail += (f'<span class=sm title="{html.escape(", ".join(na))}">'
                   f"{len(na)} check(s) did not apply</span>")
    return f"<div class=chips>{''.join(chips)}{detail}</div>"


def _status_of(payload: dict) -> str:
    validation = payload.get("validation") or {}
    if payload.get("status") == "error":
        return "failed"
    if any(v is False for v in validation.values()):
        return "failed"
    return "abstained" if payload.get("abstained") else "clean"


def card(payload: dict, doc_path: Path) -> str:
    doc_id = payload.get("doc_id", "doc")
    doc_type = payload.get("document_type", "receipt")
    status = _status_of(payload)
    rows = _rows(payload)
    confs = [r["conf"] for r in rows if r["conf"] is not None]
    avg = f"avg conf {sum(confs) / len(confs):.2f}" if confs else "no confidence model"
    uri, pages = thumbnail(doc_path)

    badge = {"clean": '<span class="badge b-ok">all checks passed</span>',
             "abstained": f'<span class="badge b-abs">{len(payload.get("abstained") or [])} abstained</span>',
             "failed": '<span class="badge b-fail">check failed</span>'}[status]

    body = ""
    for row in rows:
        title = ""
        if row["signals"]:
            title = ' title="' + html.escape(
                " | ".join(f"{k}: {v:.2f}" for k, v in row["signals"].items())) + '"'
        conf = "" if row["conf"] is None else f"{row['conf']:.2f}"
        mark = {"ok": "", "abstained": "abstained", "absent": "absent"}[row["status"]]
        body += (f'<tr class=st-{row["status"]}{title}><td class=fl>{html.escape(row["label"])}</td>'
                 f'<td class=fv>{row["html"]}</td><td class=fc>{conf}</td>'
                 f"<td class=fs>{mark}</td></tr>")

    # A document that failed must say so ON the card. Rendering an empty field table
    # with no explanation leaves a reviewer thinking the model read nothing, when in
    # fact the call never completed.
    problems = [payload["error"]] if payload.get("error") else []
    problems += list(payload.get("warnings") or [])
    problem_html = ("<div class=problems><b>Problem</b>" + "".join(
        f"<span>{html.escape(str(p))}</span>" for p in problems) + "</div>") if problems else ""

    usage = payload.get("usage") or {}
    stages = usage.get("stages") or []
    tokens = sum(s.get("prompt_tokens", 0) + s.get("completion_tokens", 0) for s in stages)
    foot = (f"{usage.get('latency_s', 0):.1f}s &middot; {tokens:,} tokens &middot; "
            f"${usage.get('cost_usd', 0):.5f}")
    transcript = payload.get("transcription") or ""
    details = (f"<details><summary>Transcription &middot; {len(transcript):,} chars</summary>"
               f"<pre>{html.escape(transcript)}</pre></details>" if transcript else "")
    pages_badge = f'<span class="pill">{pages} pages</span>' if pages > 1 else ""

    return f"""
<article class=card data-type="{doc_type}" data-id="{html.escape(doc_id)}" data-status="{status}">
  <header>
    <span class=did>{html.escape(doc_id)}</span>
    <span class="pill tp-{doc_type}">{doc_type}</span>{pages_badge}
    <span class=srcname>{html.escape(doc_path.name)}</span>
    <span class=spacer></span>
    {badge}<span class="badge b-conf">{avg}</span>
  </header>
  <div class=split>
    <div class=src><img src="{uri}" alt="source document {html.escape(doc_id)}" loading=lazy></div>
    <div class=data>
      {problem_html}
      <table class=fields><tr><th>field</th><th>value</th><th>conf</th><th></th></tr>{body}</table>
      {_line_items(payload)}
      {_chips(payload)}
      <p class="sm foot">{foot}</p>
      {details}
    </div>
  </div>
</article>"""


# --- page --------------------------------------------------------------------
_CSS = (Path(__file__).parent / "dashboard.css").read_text(encoding="utf-8")


def page(cards: list[str], meta: dict) -> str:
    counts = meta["counts"]
    stats = "".join(
        f'<div class="stat {cls}"><div class=v><span data-count="{value}"'
        f'{" data-pct=1" if pct else ""}>0{"%" if pct else ""}</span></div>'
        f"<div class=l>{label}</div></div>"
        for value, label, cls, pct in [
            (counts["total"], "documents", "", False),
            (counts["receipt"], "receipts", "a", False),
            (counts["invoice"], "invoices", "a", False),
            (counts["clean_pct"], "all checks passed", "a", True),
            (counts["abstained"], "with abstentions", "w", False),
            (counts["failed"], "failed a check", "b", False),
        ])
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Extraction review &mdash; {html.escape(meta['title'])}</title>
<style>{_CSS}</style></head><body>
<div class=wrap>
  <div class=hero>
    <span class=eyebrow><span class=dot></span>Document intelligence</span>
    <h1>Extraction review</h1>
    <p class=lede>Source document on the left, structured output on the right &mdash; with
      per-field confidence, abstentions, and the deterministic arithmetic checks that no
      model can bluff. {html.escape(meta['subtitle'])}</p>
    <div class=stats>{stats}</div>
  </div>

  <div class=toolbar>
    <div class=trow>
      <span class=glabel>Type</span>
      <div class=seg>
        <button data-type-filter=all>All<span class=cnt>{counts['total']}</span></button>
        <button data-type-filter=receipt>Receipts<span class=cnt>{counts['receipt']}</span></button>
        <button data-type-filter=invoice>Invoices<span class=cnt>{counts['invoice']}</span></button>
      </div>
      <span class=spacer></span>
      <span class=cnt id=shown></span>
      <div class=nav>
        <button id=prev>&lsaquo; Prev</button>
        <span class=pg id=pg></span>
        <button id=next>Next &rsaquo;</button>
      </div>
    </div>
    <div class=trow>
      <span class=glabel>Status</span>
      <div class=seg>
        <button data-status=all>All<span class=cnt>{counts['total']}</span></button>
        <button data-status=clean>Clean<span class=cnt>{counts['clean']}</span></button>
        <button data-status=abstained>Abstained<span class=cnt>{counts['abstained']}</span></button>
        <button data-status=failed>Failed check<span class=cnt>{counts['failed']}</span></button>
      </div>
    </div>
  </div>

  <div id=grid>{''.join(cards)}</div>
  <div class="empty hidden" id=empty>No documents match this filter.</div>

  <footer>
    <span>{counts['total']} documents &middot; {meta['model_a']} + {meta['model_b']}
      &middot; threshold {meta['threshold']}</span>
    <span class=spacer></span>
    <span>Navigate with <kbd>&larr;</kbd> <kbd>&rarr;</kbd></span>
  </footer>
</div>
<script>{_JS}</script>
</body></html>"""


_JS = """
const PAGE=5;
let fType='all', fStatus='all', page=1;
const cards=[...document.querySelectorAll('.card')];
const grid=document.getElementById('grid');
const empty=document.getElementById('empty');
function match(c){
  return (fType==='all'||c.dataset.type===fType) &&
         (fStatus==='all'||c.dataset.status===fStatus);
}
function render(){
  const vis=cards.filter(match);
  const pages=Math.max(1,Math.ceil(vis.length/PAGE));
  if(page>pages) page=pages;
  cards.forEach(c=>c.classList.add('hidden'));
  vis.slice((page-1)*PAGE,page*PAGE).forEach((c,i)=>{
    c.classList.remove('hidden');
    c.style.animation='none'; void c.offsetWidth;
    c.style.animation=''; c.style.animationDelay=(0.02+i*0.05)+'s';});
  empty.classList.toggle('hidden',vis.length>0);
  document.getElementById('shown').textContent =
    vis.length ? `${(page-1)*PAGE+1}\\u2013${Math.min(page*PAGE,vis.length)} of ${vis.length}` : '0';
  document.getElementById('pg').textContent=`${page} / ${pages}`;
  document.getElementById('prev').disabled=page<=1;
  document.getElementById('next').disabled=page>=pages;
  document.querySelectorAll('[data-type-filter]').forEach(b=>
    b.classList.toggle('on',b.dataset.typeFilter===fType));
  document.querySelectorAll('[data-status]').forEach(b=>
    b.classList.toggle('on',b.dataset.status===fStatus));
}
document.querySelectorAll('[data-type-filter]').forEach(b=>
  b.onclick=()=>{fType=b.dataset.typeFilter;page=1;render();grid.scrollIntoView({block:'start'})});
document.querySelectorAll('[data-status]').forEach(b=>
  b.onclick=()=>{fStatus=b.dataset.status;page=1;render();grid.scrollIntoView({block:'start'})});
document.getElementById('prev').onclick=()=>{if(page>1){page--;render();grid.scrollIntoView({block:'start'})}};
document.getElementById('next').onclick=()=>{page++;render();grid.scrollIntoView({block:'start'})};
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='ArrowLeft')document.getElementById('prev').click();
  if(e.key==='ArrowRight')document.getElementById('next').click();
});
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
document.querySelectorAll('[data-count]').forEach(el=>{
  const target=parseFloat(el.dataset.count), pct=el.dataset.pct==='1';
  if(reduce){el.textContent=pct?target.toFixed(0)+'%':target;return}
  let t0=null;
  const step=ts=>{if(!t0)t0=ts;const p=Math.min(1,(ts-t0)/700);
    const e=1-Math.pow(1-p,3);const v=target*e;
    el.textContent=pct?v.toFixed(0)+'%':Math.round(v);
    if(p<1)requestAnimationFrame(step)};
  requestAnimationFrame(step);
});
render();
"""


# --- build -------------------------------------------------------------------
def build(doc_ids: list[str], *, source: str = "pipeline", use_cache: bool = True,
          title: str = "document extraction") -> str:
    model, fitted = eval_run._load_model()
    threshold = eval_run.THRESHOLD
    if not fitted:
        print("! no fitted confidence model - run `python -m eval.run --calibrate` first",
              file=sys.stderr)

    cards, payloads = [], []
    for doc_id in doc_ids:
        doc_type = document_type_of(doc_id)
        doc_path = eval_run._find_doc(doc_id)
        try:
            ext = eval_run.extract_one(doc_path, source, model, threshold, use_cache, doc_type)
            payload = json.loads(from_result(doc_id, PipelineResult(
                document=ext.document, document_b=None, transcription=ext.transcription,
                confidence=ext.confidence, abstained=ext.abstained, signals=ext.signals,
                latency_s=ext.latency_s, tokens=ext.tokens, stages=ext.stages,
                cost_usd=ext.cost_usd, errors=ext.errors, cascade=ext.cascade,
            ), threshold, doc_type).model_dump_json())
        except Exception as exc:  # noqa: BLE001 - one bad document must not sink the page
            print(f"  FAILED {doc_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            payload = {"doc_id": doc_id, "document_type": doc_type, "status": "error",
                       "document": None, "error": f"{type(exc).__name__}: {exc}"}
        payloads.append(payload)
        cards.append(card(payload, doc_path))
        print(f"  rendered {doc_id}", file=sys.stderr)

    statuses = [_status_of(p) for p in payloads]
    types = [p.get("document_type") for p in payloads]
    counts = {
        "total": len(payloads),
        "receipt": types.count("receipt"),
        "invoice": types.count("invoice"),
        "clean": statuses.count("clean"),
        "abstained": statuses.count("abstained"),
        "failed": statuses.count("failed"),
    }
    counts["clean_pct"] = round(100 * counts["clean"] / max(1, counts["total"]))
    import os

    meta = {
        "title": title, "counts": counts,
        "model_a": os.environ.get("VLM_A_MODEL", "model A"),
        "model_b": os.environ.get("LLM_B_MODEL", "model B"),
        "threshold": f"{threshold:.3f}" if fitted else f"{threshold:.3f} (unfitted)",
        "subtitle": ("Confidence comes from a model fitted on the calibration split."
                     if fitted else "No fitted confidence model: scores are fallback weights."),
    }
    return page(cards, meta)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_name", default="calibration",
                    help="a data/sets manifest name (default: calibration)")
    ap.add_argument("--ids", nargs="*", help="explicit document ids instead of a set")
    ap.add_argument("--source", default="pipeline", choices=["pipeline", "cascade", "vlm-only"])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--no-cache", action="store_true", help="re-extract, spending API calls")
    ap.add_argument("--open", action="store_true", help="open it when done")
    args = ap.parse_args(argv)

    ids = args.ids or eval_run._doc_ids(args.set_name)
    html_text = build(ids, source=args.source, use_cache=not args.no_cache,
                      title=args.set_name if not args.ids else "selected documents")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    size = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out} ({size:.1f} MB, {len(ids)} documents)")
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == "__main__":
    main()
