"""Human verification of ground truth — mandatory, not optional.

    python scripts/verify_gt.py --html            # contact sheet: image beside its labels
    python scripts/verify_gt.py                   # interactive pass, records status
    python scripts/verify_gt.py --status          # how much is verified so far

Script-converted CORD/WildReceipt labels still need eyeballing (~2 min/doc), and the
header fields were drafted from the images. Verification state lives in
data/ground_truth/_provenance.json so a reviewer can see exactly which of the 65
documents a human has actually looked at.
"""

from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path

from audit_gt import audit_all
from gt_common import DOCS_DIR, GT_DIR, PROVENANCE, REPO

SETS_DIR = REPO / "data" / "sets"
REVIEW_HTML = REPO / "outputs" / "gt_review.html"


def _notes(entry: dict) -> str:
    """label_notes is a string on receipts and a list of judgement calls on invoices."""
    notes = entry.get("label_notes") or ""
    return " • ".join(notes) if isinstance(notes, list) else str(notes)


def _load_provenance() -> dict:
    return json.loads(PROVENANCE.read_text(encoding="utf-8")) if PROVENANCE.exists() else {}


def _save_provenance(data: dict) -> None:
    PROVENANCE.write_text(
        json.dumps(dict(sorted(data.items())), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _doc_path(doc_id: str) -> Path | None:
    hits = sorted(DOCS_DIR.glob(f"{doc_id}.*"))
    return hits[0] if hits else None


def _select(args) -> list[str]:
    ids = sorted(p.stem for p in GT_DIR.glob("*.json") if not p.stem.startswith("_"))
    if args.set:
        manifest = SETS_DIR / f"{args.set}.txt"
        if manifest.exists():
            wanted = {ln.strip() for ln in manifest.read_text().splitlines()
                      if ln.strip() and not ln.startswith("#")}
            ids = [i for i in ids if i in wanted]
        else:                                   # --set cord also means "the CORD source"
            ids = [i for i in ids if i.startswith(args.set)]
    if args.only_unverified:
        prov = _load_provenance()
        ids = [i for i in ids if not prov.get(i, {}).get("verified")]
    return ids


def review_order(ids: list[str]) -> list[str]:
    """Review order = how likely this label is to be wrong, from two independent signals.

    A CHECK note means the drafter could not resolve the field (a blurred header, an
    ambiguous $ or date); an audit flag means the arithmetic does not hold. The contact
    sheet and the interactive pass BOTH use this, so the page and the terminal walk the
    documents in the same sequence - they did not, and the mismatch sent a reviewer to a
    different document than the one on screen.
    """
    prov = _load_provenance()
    audit = {d["doc_id"]: d for d in audit_all()}

    def priority(doc_id: str) -> tuple:
        notes = _notes(prov.get(doc_id, {}))
        flags = [f for f in audit.get(doc_id, {}).get("flags", []) if f["severity"] != "todo"]
        return (0 if "CHECK" in notes else 1, 0 if flags else 1, -len(flags), doc_id)

    return sorted(ids, key=priority)


def write_html(ids: list[str]) -> Path:
    prov = _load_provenance()
    audit = {d["doc_id"]: d for d in audit_all()}
    ids = review_order(ids)
    rows = []
    for doc_id in ids:
        image = _doc_path(doc_id)
        labels = (GT_DIR / f"{doc_id}.json").read_text(encoding="utf-8")
        entry = prov.get(doc_id, {})
        todo = entry.get("needs_hand_label") or []
        src = html.escape(str(image.resolve().as_uri()) if image else "")
        # Multi-page invoices are PDFs: embed the whole document so every page is reviewable.
        viewer = (f'<iframe src="{src}" title="{html.escape(doc_id)}"></iframe>'
                  if image and image.suffix.lower() == ".pdf"
                  else f'<img src="{src}" alt="{html.escape(doc_id)}">')
        pages = f" &middot; {entry['page_count']} pages" if entry.get("page_count") else ""
        status = f" &middot; <b>{html.escape(entry['label_status'])}</b>" if entry.get("label_status") else ""
        badge = "verified" if entry.get("verified") else "unverified"
        notes = _notes(entry)
        note_html = (f'<p class="note{" check" if "CHECK" in notes else ""}">'
                     f"{html.escape(notes)}</p>" if notes else "")
        flags = [f for f in audit.get(doc_id, {}).get("flags", []) if f["severity"] != "todo"]
        flag_html = "".join(
            f'<li class="{f["severity"]}"><b>{html.escape(f["field"])}</b> '
            f'{html.escape(f["message"])}</li>' for f in flags)
        rows.append(f"""
<section id="{html.escape(doc_id)}">
  <h2>{html.escape(doc_id)} <span class="badge {badge}">{badge}</span>
      {f'<span class="badge flagged">{len(flags)} audit flag(s)</span>' if flags else ''}</h2>
  <div class="meta">{html.escape(entry.get('source', ''))} &middot;
    {html.escape(entry.get('original_file', ''))}{pages}{status}
    {'&middot; <b>needs hand label:</b> ' + html.escape(', '.join(todo)) if todo else ''}</div>
  {note_html}
  {f'<ul class="flags">{flag_html}</ul>' if flags else ''}
  <div class="pair">
    {viewer}
    <pre>{html.escape(labels)}</pre>
  </div>
</section>""")

    page = f"""<!doctype html><meta charset="utf-8"><title>Ground-truth review</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:#111;color:#eee}}
 h1{{font-size:18px}} h2{{font-size:15px;margin:0 0 4px}}
 section{{border-top:1px solid #333;padding:20px 0}}
 .pair{{display:flex;gap:20px;align-items:flex-start}}
 img{{max-width:46%;max-height:70vh;background:#fff;border-radius:4px}}
 iframe{{width:46%;height:80vh;border:0;background:#fff;border-radius:4px}}
 pre{{flex:1;background:#1b1b1b;padding:12px;border-radius:4px;overflow:auto;max-height:70vh}}
 .meta{{color:#999;margin-bottom:10px;font-size:12px}}
 .badge{{font-size:11px;padding:2px 6px;border-radius:3px;vertical-align:middle}}
 .verified{{background:#164;color:#cfe}} .unverified{{background:#631;color:#fed}}
 .flagged{{background:#724;color:#fde}}
 .flags{{margin:0 0 10px;padding-left:18px;font-size:13px}}
 .flags .error{{color:#f99}} .flags .check{{color:#fc8}}
</style>
<h1>Ground-truth review &mdash; {len(ids)} documents</h1>
<p class="meta">Order: documents the drafter flagged <b>CHECK</b> first, then those the
model-free audit flags, then the rest.</p>
<p class="meta">Check every field against the image. Mark status with
<code>python scripts/verify_gt.py</code>. Never adjust a value to make the arithmetic
balance &mdash; a receipt whose printed total is genuinely wrong is a valuable test case.</p>
{''.join(rows)}
"""
    REVIEW_HTML.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_HTML.write_text(page, encoding="utf-8")
    return REVIEW_HTML


def interactive(ids: list[str]) -> None:
    prov = _load_provenance()
    ids = review_order(ids)
    print(f"{len(ids)} documents. [y] verified  [n] needs fixing  [s] skip  [q] quit\n")
    for i, doc_id in enumerate(ids, 1):
        image = _doc_path(doc_id)
        print("=" * 70)
        print(f"[{i}/{len(ids)}] {doc_id}   {image if image else '(image missing!)'}")
        entry = prov.setdefault(doc_id, {})
        if entry.get("needs_hand_label"):
            print(f"  NEEDS HAND LABEL: {', '.join(entry['needs_hand_label'])}")
        note = _notes(entry)
        if note:
            marker = "CHECK -> " if "CHECK" in note else "note: "
            print(f"  {marker}{note}")
        print((GT_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
        if image:
            webbrowser.open(image.resolve().as_uri())
        answer = input("verified? [y/n/s/q] ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            continue
        entry["verified"] = answer == "y"
        note = input("note (optional): ").strip()
        entry["verified_note"] = note or None
        _save_provenance(prov)
    _save_provenance(prov)


def status() -> None:
    prov = _load_provenance()
    if not prov:
        raise SystemExit("no _provenance.json yet — run the converters first")
    by_source: dict[str, list[dict]] = {}
    for doc_id, entry in prov.items():
        by_source.setdefault(doc_id.split("_")[0], []).append(entry)
    print(f"{'source':<10}{'verified':>10}{'total':>8}{'pending hand labels':>22}")
    for source, entries in sorted(by_source.items()):
        done = sum(1 for e in entries if e.get("verified"))
        pending = sum(1 for e in entries if e.get("needs_hand_label"))
        print(f"{source:<10}{done:>10}{len(entries):>8}{pending:>22}")
    total = len(prov)
    done = sum(1 for e in prov.values() if e.get("verified"))
    print(f"\n{done}/{total} verified"
          + ("  - ground truth is NOT trustworthy until this is 100%" if done < total else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", help="a data/sets manifest name, or a source prefix (cord|wild|inv)")
    ap.add_argument("--only-unverified", action="store_true")
    ap.add_argument("--html", action="store_true", help="write a contact sheet and open it")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        status()
        return
    ids = _select(args)
    if not ids:
        raise SystemExit("nothing to review")
    if args.html:
        path = write_html(ids)
        print(f"wrote {path}")
        webbrowser.open(path.resolve().as_uri())
    else:
        interactive(ids)


if __name__ == "__main__":
    main()
