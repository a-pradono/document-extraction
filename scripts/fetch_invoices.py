"""Fetch the 10 invoices (second document type) from RVL-CDIP-N multi-page.

    python scripts/fetch_invoices.py

Streams jordyvl/rvl_cdip_n_mp's archive and extracts ONLY the pinned PDFs below, so the
1.25 GB archive is never stored (the invoice class sits ~35% into it, so the stream stops
early). Each file is checked against its SHA-256 and the script fails loudly on any
mismatch. Documents land as data/docs/inv_XXXX.pdf with a provenance entry.

Idempotent: documents already in place with the right checksum are skipped, and an
existing label file is NEVER overwritten (labels are hand-verified work).

Why this source and not the one docs/DATASETS.md first preferred: see its addendum's
"Verified 2026-09-17" note. In short, the Hugging Face checkmark set is synthetic and
single-page; these are real, multi-page public-records invoices naming both parties.

Data notice: these are public documents (many released under freedom-of-information
requests) that name real private individuals. Keep the repository private. License:
CC BY-NC 4.0 (non-commercial research use), per the dataset card.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

from gt_common import DOCS_DIR, GT_DIR, PROVENANCE, carry_verification

URL = "https://huggingface.co/datasets/jordyvl/rvl_cdip_n_mp/resolve/main/data.tar.gz"
SOURCE = ("RVL-CDIP-N multi-page (jordyvl/rvl_cdip_n_mp; Larson et al., NeurIPS 2022 "
          "Datasets & Benchmarks; CC BY-NC 4.0), invoice class")

# (doc_id, original file, pages, sha256, vendor -> customer as printed)
PINNED = [
    ("inv_0001", "20694747-invoice.pdf", 2,
     "998a4c1aeeb483a11dec67aa260741a3345ffc5997f5c3541007d101724524c1",
     "City of Richmond Police Department -> MuckRock News (FOIA fees)"),
    ("inv_0002", "600947-american-crossroads-invoice-october-1.pdf", 2,
     "b8f8fd95c15e5b24819a7dcdf75140050c254029f117d24dd0cc0005f76eb7bf",
     "WXII-TV -> Crossroads Media (advertiser: American Crossroads)"),
    ("inv_0003", "6768458-Invoice.pdf", 2,
     "3f94df3a2b9cbb85475b2e4eb00f747fcbb88132e39b579616e221a901c7e841",
     "Flock Safety -> Cobb County Finance Department"),
    ("inv_0004", "6779372-Invoice.pdf", 2,
     "423d3627c29e11e531c6560857bc9fb9ef0c006112854328042c61b0808ac118",
     "Williamsburg-James City County Public Schools -> MuckRock"),
    ("inv_0005", "1314570-20141016142248.pdf", 3,
     "9c932060ec2acdba28484799db1b70ea21d1d2b455790953ae6bea54761bc240",
     "William H. Stewart III -> Plastic Revolutions, Inc. (handwritten annotations)"),
    ("inv_0006", "1314577-20141016142902.pdf", 3,
     "8bfd9f1e16fce6eccd15707458f2fc0fa4dbc3e7daeb5e8c8958e07571f280f8",
     "groSolar -> SolarDyne, LLC (redactions, handwritten payment note)"),
    ("inv_0007", "7331212-invoice.pdf", 3,
     "4009f91dd1358296960adfbff7f7b2ab19e839aa339e35bc584d2a9ac4eb2cdf",
     "U.S. FDA (HHS) -> requester (FOIA form layout)"),
    ("inv_0008", "20519891-correll-invoice.pdf", 4,
     "d8cf484e3b3bb72f7c50c2a0fe8848e2951ebf1eb2f0c2c37c821f731639de0d",
     "Correll Law Group -> National Rifle Association (c/o named officer)"),
    ("inv_0009", "1314575-20141016142748.pdf", 5,
     "5022db56d019aee31d3c5564f8f21edc66ef48ed9313d8cd61b420583048efca",
     "Circuit Makers Inc. -> Plastic Revolutions (redacted address)"),
    ("inv_0010", "3982955-Invoice.pdf", 5,
     "51f3286e3dab76b69c93a526acdbc5a660b6ce13d1e638f56dfc6dcfe555b516",
     "Florida Dept of Agriculture & Consumer Services -> individual (labor table)"),
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _already_in_place() -> set[str]:
    done = set()
    for doc_id, _, _, sha, _ in PINNED:
        path = DOCS_DIR / f"{doc_id}.pdf"
        if path.exists() and _sha(path.read_bytes()) == sha:
            done.add(doc_id)
    return done


def _stream_pinned(wanted: dict[str, tuple]) -> dict[str, bytes]:
    import httpx

    class Raw(io.RawIOBase):
        def __init__(self, chunks):
            self.chunks, self.buf = chunks, b""

        def readable(self):
            return True

        def readinto(self, b):
            while not self.buf:
                try:
                    self.buf = next(self.chunks)
                except StopIteration:
                    return 0
            n = min(len(b), len(self.buf))
            b[:n], self.buf = self.buf[:n], self.buf[n:]
            return n

    found: dict[str, bytes] = {}
    with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        reader = io.BufferedReader(Raw(response.iter_bytes(1 << 20)), 1 << 20)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if member.isfile() and name in wanted:
                    found[name] = tar.extractfile(member).read()
                    print(f"  extracted {name}", flush=True)
                    if len(found) == len(wanted):
                        break                      # stop streaming once all are in hand
    return found


def main() -> int:
    done = _already_in_place()
    todo = {f: (doc_id, pages, sha) for doc_id, f, pages, sha, _ in PINNED if doc_id not in done}
    if todo:
        print(f"streaming {URL} for {len(todo)} invoice(s)...")
        found = _stream_pinned(todo)
        missing = sorted(set(todo) - set(found))
        if missing:
            print(f"ERROR: not found in archive: {missing}", file=sys.stderr)
            return 1
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in found.items():
            doc_id, _, sha = todo[name]
            if _sha(data) != sha:
                print(f"ERROR: checksum mismatch for {name}; refusing to use it", file=sys.stderr)
                return 1
            (DOCS_DIR / f"{doc_id}.pdf").write_bytes(data)
    else:
        print("all 10 invoices already in place with matching checksums")

    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8")) if PROVENANCE.exists() else {}
    entries = {}
    for doc_id, name, pages, _, parties in PINNED:
        previous = provenance.get(doc_id, {})
        entries[doc_id] = {
            "source": SOURCE,
            "original_file": name,
            "document_type": "invoice",
            "language": "en",
            "page_count": pages,
            "parties": parties,
            # Nothing is script-converted: the dataset has class labels only, no fields.
            "script_converted_fields": [],
            "hand_labeled_fields": previous.get("hand_labeled_fields", []),
            "label_status": previous.get("label_status", "not labeled"),
            "needs_hand_label": previous.get("needs_hand_label", ["*"]),
            "verified": previous.get("verified", False),
            "verified_note": previous.get("verified_note"),
        }
        GT_DIR.mkdir(parents=True, exist_ok=True)
        if not (GT_DIR / f"{doc_id}.json").exists():
            (GT_DIR / f"{doc_id}.json").write_text('{"document_type": "invoice"}\n', encoding="utf-8")
    carry_verification(entries)
    provenance.update(entries)
    PROVENANCE.write_text(json.dumps(dict(sorted(provenance.items())), indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    print(f"10 invoices in {DOCS_DIR} ({sum(p for _, _, p, _, _ in PINNED)} pages); provenance updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
