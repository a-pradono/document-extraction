"""Shared plumbing for the three dataset converters.

Every converter emits the SAME artifacts, so the parts that must not drift between
them — money normalization, schema validation, provenance bookkeeping — live here.
Source-specific parsing stays in convert_<source>.py.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # scripts/ run directly; make the package importable

from extractor.schema import Receipt  # noqa: E402

DOCS_DIR = REPO / "data" / "docs"
GT_DIR = REPO / "data" / "ground_truth"
PROVENANCE = GT_DIR / "_provenance.json"

_NUM = re.compile(r"\d")


def normalize_money(raw: Any) -> str | None:
    """Return a bare decimal string, or None if there is no number in `raw`.

    Handles both separator conventions, since CORD is Indonesian ("53.280" = fifty-three
    thousand) and WildReceipt is mostly US ("53.28" = fifty-three and a quarter):
    a trailing group of exactly 3 digits after the last separator is thousands, a group
    of 1-2 digits is a fraction.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not _NUM.search(s):
        return None
    neg = s.lstrip().startswith("-") or ("(" in s and ")" in s)
    s = re.sub(r"[^\d.,]", "", s)
    if not s:
        return None

    last = max(s.rfind("."), s.rfind(","))
    if last == -1:
        out = s
    else:
        tail = s[last + 1:]
        if len(tail) == 3 or not tail.isdigit():
            out = re.sub(r"[.,]", "", s)          # thousands separator
        else:
            out = re.sub(r"[.,]", "", s[:last]) + "." + tail
    out = out.lstrip("0") or "0"
    if out.startswith("."):
        out = "0" + out
    return ("-" + out) if neg else out


def normalize_qty(raw: Any) -> str | None:
    """Quantities like 'x2', '2 pcs', '1.0' -> '2', '2', '1'."""
    v = normalize_money(raw)
    if v is None:
        return None
    if "." in v:
        whole, frac = v.split(".", 1)
        if frac.strip("0") == "":
            return whole or "0"
    return v


def prune(d: dict) -> dict:
    """Drop keys whose value is None/empty so absent fields stay absent, not ''."""
    return {k: v for k, v in d.items() if v not in (None, "", [])}


def write_document(doc_id: str, image_src: Path, receipt: dict) -> None:
    """Validate against the schema, copy the image, write data/ground_truth/<id>.json.

    Validation happens before anything touches disk: a converter bug should fail here,
    not leave a half-written eval set behind.
    """
    Receipt.model_validate(receipt)  # raises on anything the schema won't accept
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_src, DOCS_DIR / f"{doc_id}{image_src.suffix.lower()}")
    (GT_DIR / f"{doc_id}.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def update_provenance(entries: dict[str, dict]) -> None:
    """Merge this source's entries into _provenance.json, leaving other sources alone."""
    existing: dict[str, dict] = {}
    if PROVENANCE.exists():
        existing = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    existing.update(entries)
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(
        json.dumps(dict(sorted(existing.items())), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def provenance_entry(
    source: str,
    original_file: str,
    script_fields: Iterable[str],
    hand_fields: Iterable[str] = (),
    needs_hand_label: Iterable[str] = (),
) -> dict:
    """`needs_hand_label` is fields the source dataset simply does not annotate.

    They are recorded as an open TODO rather than silently emitted as null, because a
    null that means "not on the receipt" and a null that means "nobody looked" score
    identically but mean very different things about the ground truth.
    """
    return {
        "source": source,
        "original_file": original_file,
        "script_converted_fields": sorted(script_fields),
        "hand_labeled_fields": sorted(hand_fields),
        "needs_hand_label": sorted(needs_hand_label),
        "verified": False,          # flipped by scripts/verify_gt.py
        "verified_note": None,
    }


def pinned_selection(prefix: str) -> dict[str, str]:
    """doc_id -> original file for a source's current sample, from _provenance.json.

    Converters reuse this by default. Otherwise fixing a converter changes line-item
    counts, the stratified sampler draws DIFFERENT receipts, and every split, every
    comparison and every hand-verification done so far silently refers to other
    documents. Pass --resample to redraw deliberately.
    """
    if not PROVENANCE.exists():
        return {}
    data = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    return {doc_id: entry["original_file"] for doc_id, entry in data.items()
            if doc_id.startswith(f"{prefix}_") and entry.get("original_file")}


def carry_verification(entries: dict[str, dict]) -> None:
    """Keep a document's verification status across a re-conversion ONLY if its labels
    are byte-identical to what was verified. A changed label must be looked at again."""
    if not PROVENANCE.exists():
        return
    old = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for doc_id, entry in entries.items():
        previous = old.get(doc_id, {})
        path = GT_DIR / f"{doc_id}.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None
        entry["label_sha"] = digest
        if previous.get("verified") and previous.get("label_sha") == digest:
            entry["verified"], entry["verified_note"] = True, previous.get("verified_note")
        elif previous.get("verified"):
            entry["verified_note"] = "labels changed since verification: re-verify"


def clear_source(prefix: str) -> None:
    """Remove a source's previous output so re-running a converter is idempotent."""
    for path in list(DOCS_DIR.glob(f"{prefix}_*")) + list(GT_DIR.glob(f"{prefix}_*.json")):
        path.unlink()


def stratified_sample(
    pool: list[tuple[str, int, str]], n: int, seed: int = 20240917
) -> list[str]:
    """Pick `n` ids spread across the line-item-count range and across merchants.

    `pool` is (id, n_line_items, merchant_key). Taking the first n would bias the eval
    set toward whatever order the dataset ships in — usually short, clean receipts —
    and flatter every metric downstream. Instead: bucket by item count, round-robin the
    buckets, and prefer an unseen merchant at each pick.
    """
    import random

    rng = random.Random(seed)
    buckets: dict[str, list[tuple[str, int, str]]] = {}
    for item in pool:
        count = item[1]
        key = "short" if count <= 3 else "medium" if count <= 7 else "long" if count <= 14 else "xlong"
        buckets.setdefault(key, []).append(item)
    for items in buckets.values():
        rng.shuffle(items)

    order = [k for k in ("short", "medium", "long", "xlong") if k in buckets]
    picked: list[str] = []
    seen_merchants: set[str] = set()
    while len(picked) < n and any(buckets[k] for k in order):
        for key in order:
            if len(picked) >= n or not buckets[key]:
                continue
            candidates = buckets[key]
            idx = next(
                (i for i, c in enumerate(candidates) if c[2] not in seen_merchants), 0
            )
            doc_id, _, merchant = candidates.pop(idx)
            seen_merchants.add(merchant)
            picked.append(doc_id)
    return picked
