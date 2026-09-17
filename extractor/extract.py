"""Single-document extraction — the walking skeleton (Phase 2).

image/PDF in -> every page downscaled -> one VLM call (JSON mode) -> validated Receipt
or Invoice out, with one corrective retry on invalid JSON. Also home to page preparation, which the
full pipeline and the cascade share.

Run:  python -m extractor.extract data/sample.jpg

This is Path A (VLM image -> JSON) only. Phase 4 adds Path B (OCR -> LLM) and
combines them into confidence + abstention.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps
from pydantic import BaseModel

from extractor.output import _strip_fences
from extractor.provider import Provider
from extractor.schema import DOCUMENT_MODELS

load_dotenv()

MAX_EDGE = 1600  # downscale longest side -> fewer vision tokens, lower cost/latency

# Real-model trial: given "TOTAL Rp 55.440", a model returned 55.44 and scaled EVERY
# amount the same way (48.000 -> 48.0). The arithmetic still reconciled and "55.440"
# can legitimately be read as 55.44, so neither model-free signal caught it; only
# disagreement with the other model could. With this rule the same model returned 55440.
NUMBER_RULES = (
    "Amounts: output plain numbers with no thousands separators and no currency symbols. "
    "Receipts group thousands with '.' or ',' depending on locale: when the last separator "
    "is followed by exactly three digits it groups thousands, so Indonesian '53.280' and "
    "'53,280' both mean 53280, while '53.28' means 53.28."
)

# Invoice-only guidance. Role resolution is the graded part, and the two parties look
# alike on the page, so the rule for telling them apart is stated rather than left to the
# schema description alone.
INVOICE_RULES = (
    "This is an INVOICE. vendor = the party that issued it and is to be paid (letterhead, "
    "'Remit to', 'Make checks payable to'). customer = the party billed ('Bill To', 'To:', "
    "'Sold To'); a 'c/o' or 'Attention' person belongs to the customer. Never copy one "
    "party's details into the other. A redacted (blacked-out) value is not visible: null."
)

SYSTEM_PROMPT = (
    "You are a precise receipt and invoice extraction system. Extract only what is visibly "
    "present on the document. If a field is not present, return null. Never guess "
    "or invent values. Return a single JSON object and nothing else."
)


PDF_DPI = 200
MAX_PAGES = int(os.environ.get("MAX_PAGES", "5"))


class TooManyPages(ValueError):
    pass


def prepare_pages(path: str | Path) -> list[Path]:
    """Every page of a document as a downscaled PNG, in page order.

    Multi-page PDFs are rendered in FULL and sent to the model together: a long receipt
    or an invoice-style bill puts its line items on page 1 and its total on the last
    page, so reading only page 1 is how a missing or invented total gets in. A document
    over MAX_PAGES is refused rather than truncated for the same reason.

    Rendering uses pypdfium2 (a self-contained wheel) rather than pdf2image, which needs
    the poppler binaries and silently made every PDF fail on a machine without them.
    """
    path = Path(path)
    source = path.read_bytes()
    # Named by content, not by filename: concurrent requests routinely share a stem
    # ("receipt.jpg"), and a stem-named temp file let one request's image overwrite
    # another's mid-flight. Content naming also keeps the image-hash cache stable.
    digest = hashlib.sha256(source + f"|{MAX_EDGE}|{PDF_DPI}".encode()).hexdigest()[:24]

    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(source)
        try:
            n = len(doc)
            if n == 0:
                raise ValueError(f"{path.name}: PDF has no pages")
            if n > MAX_PAGES:
                raise TooManyPages(f"{path.name}: {n} pages exceeds MAX_PAGES={MAX_PAGES}; "
                                   f"refusing to extract a truncated document")
            outs = [_page_path(digest, i) for i in range(n)]
            for i, out in enumerate(outs):
                if not out.exists():
                    _save(doc[i].render(scale=PDF_DPI / 72).to_pil(), out)
            return outs
        finally:
            doc.close()

    out = _page_path(digest, 0)
    if not out.exists():
        with Image.open(path) as img:
            # Phone photos store rotation as EXIF metadata that PIL ignores by default,
            # so without this the model is handed a sideways receipt.
            _save(ImageOps.exif_transpose(img), out)
    return [out]


def prepare_image(path: str | Path) -> Path:
    """Single-image convenience for callers that cannot take several pages. Refuses a
    multi-page document instead of quietly handing back page 1."""
    pages = prepare_pages(path)
    if len(pages) > 1:
        raise ValueError(f"{Path(path).name} has {len(pages)} pages; use prepare_pages()")
    return pages[0]


def _page_path(digest: str, index: int) -> Path:
    return Path(tempfile.gettempdir()) / f"prepared_{digest}_p{index}.png"


def _save(img: Image.Image, out: Path) -> None:
    img = img.convert("RGB")
    img.thumbnail((MAX_EDGE, MAX_EDGE))
    tmp = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    img.save(tmp, format="PNG")
    tmp.replace(out)


def page_note(n_pages: int) -> str:
    """Prompt preamble for multi-page documents; empty for the common single image."""
    if n_pages <= 1:
        return ""
    return (f"This document has {n_pages} pages, provided in order as {n_pages} images. "
            "Treat them as ONE document: line items may continue across pages and the "
            "total is usually on the last page. In the transcription, start each page "
            "with a line '--- page N ---'.\n\n")


def build_prompt(doc_type: str = "receipt") -> str:
    schema = json.dumps(DOCUMENT_MODELS[doc_type].model_json_schema(), indent=2)
    return (
        f"Extract the {doc_type} fields into a JSON object matching this schema. "
        "Money values as numbers, dates as YYYY-MM-DD, currency as ISO-4217 "
        "(e.g. IDR, USD). Include every line item. Use null for anything not "
        "present.\n" + NUMBER_RULES + "\n"
        + (INVOICE_RULES + "\n" if doc_type == "invoice" else "")
        + "\nJSON schema:\n" + schema
    )


def extract(path: str | Path, doc_type: str = "receipt") -> BaseModel:
    """VLM-only baseline (Phase 3). No confidence; kept so improvements stay measurable."""
    schema_model = DOCUMENT_MODELS[doc_type]
    # VLM_A_* is what .env.example defines. The skeleton originally read VLM_BASE_URL /
    # VLM_MODEL / API_KEY, which nothing sets, so the Phase 3 baseline (--source vlm-only)
    # failed every document. The old names still win if present.
    provider = Provider(
        base_url=os.environ.get("VLM_BASE_URL") or os.environ["VLM_A_BASE_URL"],
        api_key=os.environ.get("API_KEY") or os.environ.get("VLM_A_API_KEY", ""),
        model=os.environ.get("VLM_MODEL") or os.environ["VLM_A_MODEL"],
    )
    pages = prepare_pages(path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [Provider.text_block(page_note(len(pages)) + build_prompt(doc_type)),
                        *(Provider.image_block(p) for p in pages)],
        },
    ]

    resp = provider.complete(messages, json_mode=True)
    raw = _strip_fences(resp.text)

    try:
        document = schema_model.model_validate_json(raw)
    except Exception:
        # one corrective retry: show the model its bad output and ask for valid JSON
        fix = provider.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(doc_type)},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That was not valid. Return ONLY the JSON object."},
            ],
            json_mode=True,
        )
        document = schema_model.model_validate_json(_strip_fences(fix.text))

    print(
        f"[tokens in/out: {resp.prompt_tokens}/{resp.completion_tokens} | "
        f"latency: {resp.latency_s:.2f}s | model: {resp.model}]",
        file=sys.stderr,
    )
    return document


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m extractor.extract <path-to-receipt>", file=sys.stderr)
        sys.exit(1)
    print(extract(sys.argv[1]).model_dump_json(indent=2))
