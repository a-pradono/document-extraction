"""OCR reader — ABANDONED PATH, kept for reference (PaddleOCR is too slow on CPU).

The VLM's transcription replaced this stage. The groundedness check that used to live
here moved to extractor/grounding.py and became type-aware.

Wraps PaddleOCR: a classical detection + recognition pipeline, not an LLM. It runs
fast on CPU with no network dependency, which is why it lives outside the
OpenAI-compatible `Provider` class in provider.py — PaddleOCR is a local library
call, not a chat-completions HTTP endpoint, and routing it through that
abstraction would just add a pointless server hop on a CPU laptop.

Produces:
  - `text`  : a layout-preserving string (lines in reading order) — this is the
              input to Path B's LLM reasoner (see extractor/pipeline.py, Phase 4).
  - `lines` : per-line (text, confidence, bbox) — kept for the groundedness check
              in confidence scoring: does an extracted field value literally
              appear somewhere in the OCR output?

The PaddleOCR import is deferred into `_load()` so importing this module (or
running the Phase 3 eval harness, which doesn't need OCR yet) never requires
paddleocr/paddlepaddle to be installed.

CLI:  python -m extractor.ocr data/docs/<id>.jpg
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OcrLine:
    text: str
    confidence: float
    bbox: list[tuple[float, float]]  # 4 (x, y) corners of the detected box


@dataclass
class OcrResult:
    text: str
    lines: list[OcrLine]


class OcrReader:
    """Lazy-loaded PaddleOCR wrapper. One instance per process — the model load
    is the slow part (a few seconds on CPU), so don't recreate it per document."""

    def __init__(self, lang: str | None = None):
        self.lang = lang or os.environ.get("OCR_LANG", "en")
        self._engine = None

    def _load(self):
        if self._engine is None:
            from paddleocr import PaddleOCR  # heavy import, deferred on purpose

            # use_angle_cls helps with tilted phone-photo receipts; use_gpu=False
            # is explicit even though it's the CPU default, so intent is obvious.
            self._engine = PaddleOCR(
                use_angle_cls=True, lang=self.lang, use_gpu=False, show_log=False
            )
        return self._engine

    def read(self, image_path: str | Path) -> OcrResult:
        engine = self._load()
        raw = engine.ocr(str(image_path), cls=True)
        page = raw[0] if raw else []  # PaddleOCR returns a list per input image

        lines: list[OcrLine] = []
        for box, (text, conf) in page or []:
            lines.append(OcrLine(text=text, confidence=float(conf), bbox=[tuple(p) for p in box]))

        # Reading order for a single-column receipt: sort by top-y, then left-x.
        # Good enough for most receipts; revisit if you extract multi-column forms.
        lines.sort(key=lambda l: (round(l.bbox[0][1], 1), l.bbox[0][0]))
        return OcrResult(text="\n".join(l.text for l in lines), lines=lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m extractor.ocr <path-to-receipt>", file=sys.stderr)
        sys.exit(1)
    result = OcrReader().read(sys.argv[1])
    print(result.text)
