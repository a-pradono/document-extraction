"""Synthetic robustness tier — controlled degradation of clean CORD scans.

    python scripts/degrade.py [--source cord] [--per-doc 1] [--seed 20240917]

We own no scans of our own, so this compensates: take clean documents, apply one
controlled degradation each, and re-run extraction against the UNCHANGED ground truth.
Labeling cost is zero and the result is a robustness curve instead of an anecdote.

Outputs land in data/docs/synthetic/ with ground truth in data/ground_truth/synthetic/,
listed in data/sets/synthetic.txt. Both are kept out of the default document sweep on
purpose: augmented images must never appear in the headline scorecard or be presented
as real-world variation. Report them as their own clearly-labeled tier.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from gt_common import DOCS_DIR, GT_DIR, REPO

OUT_DOCS = DOCS_DIR / "synthetic"
OUT_GT = GT_DIR / "synthetic"
SETS_DIR = REPO / "data" / "sets"


def blur(im: Image.Image, rng: random.Random) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(1.2, 2.4)))


def rotate(im: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.choice([-1, 1]) * rng.uniform(3.0, 8.0)
    return im.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(235, 235, 235))


def dim(im: Image.Image, rng: random.Random) -> Image.Image:
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.55, 0.75))
    return ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 0.9))


def thermal_fade(im: Image.Image, rng: random.Random) -> Image.Image:
    """Thermal paper fades from the top and loses contrast — a vertical alpha ramp
    toward white reproduces the characteristic 'top half is gone' failure."""
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.45, 0.6))
    width, height = im.size
    ramp = Image.linear_gradient("L").resize((width, height))
    return Image.composite(im, Image.new("RGB", im.size, (255, 255, 255)), ramp)


VARIANTS = {"blur": blur, "rotate": rotate, "dim": dim, "fade": thermal_fade}
# JPEG recompression is applied on save rather than as a filter, since the artifact is
# produced by the encoder itself.
JPEG_QUALITY = {"jpeg": 22}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cord", help="id prefix to degrade")
    ap.add_argument("--per-doc", type=int, default=1, help="variants per source document")
    ap.add_argument("--seed", type=int, default=20240917)
    args = ap.parse_args()

    sources = sorted(p for p in DOCS_DIR.glob(f"{args.source}_*") if p.is_file())
    if not sources:
        raise SystemExit(f"no documents matching {args.source}_* in {DOCS_DIR}")

    if OUT_DOCS.exists():
        shutil.rmtree(OUT_DOCS)
    if OUT_GT.exists():
        shutil.rmtree(OUT_GT)
    OUT_DOCS.mkdir(parents=True)
    OUT_GT.mkdir(parents=True)

    rng = random.Random(args.seed)
    names = list(VARIANTS) + list(JPEG_QUALITY)
    written: list[str] = []

    for path in sources:
        gt_src = GT_DIR / f"{path.stem}.json"
        if not gt_src.exists():
            continue
        for variant in rng.sample(names, min(args.per_doc, len(names))):
            doc_id = f"syn_{path.stem}_{variant}"
            with Image.open(path) as im:
                im = im.convert("RGB")
                out = VARIANTS[variant](im, rng) if variant in VARIANTS else im
                out.save(OUT_DOCS / f"{doc_id}.jpg", "JPEG",
                         quality=JPEG_QUALITY.get(variant, 88))
            shutil.copyfile(gt_src, OUT_GT / f"{doc_id}.json")
            written.append(doc_id)

    SETS_DIR.mkdir(parents=True, exist_ok=True)
    (SETS_DIR / "synthetic.txt").write_text(
        "# SYNTHETIC — augmented copies of clean documents, not real-world variation.\n"
        "# Reported as its own tier, never inside the headline scorecard.\n"
        + "\n".join(written) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for doc_id in written:
        counts[doc_id.rsplit("_", 1)[1]] = counts.get(doc_id.rsplit("_", 1)[1], 0) + 1
    print(f"wrote {len(written)} synthetic docs to {OUT_DOCS.relative_to(REPO)}")
    print("  " + json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
