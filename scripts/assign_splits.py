"""Assign eval splits -> data/sets/*.txt

    python scripts/assign_splits.py [--seed 20240917]

    calibration  22   (cord 9,  wild 9,  retail 4)   fits confidence weights + threshold
    test         43   (cord 16, wild 16, retail 11)  the reported scorecard
    smoke        12   drawn FROM calibration          fast dev loop
    degraded     ~12  a TAG on documents inside test  per-subset breakdown

Two rules this file exists to enforce:
  1. smoke is drawn from calibration, never from test. Smoke runs dozens of times while
     iterating; if it overlapped test, the published numbers would be quietly overfit.
  2. nothing is fitted on test — the confidence weights and the abstention threshold see
     calibration only.

The seed is fixed and the outputs are committed so runs stay comparable across versions,
which is what the Phase 7 regression diff depends on.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from gt_common import GT_DIR, REPO

SETS_DIR = REPO / "data" / "sets"
DOCS_DIR = REPO / "data" / "docs"

QUOTAS = {
    "calibration": {"cord": 9, "wild": 9, "retail": 4},
    "test": {"cord": 16, "wild": 16, "retail": 11},
}
SMOKE_N = 12
DEGRADED_N = 12


def _ids_by_source() -> dict[str, list[str]]:
    by_source: dict[str, list[str]] = {}
    for path in sorted(GT_DIR.glob("*.json")):
        if path.stem.startswith("_"):
            continue
        by_source.setdefault(path.stem.split("_")[0], []).append(path.stem)
    return by_source


def _write(name: str, ids: list[str], header: str) -> None:
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"# {line}" for line in header.strip().splitlines())
    (SETS_DIR / f"{name}.txt").write_text(
        body + "\n" + "\n".join(sorted(ids)) + "\n", encoding="utf-8"
    )
    print(f"  {name:<12} {len(ids):>3} docs")


def _sharpness(path: Path) -> float:
    """Edge energy per pixel — a cheap blur/fade proxy. Low = soft, faded or out of focus."""
    from PIL import Image, ImageFilter, ImageStat

    with Image.open(path) as im:
        small = im.convert("L")
        small.thumbnail((640, 640))
        stat = ImageStat.Stat(small.filter(ImageFilter.FIND_EDGES))
    return stat.stddev[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20240917)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_source = _ids_by_source()
    if not by_source:
        raise SystemExit(f"no ground truth in {GT_DIR} — run the converters first")

    assigned: dict[str, list[str]] = {"calibration": [], "test": []}
    for source, ids in sorted(by_source.items()):
        pool = sorted(ids)
        rng.shuffle(pool)
        want = {s: QUOTAS[s].get(source, 0) for s in assigned}
        have = len(pool)
        if have < sum(want.values()):
            print(f"  ! {source}: {have} docs, quotas want {sum(want.values())} "
                  f"— splitting proportionally instead")
            scale = have / max(sum(want.values()), 1)
            want = {s: int(n * scale) for s, n in want.items()}
            want["test"] += have - sum(want.values())
        for split, n in want.items():
            assigned[split].extend(pool[:n])
            pool = pool[n:]

    print("splits:")
    _write("calibration", assigned["calibration"],
           "Fits confidence weights + abstention threshold. NEVER reported as a result.")
    _write("test", assigned["test"],
           "The reported scorecard. Nothing is ever fitted on these documents.")

    smoke = rng.sample(assigned["calibration"], min(SMOKE_N, len(assigned["calibration"])))
    _write("smoke", smoke,
           "Fast dev loop. Drawn FROM calibration so heavy iteration cannot overfit test.")

    # A heuristic seed for the degraded tag, not the final word: it ranks test documents
    # by edge energy (blur/fade proxy). Confirm by eye during verification and edit freely.
    #
    # Ranked WITHIN each source, then taken proportionally. A global ranking put 11 of 12
    # tags on CORD, which would make "F1 drops on degraded" indistinguishable from "F1
    # drops on CORD" — the subset has to vary image quality, not source.
    by_source_scores: dict[str, list[tuple[float, str]]] = {}
    for doc_id in assigned["test"]:
        hits = list(DOCS_DIR.glob(f"{doc_id}.*"))
        if not hits:
            continue
        try:
            by_source_scores.setdefault(doc_id.split("_")[0], []).append(
                (_sharpness(hits[0]), doc_id))
        except Exception as exc:                     # unreadable image is itself a signal
            print(f"  ! could not score {doc_id}: {exc}")

    n_test = sum(len(v) for v in by_source_scores.values()) or 1
    degraded = []
    for source, scores in sorted(by_source_scores.items()):
        take = round(DEGRADED_N * len(scores) / n_test)
        degraded.extend(doc_id for _, doc_id in sorted(scores)[:take])
    _write("degraded", degraded,
           "A TAG on documents already inside test, for a per-subset breakdown — not a\n"
           "separate held-out set. Seeded by an edge-energy blur proxy; confirm by eye.")

    overlap = set(assigned["calibration"]) & set(assigned["test"])
    assert not overlap, f"calibration/test overlap: {overlap}"
    assert not set(smoke) - set(assigned["calibration"]), "smoke leaked outside calibration"
    total = len(assigned["calibration"]) + len(assigned["test"])
    print(f"\n{total} documents split with seed {args.seed}. Commit data/sets/*.txt.")


if __name__ == "__main__":
    main()
