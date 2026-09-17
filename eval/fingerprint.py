"""What a run measured, as hashes, so two runs can be proven comparable.

A regression diff is only meaningful if the system is the ONLY thing that changed. Two
things silently break that:

  - Ground truth edits. Verifying labels fixes some of them; a v1 -> v2 diff across a
    label fix measures the fix, not the system. The regression gate refuses it.
  - Unrecorded config changes. If the models, prompts, threshold or fitted confidence
    model changed, the diff should SAY so, so a delta is attributed to the right cause.

Hashes, not copies: short, stable, and enough to prove equality.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def _sha(parts: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:16]


def ground_truth_sha(doc_ids: list[str], find_gt) -> str:
    """Over each document's id AND label bytes, so relabeling any field changes it."""
    return _sha(doc_id.encode() + b"\0" + find_gt(doc_id).read_bytes() for doc_id in sorted(doc_ids))


def documents_sha(doc_ids: list[str], find_doc) -> str:
    return _sha(doc_id.encode() + b"\0" + find_doc(doc_id).read_bytes() for doc_id in sorted(doc_ids))


def prompts_sha() -> str:
    from extractor import extract, pipeline
    return _sha(p.encode() for p in (pipeline.VLM_SYSTEM, pipeline._vlm_prompt(), pipeline.LLM_SYSTEM,
                                     pipeline._llm_prompt(""), extract.SYSTEM_PROMPT,
                                     extract.build_prompt()))


def file_sha(path: Path) -> Optional[str]:
    return _sha([path.read_bytes()]) if path.exists() else None


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return (out.stdout.strip() + ("-dirty" if dirty else "")) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def build(doc_ids: list[str], find_gt, find_doc, *, source: str, threshold: float,
          confidence_model_path: Path, server: Optional[dict] = None) -> dict:
    models = ({k: server.get(k) for k in ("model_a", "model_b", "cascade_model", "mode")}
              if server else
              {"model_a": os.environ.get("VLM_A_MODEL"), "model_b": os.environ.get("LLM_B_MODEL"),
               "cascade_model": os.environ.get("CASCADE_MODEL") if source == "cascade" else None})
    docs_found = [d for d in doc_ids if _exists(find_doc, d)]
    return {
        "ground_truth": ground_truth_sha(doc_ids, find_gt),
        "documents": documents_sha(docs_found, find_doc),
        "prompts": prompts_sha(),
        "models": models,
        "threshold": threshold,
        # Over the API the server's fitted model is not visible to this process.
        "confidence_model": None if server else file_sha(confidence_model_path),
        "code": git_commit(),
    }


def _exists(find, doc_id: str) -> bool:
    try:
        find(doc_id)
        return True
    except FileNotFoundError:
        return False


def differences(a: Optional[dict], b: Optional[dict]) -> dict[str, tuple]:
    """Field -> (before, after) for everything that differs, flattening `models`."""
    if not a or not b:
        return {}
    out: dict[str, tuple] = {}
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if key == "models" and isinstance(va, dict) and isinstance(vb, dict):
            for m in sorted(set(va) | set(vb)):
                if va.get(m) != vb.get(m):
                    out[f"models.{m}"] = (va.get(m), vb.get(m))
        elif va != vb:
            out[key] = (va, vb)
    return out
