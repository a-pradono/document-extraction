"""The single extraction entry point behind both the API and the batch runner."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from api.schemas import ExtractionOut, from_error, from_result
from extractor.cascade import extract_cascade
from extractor.confidence import ConfidenceModel
from extractor.pipeline import PipelineResult, extract_with_confidence

load_dotenv()
log = logging.getLogger("api.service")

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}

Extractor = Callable[..., PipelineResult]


REQUIRED_ENV = ("VLM_A_BASE_URL", "VLM_A_MODEL", "VLM_A_API_KEY")
RECOMMENDED_ENV = ("LLM_B_BASE_URL", "LLM_B_MODEL", "LLM_B_API_KEY")


def config_problems() -> dict[str, list[str]]:
    """Missing configuration is a property of the deployment, not of any document.

    Checked once up front so a batch fails with one clear message instead of recording
    the same KeyError against every receipt, and the API can report itself unready.
    Model B is only recommended: without it the pipeline degrades to single-model with
    the agreement signal neutralized, which is worth a warning, not a refusal.
    """
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    degraded = [k for k in RECOMMENDED_ENV if not os.environ.get(k)]
    if mode() == "cascade" and not os.environ.get("CASCADE_MODEL"):
        missing.append("CASCADE_MODEL")
    if mode() not in MODES:
        missing.append(f"EXTRACTION_MODE (got '{mode()}', expected one of {sorted(MODES)})")
    return {"missing": missing, "degraded": degraded}


MODES = {"pipeline", "cascade"}


def mode() -> str:
    return os.environ.get("EXTRACTION_MODE", "pipeline").strip().lower()


def threshold() -> float:
    return float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))


@lru_cache(maxsize=1)
def confidence_model() -> Optional[ConfidenceModel]:
    """The fitted model if `eval.run --calibrate` has produced one, else None (fallback
    weights). Loaded once per process — it is read on every request."""
    path = Path(os.environ.get("CONFIDENCE_MODEL_PATH", ".cache/confidence_model.json"))
    if path.exists():
        return ConfidenceModel.load(path)
    log.warning("no fitted confidence model at %s; using fallback weights", path)
    return None


def extract_document(
    path: Path,
    doc_id: str,
    *,
    extractor: Optional[Extractor] = None,
    use_cache: bool = True,
) -> ExtractionOut:
    """Never raises for a bad document: the failure is returned as a status="error"
    result, so one unreadable receipt cannot take down a batch or a server worker."""
    # Looked up by name at call time (not bound at import) so EXTRACTION_MODE and test
    # patches of this module's functions both take effect.
    extractor = extractor or {"pipeline": extract_with_confidence,
                              "cascade": extract_cascade}.get(mode(), extract_with_confidence)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return from_error(doc_id, ValueError(f"unsupported file type '{path.suffix}'"))
    t = threshold()
    try:
        result = extractor(path, threshold=t, model=confidence_model(), use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001 — per-document isolation is the requirement
        log.exception("extraction failed for %s", doc_id)
        return from_error(doc_id, exc)
    return from_result(doc_id, result, t)
