"""Error taxonomy (Phase 3).

Classifies each field comparison by the KIND of failure, not just pass/fail. This is
what lets the scorecard say *how* the system fails. Labels operate on the NORMALIZED
comparison (the honest metric); `format_error` is separated from `wrong_value` so a
right-content/wrong-format miss doesn't look like a genuinely wrong extraction.
"""

from __future__ import annotations

CORRECT = "correct"
MISSING = "missing"            # GT has a value, prediction is null -> under-extraction
SPURIOUS = "spurious"          # GT is null, prediction has a value -> hallucination
WRONG_VALUE = "wrong_value"    # both present, genuinely different content
FORMAT_ERROR = "format_error"  # both present, same content, format/parse difference
SWAPPED_FIELD = "swapped_field"  # predicted value belongs to a different field

FAILURE_LABELS = (MISSING, SPURIOUS, WRONG_VALUE, FORMAT_ERROR, SWAPPED_FIELD)
ALL_LABELS = (CORRECT,) + FAILURE_LABELS


def classify(
    gt_present: bool,
    pred_present: bool,
    normalized_match: bool,
    aggressive_match: bool,
) -> str:
    """Base label from comparison facts. Swapped-field is applied later with
    whole-document context (see metrics.compare_document)."""
    if not gt_present and not pred_present:
        return CORRECT  # true negative: correctly left blank
    if normalized_match:
        return CORRECT
    if gt_present and not pred_present:
        return MISSING
    if pred_present and not gt_present:
        return SPURIOUS
    # both present, normalized mismatch:
    if aggressive_match:
        return FORMAT_ERROR
    return WRONG_VALUE


def rollup(labels) -> dict:
    counts = {label: 0 for label in ALL_LABELS}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
