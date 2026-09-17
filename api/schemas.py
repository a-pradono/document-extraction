"""Wire format shared by POST /extract and the batch runner.

One definition, so a document extracted through the API and the same document in a
batch results file are byte-for-byte comparable.
"""

from __future__ import annotations

import json
from typing import Optional, Union

from pydantic import BaseModel, Field

from extractor.pipeline import PipelineResult
from extractor.schema import Invoice, Receipt


class StageUsageOut(BaseModel):
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool
    priced: bool


class UsageOut(BaseModel):
    latency_s: float = Field(description="Sum of model-call latency; a property of the system")
    wall_s: float = Field(description="This request's wall clock; near zero on a cache hit")
    cost_usd: float = Field(description="List-price equivalent; actual spend on free tiers is $0")
    stages: list[StageUsageOut]


class ExtractionOut(BaseModel):
    doc_id: str
    status: str = "ok"
    document_type: str = "receipt"
    document: Optional[Union[Invoice, Receipt]] = None
    confidence: dict[str, float] = Field(default_factory=dict)
    abstained: list[str] = Field(
        default_factory=list,
        description="Fields whose confidence fell below the threshold. Their values are "
                    "withheld (null) in `document`; see `withheld` for what was read. "
                    "Nested fields use dotted names, e.g. `vendor.name`.",
    )
    withheld: dict[str, Optional[str]] = Field(default_factory=dict)
    threshold: Optional[float] = None
    usage: Optional[UsageOut] = None
    cascade: Optional[dict] = Field(
        default=None, description="Present in cascade mode: {escalated, reasons}")
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


def from_result(doc_id: str, result: PipelineResult, threshold: float,
                doc_type: str = "receipt") -> ExtractionOut:
    """Abstention is applied HERE, at the boundary: a consumer reading `document` must
    never see a value the system is not confident in. The raw value is kept under
    `withheld` so a human reviewer can still see what the model read."""
    document = result.document.model_copy(deep=True)
    withheld: dict[str, Optional[str]] = {}
    for field in result.abstained:
        *parents, leaf = field.split(".")               # "vendor.name" -> document.vendor
        owner = document
        for parent in parents:
            owner = getattr(owner, parent, None)
        if owner is None:
            continue
        value = getattr(owner, leaf, None)
        if field == "line_items":
            # The table is withheld whole (an empty list, never a partial table) and kept
            # as JSON so a reviewer, or the eval harness, can still see every row read.
            withheld[field] = json.dumps([i.model_dump(mode="json") for i in value])
            setattr(owner, leaf, [])
            continue
        withheld[field] = None if value is None else str(value)
        setattr(owner, leaf, None)

    return ExtractionOut(
        doc_id=doc_id,
        document_type=doc_type,
        document=document,
        confidence=result.confidence,
        abstained=list(result.abstained),
        withheld=withheld,
        threshold=threshold,
        usage=UsageOut(
            latency_s=result.latency_s,
            wall_s=result.wall_s,
            cost_usd=result.cost_usd,
            stages=[StageUsageOut(**s.to_dict()) for s in result.stages],
        ),
        cascade=result.cascade or None,
        warnings=list(result.errors),
    )


def from_error(doc_id: str, exc: BaseException) -> ExtractionOut:
    return ExtractionOut(doc_id=doc_id, status="error", error=f"{type(exc).__name__}: {exc}")
