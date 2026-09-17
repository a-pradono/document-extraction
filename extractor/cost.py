"""Per-call cost accounting (Phase 5).

Prices come from .env, never from code, so swapping a model never leaves a stale price
behind in a source file:

    VLM_A_PRICE_IN_PER_M=0.30      # USD per 1M prompt tokens
    VLM_A_PRICE_OUT_PER_M=2.50     # USD per 1M completion tokens

We run on free tiers, so the money actually spent is $0. The reported figure is the
LIST-PRICE EQUIVALENT of the tokens consumed — the number that makes the Phase 6
optimization measurable and tells a reader what this would cost on a paid tier. The
scorecard labels it that way; it is never presented as an invoice.

Cached calls still carry their original token usage and cost, because cost-per-document
is a property of the system, not of whether this particular run happened to hit the
cache. `cached` is recorded alongside so the two can be told apart.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

STAGES = ("model_a", "model_b")
_ENV_PREFIX = {"model_a": "VLM_A", "model_b": "LLM_B", "cascade": "CASCADE"}


@dataclass
class StageUsage:
    stage: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    priced: bool = False       # False when .env has no price for this stage

    def to_dict(self) -> dict:
        return asdict(self)


def _price(stage: str, direction: str) -> float | None:
    raw = os.environ.get(f"{_ENV_PREFIX.get(stage, stage.upper())}_PRICE_{direction}_PER_M")
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"bad price in .env for {stage} {direction}: {raw!r}") from exc


def stage_usage(stage: str, usage: dict | None, cached: bool = False) -> StageUsage:
    """Build a StageUsage from the usage dict the pipeline stores for each call."""
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    price_in, price_out = _price(stage, "IN"), _price(stage, "OUT")
    priced = price_in is not None and price_out is not None
    cost = (prompt * price_in + completion * price_out) / 1_000_000 if priced else 0.0
    return StageUsage(
        stage=stage,
        model=str(usage.get("model", "")),
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_s=float(usage.get("latency_s", 0.0) or 0.0),
        cost_usd=round(cost, 8),
        cached=cached,
        priced=priced,
    )


def summarize(per_doc: list[list[StageUsage]], latencies: list[float]) -> dict:
    """Roll document-level stage usage into the cost/latency block of a report.

    `latencies` is end-to-end seconds per successfully processed document.
    """
    n = len(per_doc)
    stages: dict[str, dict] = {}
    for doc in per_doc:
        for s in doc:
            agg = stages.setdefault(s.stage, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "cost_usd": 0.0, "latency_s": 0.0, "cache_hits": 0, "unpriced_calls": 0,
            })
            agg["calls"] += 1
            agg["prompt_tokens"] += s.prompt_tokens
            agg["completion_tokens"] += s.completion_tokens
            agg["cost_usd"] += s.cost_usd
            agg["latency_s"] += s.latency_s
            agg["cache_hits"] += int(s.cached)
            agg["unpriced_calls"] += int(not s.priced and (s.prompt_tokens or s.completion_tokens))
    for agg in stages.values():
        agg["cost_usd"] = round(agg["cost_usd"], 6)
        agg["mean_latency_s"] = round(agg.pop("latency_s") / agg["calls"], 3) if agg["calls"] else 0.0

    total_cost = sum(a["cost_usd"] for a in stages.values())
    total_tokens = sum(a["prompt_tokens"] + a["completion_tokens"] for a in stages.values())
    return {
        "n_docs": n,
        "cost_basis": "list-price equivalent (free tier: actual spend $0)",
        "total_cost_usd": round(total_cost, 6),
        "cost_per_doc_usd": round(total_cost / n, 6) if n else 0.0,
        "total_tokens": total_tokens,
        "tokens_per_doc": round(total_tokens / n, 1) if n else 0.0,
        "latency": latency_percentiles(latencies),
        "stages": stages,
        "fully_priced": all(a["unpriced_calls"] == 0 for a in stages.values()),
    }


def latency_percentiles(values: list[float]) -> dict:
    """Nearest-rank percentiles. At n≈40 p95 is roughly the 2nd-slowest document, so it
    is reported with n alongside it rather than as a precise tail estimate."""
    if not values:
        return {}
    ordered = sorted(values)

    def rank(p: float) -> float:
        idx = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
        return round(ordered[idx], 3)

    return {
        "n": len(ordered),
        "p50_s": rank(0.50),
        "p95_s": rank(0.95),
        "max_s": round(ordered[-1], 3),
        "mean_s": round(sum(ordered) / len(ordered), 3),
    }
