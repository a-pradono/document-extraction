"""In-process service counters behind GET /metrics.

Plain JSON rather than a Prometheus exporter: one process, no scraper in this
deployment, and no new dependency. Counters reset on restart; the durable record of
quality is the eval scorecard, not this endpoint.

Latency percentiles cover a bounded rolling window, so a long-running server reports
recent behaviour instead of averaging away a regression under its whole history.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from api.schemas import ExtractionOut
from extractor.cost import latency_percentiles

WINDOW = 500


class Telemetry:
    def __init__(self, window: int = WINDOW):
        self._lock = threading.Lock()
        self._started = time.time()
        self._latency: deque[float] = deque(maxlen=window)
        self._wall: deque[float] = deque(maxlen=window)
        self.requests = 0
        self.rejected = 0
        self.docs_ok = 0
        self.docs_failed = 0
        self.fields_scored = 0
        self.fields_abstained = 0
        self.escalated = 0
        self.cascade_docs = 0
        self.model_b_degraded = 0
        self.cost_usd = 0.0
        self.tokens = 0
        self.cache_hits = 0
        self.model_calls = 0

    def request(self) -> None:
        with self._lock:
            self.requests += 1

    def reject(self) -> None:
        with self._lock:
            self.rejected += 1

    def record(self, result: ExtractionOut) -> None:
        with self._lock:
            if result.status != "ok":
                self.docs_failed += 1
                return
            self.docs_ok += 1
            self.fields_scored += len(result.confidence)
            self.fields_abstained += len(result.abstained)
            if result.cascade:
                self.cascade_docs += 1
                self.escalated += int(bool(result.cascade.get("escalated")))
            if any("model_b_failed" in w for w in result.warnings):
                self.model_b_degraded += 1
            if result.usage:
                self.cost_usd += result.usage.cost_usd
                self._latency.append(result.usage.latency_s)
                self._wall.append(result.usage.wall_s)
                for s in result.usage.stages:
                    self.model_calls += 1
                    self.cache_hits += int(s.cached)
                    self.tokens += s.prompt_tokens + s.completion_tokens

    def snapshot(self) -> dict:
        with self._lock:
            docs = self.docs_ok + self.docs_failed
            return {
                "uptime_s": round(time.time() - self._started, 1),
                "requests": self.requests,
                "rejected_requests": self.rejected,
                "documents": {
                    "processed": docs,
                    "succeeded": self.docs_ok,
                    "failed": self.docs_failed,
                    "failure_rate": round(self.docs_failed / docs, 4) if docs else 0.0,
                    "model_b_degraded": self.model_b_degraded,
                },
                "abstention_rate": (round(self.fields_abstained / self.fields_scored, 4)
                                    if self.fields_scored else 0.0),
                "cascade": {
                    "documents": self.cascade_docs,
                    "escalation_rate": (round(self.escalated / self.cascade_docs, 4)
                                        if self.cascade_docs else None),
                },
                "cost": {
                    "basis": "list-price equivalent (free tier: actual spend $0)",
                    "total_usd": round(self.cost_usd, 6),
                    "per_doc_usd": round(self.cost_usd / self.docs_ok, 6) if self.docs_ok else 0.0,
                    "tokens": self.tokens,
                },
                "cache": {
                    "model_calls": self.model_calls,
                    "hits": self.cache_hits,
                    "hit_rate": (round(self.cache_hits / self.model_calls, 4)
                                 if self.model_calls else 0.0),
                },
                "latency_window": {
                    "model": latency_percentiles(list(self._latency)),
                    "wall": latency_percentiles(list(self._wall)),
                },
            }
