"""Cost and latency accounting (Phase 5). Run: pytest eval/tests -q"""

from __future__ import annotations

import pytest

from extractor.cost import StageUsage, latency_percentiles, stage_usage, summarize

USAGE = {"prompt_tokens": 1_000_000, "completion_tokens": 200_000, "latency_s": 2.5,
         "model": "vlm"}


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setenv("VLM_A_PRICE_IN_PER_M", "0.30")
    monkeypatch.setenv("VLM_A_PRICE_OUT_PER_M", "2.50")
    monkeypatch.setenv("LLM_B_PRICE_IN_PER_M", "0.04")
    monkeypatch.setenv("LLM_B_PRICE_OUT_PER_M", "0.10")


class TestStageUsage:
    def test_prices_both_directions(self, priced):
        s = stage_usage("model_a", USAGE)
        assert s.cost_usd == pytest.approx(0.30 + 0.2 * 2.50)
        assert s.priced

    def test_unpriced_stage_costs_zero_and_says_so(self, monkeypatch):
        for var in ("VLM_A_PRICE_IN_PER_M", "VLM_A_PRICE_OUT_PER_M"):
            monkeypatch.delenv(var, raising=False)
        s = stage_usage("model_a", USAGE)
        assert s.cost_usd == 0.0 and not s.priced

    def test_cached_call_keeps_its_cost(self, priced):
        """$/doc is a property of the system, not of whether this run hit the cache."""
        assert stage_usage("model_a", USAGE, cached=True).cost_usd == \
            stage_usage("model_a", USAGE, cached=False).cost_usd

    def test_bad_price_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("VLM_A_PRICE_IN_PER_M", "cheap")
        monkeypatch.setenv("VLM_A_PRICE_OUT_PER_M", "1")
        with pytest.raises(ValueError, match="bad price"):
            stage_usage("model_a", USAGE)

    def test_missing_usage_is_zero_not_a_crash(self, priced):
        s = stage_usage("model_b", None)
        assert s.prompt_tokens == 0 and s.cost_usd == 0.0


class TestLatencyPercentiles:
    def test_empty(self):
        assert latency_percentiles([]) == {}

    @pytest.mark.parametrize("values,p50", [
        ([1, 2, 3, 4], 2),
        ([1, 2, 3, 4, 5], 3),
        # Regression: Python's round() is banker's rounding, which put p50 of six values
        # on the 4th instead of the 3rd. Nearest-rank is ceil(p*n).
        ([1, 2, 3, 4, 5, 6], 3),
    ])
    def test_nearest_rank_p50(self, values, p50):
        assert latency_percentiles(values)["p50_s"] == p50

    def test_p95_is_the_tail_not_the_max_at_larger_n(self):
        values = list(range(1, 41))                  # ceil(0.95 * 40) = 38
        out = latency_percentiles(values)
        assert out["p95_s"] == 38 and out["max_s"] == 40 and out["n"] == 40

    def test_order_does_not_matter(self):
        assert latency_percentiles([5, 1, 3]) == latency_percentiles([1, 3, 5])


class TestSummarize:
    def test_rolls_up_per_stage_and_per_doc(self):
        docs = [
            [StageUsage("model_a", "a", 1000, 200, 3.0, 0.01, priced=True),
             StageUsage("model_b", "b", 500, 100, 1.0, 0.001, cached=True, priced=True)],
            [StageUsage("model_a", "a", 1000, 200, 5.0, 0.01, priced=True)],
        ]
        s = summarize(docs, [4.0, 5.0])
        assert s["n_docs"] == 2
        assert s["total_cost_usd"] == pytest.approx(0.021)
        assert s["cost_per_doc_usd"] == pytest.approx(0.0105)
        assert s["stages"]["model_a"]["calls"] == 2
        assert s["stages"]["model_a"]["mean_latency_s"] == 4.0
        assert s["stages"]["model_b"]["cache_hits"] == 1
        assert s["tokens_per_doc"] == 1500
        assert s["fully_priced"]

    def test_flags_unpriced_calls(self):
        s = summarize([[StageUsage("model_a", "a", 10, 10, 1.0, 0.0, priced=False)]], [1.0])
        assert not s["fully_priced"]

    def test_labels_the_cost_basis(self):
        assert "list-price" in summarize([], [])["cost_basis"]
