"""Cache correctness and concurrency safety in the pipeline. Run: pytest eval/tests -q

A wrong cache hit is the worst kind of bug for an eval harness: the run succeeds, the
scorecard looks healthy, and it measured a different model than the one it names.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from extractor import pipeline
from extractor.extract import prepare_image


class FakeProvider:
    def __init__(self, model: str, text: str = '{"fields": {"total": "1.00"}}'):
        self.model, self.text, self.calls = model, text, 0

    def complete(self, messages, **_):
        self.calls += 1

        class R:
            pass

        r = R()
        r.text, r.prompt_tokens, r.completion_tokens = self.text, 100, 20
        r.latency_s, r.model = 0.5, self.model
        return r


def _image(tmp_path, name="r.png", color=(255, 255, 255)):
    path = tmp_path / name
    Image.new("RGB", (40, 60), color).save(path)
    return path


class TestCacheKeys:
    def test_second_call_is_a_cache_hit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        img, p = _image(tmp_path), FakeProvider("flash")
        _, _, cached_1 = pipeline.run_vlm(img, p)
        _, usage, cached_2 = pipeline.run_vlm(img, p)
        assert (cached_1, cached_2) == (False, True) and p.calls == 1
        assert usage["prompt_tokens"] == 100          # cached usage survives for $/doc

    def test_a_different_model_does_not_reuse_the_cache(self, tmp_path, monkeypatch):
        """Regression: the VLM cache was keyed on the image alone, so the Phase 6
        cascade model would have been served Flash's answers and 'measured' nothing."""
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        img = _image(tmp_path)
        pipeline.run_vlm(img, FakeProvider("gemini-3.6-flash"))
        lite = FakeProvider("gemini-3.5-flash-lite")
        _, _, cached = pipeline.run_vlm(img, lite)
        assert not cached and lite.calls == 1

    def test_an_edited_prompt_does_not_reuse_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        img = _image(tmp_path)
        pipeline.run_vlm(img, FakeProvider("flash"))
        monkeypatch.setattr(pipeline, "VLM_SYSTEM", pipeline.VLM_SYSTEM + " Be terse.")
        again = FakeProvider("flash")
        _, _, cached = pipeline.run_vlm(img, again)
        assert not cached and again.calls == 1

    def test_llm_cache_is_keyed_on_model_too(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        pipeline.run_llm("TOTAL 1.00", FakeProvider("qwen"))
        other = FakeProvider("llama")
        _, _, cached = pipeline.run_llm("TOTAL 1.00", other)
        assert not cached

    def test_use_cache_false_always_calls(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        img, p = _image(tmp_path), FakeProvider("flash")
        pipeline.run_vlm(img, p, use_cache=False)
        pipeline.run_vlm(img, p, use_cache=False)
        assert p.calls == 2 and not (tmp_path / "cache").exists()


class TestPrepareImage:
    def test_same_filename_different_content_do_not_collide(self, tmp_path):
        """Regression: prepared images were named after the file stem, so two
        concurrent uploads called receipt.jpg overwrote each other's image."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        red = _image(tmp_path / "a", "receipt.png", (255, 0, 0))
        blue = _image(tmp_path / "b", "receipt.png", (0, 0, 255))
        out_red, out_blue = prepare_image(red), prepare_image(blue)
        assert out_red != out_blue
        assert Image.open(out_red).getpixel((0, 0)) == (255, 0, 0)
        assert Image.open(out_blue).getpixel((0, 0)) == (0, 0, 255)

    def test_same_content_is_deterministic(self, tmp_path):
        a = _image(tmp_path, "x.png", (1, 2, 3))
        b = _image(tmp_path, "y.png", (1, 2, 3))
        assert prepare_image(a) == prepare_image(b)

    def test_concurrent_preparation_is_safe(self, tmp_path):
        paths = [_image(tmp_path, f"r{i}.png", (i, i, i)) for i in range(12)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            outs = list(pool.map(prepare_image, paths))
        assert len(set(outs)) == 12
        for i, out in enumerate(outs):
            assert Image.open(out).getpixel((0, 0)) == (i, i, i)


class TestPromptsCarryTheMeasuredRules:
    """The first real test run produced 13 hallucinations, 12 of which were an INFERRED
    currency on receipts that print no symbol. 'Never guess' was not enough, because a
    currency is deducible from the language and the amount format; the rule has to name
    that inference and forbid it. Every prompt that asks for fields must carry it."""

    def test_every_prompt_forbids_inferring_a_currency(self):
        from extractor import extract, pipeline
        prompts = {
            "vlm receipt": pipeline._vlm_prompt("receipt"),
            "vlm invoice": pipeline._vlm_prompt("invoice"),
            "llm receipt": pipeline._llm_prompt("transcription", "receipt"),
            "llm invoice": pipeline._llm_prompt("transcription", "invoice"),
            "vlm-only baseline": extract.build_prompt("receipt"),
        }
        for name, text in prompts.items():
            assert "Do NOT infer it from the language" in text, f"{name} lost the currency rule"
            assert "null" in text

    def test_the_rule_names_the_failure_it_prevents(self):
        from extractor.extract import CURRENCY_RULE
        assert "hallucination" in CURRENCY_RULE and "Rp" in CURRENCY_RULE


class TestTokenBudget:
    """inv_0010 (225 line items over 5 pages) hit the ceiling and failed as
    TruncatedResponse - correctly loud, but it scored as an empty extraction."""

    def test_a_five_page_document_gets_room_for_a_long_table(self, monkeypatch):
        from extractor import pipeline
        monkeypatch.setenv("VLM_MAX_TOKENS", "16384")
        monkeypatch.setenv("VLM_MAX_TOKENS_PER_EXTRA_PAGE", "8192")
        assert pipeline._vlm_max_tokens(5) == 16384 + 4 * 8192      # 49,152
        assert pipeline._vlm_max_tokens(5) > 24576                  # what inv_0010 needed

    def test_the_budget_still_scales_with_pages(self, monkeypatch):
        from extractor import pipeline
        monkeypatch.setenv("VLM_MAX_TOKENS", "16384")
        monkeypatch.setenv("VLM_MAX_TOKENS_PER_EXTRA_PAGE", "8192")
        assert pipeline._vlm_max_tokens(1) < pipeline._vlm_max_tokens(3)


class TestBaselineIsAFairComparison:
    """The VLM-only baseline exists to make "two models beat one" measurable, which needs
    the two paths to differ ONLY in architecture. Left on the provider default of 1500
    tokens, 18 of 39 documents truncated in the baseline while the pipeline had 16384 +
    8192 per extra page - so the first comparison measured the token budget."""

    def test_the_baseline_gets_the_same_budget_as_the_pipeline(self, monkeypatch):
        from extractor import extract, pipeline
        monkeypatch.setenv("VLM_MAX_TOKENS", "16384")
        monkeypatch.setenv("VLM_MAX_TOKENS_PER_EXTRA_PAGE", "8192")
        for pages in (1, 3, 5):
            assert extract._baseline_max_tokens(pages) == pipeline._vlm_max_tokens(pages)

    def test_the_baseline_passes_a_budget_at_all(self):
        import inspect

        from extractor import extract
        assert "max_tokens=budget" in inspect.getsource(extract.extract)
