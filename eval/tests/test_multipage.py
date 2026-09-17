"""Multi-page documents and page preparation. Run: pytest eval/tests -q

PDFs are generated on the fly with PIL, so these run anywhere with no fixtures on disk
and no poppler.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from extractor import extract, pipeline
from extractor.provider import LLMResponse, Provider, TruncatedResponse


def _pdf(tmp_path, n_pages: int, name: str = "doc.pdf"):
    pages = []
    for i in range(n_pages):
        im = Image.new("RGB", (400, 600), (255 - i * 40, 255, 255))
        ImageDraw.Draw(im).text((20, 20), f"PAGE {i + 1}", fill="black")
        pages.append(im)
    path = tmp_path / name
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=72)
    return path


class TestPreparePages:
    def test_every_pdf_page_is_rendered_in_order(self, tmp_path):
        """Regression: only page 1 was read, and receipt totals live on the last page."""
        pages = extract.prepare_pages(_pdf(tmp_path, 3))
        assert len(pages) == 3
        reds = [Image.open(p).getpixel((5, 5))[0] for p in pages]
        assert reds == sorted(reds, reverse=True)          # page 1 is the lightest

    def test_pdf_rendering_needs_no_poppler(self, tmp_path, monkeypatch):
        """Regression: pdf2image shelled out to poppler, so every PDF failed without it.
        With no poppler on PATH, rendering must still work."""
        monkeypatch.setenv("PATH", "")
        assert len(extract.prepare_pages(_pdf(tmp_path, 1, "nopoppler.pdf"))) == 1

    def test_pages_are_downscaled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extract, "MAX_EDGE", 300)
        for page in extract.prepare_pages(_pdf(tmp_path, 2, "small.pdf")):
            assert max(Image.open(page).size) <= 300

    def test_over_the_page_limit_is_refused_not_truncated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extract, "MAX_PAGES", 2)
        with pytest.raises(extract.TooManyPages, match="refusing to extract a truncated"):
            extract.prepare_pages(_pdf(tmp_path, 3))

    def test_single_image_is_one_page(self, tmp_path):
        path = tmp_path / "r.png"
        Image.new("RGB", (50, 80), "white").save(path)
        assert len(extract.prepare_pages(path)) == 1

    def test_exif_rotation_is_applied(self, tmp_path):
        """Phone photos store rotation in EXIF; without transposing, the model gets a
        sideways receipt."""
        path = tmp_path / "rotated.jpg"
        img = Image.new("RGB", (200, 100), "white")        # landscape pixels
        exif = img.getexif()
        exif[0x0112] = 6                                    # "rotate 90 CW to display"
        img.save(path, exif=exif)
        page = extract.prepare_pages(path)[0]
        assert Image.open(page).size == (100, 200)          # displayed as portrait

    def test_prepare_image_refuses_multipage_instead_of_returning_page_one(self, tmp_path):
        with pytest.raises(ValueError, match="has 3 pages"):
            extract.prepare_image(_pdf(tmp_path, 3))


class RecordingProvider:
    model = "vlm"

    def __init__(self):
        self.messages, self.max_tokens = None, None

    def complete(self, messages, *, max_tokens=0, **_):
        self.messages, self.max_tokens = messages, max_tokens
        return LLMResponse('{"transcription": "", "fields": {}}', 10, 10, 0.1, "vlm", "stop")


class TestVlmCall:
    def test_all_pages_go_in_one_call_with_a_page_note(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        provider = RecordingProvider()
        pipeline.run_vlm(extract.prepare_pages(_pdf(tmp_path, 3)), provider)
        content = provider.messages[1]["content"]
        assert sum(1 for b in content if b["type"] == "image_url") == 3
        assert "This document has 3 pages" in content[0]["text"]

    def test_single_page_prompt_has_no_page_note(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        provider = RecordingProvider()
        pipeline.run_vlm(extract.prepare_pages(_pdf(tmp_path, 1)), provider)
        assert "pages" not in provider.messages[1]["content"][0]["text"].split("Schema")[0]

    def test_token_budget_grows_with_pages(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.delenv("VLM_MAX_TOKENS", raising=False)
        monkeypatch.delenv("VLM_MAX_TOKENS_PER_EXTRA_PAGE", raising=False)
        one, three = RecordingProvider(), RecordingProvider()
        pipeline.run_vlm(extract.prepare_pages(_pdf(tmp_path, 1, "a.pdf")), one)
        pipeline.run_vlm(extract.prepare_pages(_pdf(tmp_path, 3, "b.pdf")), three)
        assert one.max_tokens == 8192 and three.max_tokens == 8192 + 2 * 4096

    def test_cache_key_covers_every_page(self, tmp_path, monkeypatch):
        """Two PDFs sharing page 1 but differing on page 2 must not share a cache entry."""
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        a = extract.prepare_pages(_pdf(tmp_path, 2, "a.pdf"))
        other = tmp_path / "different_p2.png"
        Image.new("RGB", (400, 600), "black").save(other)
        b = [a[0], extract.prepare_pages(other)[0]]
        pipeline.run_vlm(a, RecordingProvider())
        _, _, cached = pipeline.run_vlm(b, RecordingProvider())
        assert not cached


class TestTruncation:
    def test_length_finish_raises_instead_of_a_json_error(self, monkeypatch):
        class Choice:
            finish_reason = "length"

            class message:
                content = '{"transcription": "TOTAL'

        class Resp:
            choices = [Choice()]

            class usage:
                prompt_tokens, completion_tokens = 1000, 3000

        provider = Provider(base_url="http://127.0.0.1:1/v1", api_key="x", model="flash")
        monkeypatch.setattr(provider.client.chat.completions, "create", lambda **_: Resp())
        with pytest.raises(TruncatedResponse, match="hit max_tokens=3000"):
            provider.complete([{"role": "user", "content": "hi"}], max_tokens=3000)


class TestEmptyResponse:
    def test_no_choices_raises_a_clear_error(self, monkeypatch):
        """Seen live on an OpenRouter free model: HTTP 200 with an upstream error and
        choices=None, which used to surface as TypeError: 'NoneType' is not subscriptable."""
        from extractor.provider import EmptyResponse

        class Resp:
            choices = None
            usage = None
            error = {"message": "Provider returned error", "code": 429}

        provider = Provider(base_url="http://127.0.0.1:1/v1", api_key="x", model="free-model")
        monkeypatch.setattr(provider.client.chat.completions, "create", lambda **_: Resp())
        with pytest.raises(EmptyResponse, match="returned no choices"):
            provider.complete([{"role": "user", "content": "hi"}])


class TestBaselineEnvNames:
    def test_vlm_only_baseline_reads_the_documented_env_names(self, tmp_path, monkeypatch):
        """Regression: extract() read VLM_BASE_URL/VLM_MODEL, which .env.example never
        defines, so --source vlm-only failed every document."""
        for old in ("VLM_BASE_URL", "VLM_MODEL", "API_KEY"):
            monkeypatch.delenv(old, raising=False)
        monkeypatch.setenv("VLM_A_BASE_URL", "http://127.0.0.1:1/v1")
        monkeypatch.setenv("VLM_A_MODEL", "flash")
        seen = {}

        def fake_complete(self, messages, **_):
            seen["model"] = self.model
            return LLMResponse('{"total": "1.00"}', 1, 1, 0.1, self.model, "stop")

        monkeypatch.setattr(Provider, "complete", fake_complete)
        path = tmp_path / "r.png"
        Image.new("RGB", (40, 60), "white").save(path)
        assert str(extract.extract(path).total) == "1.00" and seen["model"] == "flash"
