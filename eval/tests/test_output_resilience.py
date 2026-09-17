"""Resilience to bad model output. Run: pytest eval/tests -q

A model will eventually return prose around its JSON, an unclosed brace, "N/A" for a
total or "12 Maret" for a date. None of that should cost the rest of the document, and
none of it may be shipped as a value.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from PIL import Image

from extractor import output, pipeline
from extractor.output import ModelOutputError, parse_json_object, parse_or_repair, salvage_receipt
from extractor.provider import LLMResponse, TruncatedResponse


def _resp(text: str, completion_tokens: int = 200) -> LLMResponse:
    return LLMResponse(text, 1000, completion_tokens, 1.0, "m", "stop")


class RepairProvider:
    model = "m"

    def __init__(self, reply: str | Exception):
        self.reply, self.calls = reply, []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(self.reply, 300, 150, 0.5, "m", "stop")


class TestParse:
    @pytest.mark.parametrize("text", [
        '{"total": 1}',
        '```json\n{"total": 1}\n```',
        'Here is the result:\n{"total": 1}\nHope that helps!',
    ])
    def test_recoverable_without_a_call(self, text):
        assert parse_json_object(text) == {"total": 1}

    def test_top_level_array_is_not_an_object(self):
        with pytest.raises(ValueError):
            parse_json_object("[1, 2]")


class TestRepair:
    def test_valid_output_makes_no_repair_call(self):
        provider = RepairProvider("unused")
        assert parse_or_repair(_resp('{"total": 1}'), provider, {}) == {"total": 1}
        assert provider.calls == []

    def test_broken_json_gets_one_text_only_repair(self):
        provider = RepairProvider('{"total": 53280}')
        usage = {"prompt_tokens": 1000, "completion_tokens": 200, "latency_s": 1.0}
        raw = parse_or_repair(_resp('{"total": 53280,,'), provider, usage)
        assert raw == {"total": 53280}
        messages, _ = provider.calls[0]
        assert all(isinstance(m["content"], str) for m in messages)   # no images re-sent
        assert usage["repaired"] is True
        assert (usage["prompt_tokens"], usage["completion_tokens"]) == (1300, 350)  # billed
        assert usage["latency_s"] == pytest.approx(1.5)

    def test_unrepairable_output_raises_a_clear_error(self):
        with pytest.raises(ModelOutputError, match="even after one repair"):
            parse_or_repair(_resp("total is fifty"), RepairProvider("still not json"), {})

    def test_failed_repair_call_raises_with_context(self):
        with pytest.raises(ModelOutputError, match="repair call failed"):
            parse_or_repair(_resp("nope"), RepairProvider(TimeoutError("slow")), {})

    def test_truncation_is_never_repaired(self, tmp_path, monkeypatch):
        """Repairing a cut-off response would invent its missing half."""
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")

        class Truncating:
            model = "m"
            calls = 0

            def complete(self, *_, **__):
                Truncating.calls += 1
                raise TruncatedResponse("hit max_tokens")

        page = tmp_path / "p.png"
        Image.new("RGB", (10, 10)).save(page)
        with pytest.raises(TruncatedResponse):
            pipeline.run_vlm([page], Truncating())
        assert Truncating.calls == 1


class TestSalvage:
    def test_valid_receipt_passes_through_untouched(self):
        s = salvage_receipt({"total": "53280", "date": "2024-03-12"})
        assert s.document.total == Decimal("53280") and s.invalid == []

    def test_one_bad_value_does_not_cost_the_other_fields(self):
        s = salvage_receipt({"merchant_name": "Kopi", "total": "53280", "date": "12 Maret 2024"})
        assert s.document.merchant_name == "Kopi" and s.document.total == Decimal("53280")
        assert s.document.date is None
        assert s.invalid == ["date"] and "12 Maret 2024" in s.notes[0]

    def test_bad_row_keeps_the_good_rows_but_marks_the_table_invalid(self):
        rows = [{"description": "Kopi", "line_total": "18000"},
                {"description": "Roti", "line_total": "N/A"}]
        s = salvage_receipt({"total": "18000", "line_items": rows})
        assert [i.description for i in s.document.line_items] == ["Kopi"]
        assert "line_items" in s.invalid

    @pytest.mark.parametrize("fields", ["a string", [1, 2], 42])
    def test_non_object_fields_mark_everything_invalid(self, fields):
        s = salvage_receipt(fields)
        assert s.invalid == ["*"] and s.document.total is None

    def test_none_is_just_empty(self):
        assert salvage_receipt(None).invalid == []


class TestPipelineIntegration:
    @pytest.fixture
    def env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
        for k, v in {"VLM_A_BASE_URL": "http://127.0.0.1:1/v1", "VLM_A_MODEL": "vlm",
                     "LLM_B_BASE_URL": "http://127.0.0.1:1/v1", "LLM_B_MODEL": "llm"}.items():
            monkeypatch.setenv(k, v)
        page = tmp_path / "r.png"
        Image.new("RGB", (20, 30), "white").save(page)
        return page

    def _vlm(self, fields):
        text = "KOPI\nTOTAL 53.280\nTgl 12/03/2024"
        return {"transcription": text, "fields": fields}, {"latency_s": 1.0}, False

    def test_invalid_field_is_abstained_not_shipped_or_dropped_silently(self, env, monkeypatch):
        monkeypatch.setattr(pipeline, "run_vlm",
                            lambda *a, **k: self._vlm({"total": "53280", "date": "12 Maret"}))
        monkeypatch.setattr(pipeline, "run_llm",
                            lambda *a, **k: ({"total": "53280"}, {"latency_s": 0.1}, False))
        r = pipeline.extract_with_confidence(env, threshold=0.5)
        assert r.document.total == Decimal("53280")
        assert r.confidence["date"] == 0.0 and "date" in r.abstained
        assert any("invalid date" in e for e in r.errors)

    def test_fields_at_top_level_are_not_read_as_an_empty_receipt(self, env, monkeypatch):
        """A model that returns {"total": ...} instead of {"fields": {"total": ...}} must
        not become an all-null receipt that looks legitimately empty."""
        monkeypatch.setattr(pipeline, "run_vlm", lambda *a, **k: (
            {"transcription": "TOTAL 53.280", "total": 53280}, {"latency_s": 1.0}, False))
        monkeypatch.setattr(pipeline, "run_llm",
                            lambda *a, **k: ({"total": 53280}, {"latency_s": 0.1}, False))
        r = pipeline.extract_with_confidence(env)
        assert r.document.total == Decimal("53280")
        assert r.document_b.total == Decimal("53280")

    def test_repair_is_surfaced_as_a_warning(self, env, monkeypatch):
        monkeypatch.setattr(pipeline, "run_vlm", lambda *a, **k: (
            {"transcription": "TOTAL 53.280", "fields": {"total": 53280}},
            {"latency_s": 1.0, "repaired": True}, False))
        monkeypatch.setattr(pipeline, "run_llm",
                            lambda *a, **k: ({"total": 53280}, {"latency_s": 0.1}, False))
        r = pipeline.extract_with_confidence(env)
        assert any(e.startswith("model_a_output_repaired") for e in r.errors)

    def test_garbled_model_b_lowers_agreement_without_forcing_abstention(self, env, monkeypatch):
        monkeypatch.setattr(pipeline, "run_vlm", lambda *a, **k: self._vlm({"total": "53280"}))
        monkeypatch.setattr(pipeline, "run_llm",
                            lambda *a, **k: ({"total": "fifty"}, {"latency_s": 0.1}, False))
        r = pipeline.extract_with_confidence(env, threshold=0.0)
        assert r.signals["total"].agreement == 0.0
        assert r.confidence["total"] > 0.0 and "total" not in r.abstained
