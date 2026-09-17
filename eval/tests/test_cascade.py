"""Confidence cascade and the before/after table (Phase 6). Run: pytest eval/tests -q"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from PIL import Image

from eval import compare as cmp
from eval import metrics
from eval.run import cascade_report
from extractor import cascade
from extractor.confidence import FieldSignals
from extractor.cost import StageUsage
from extractor.pipeline import PipelineResult
from extractor.schema import LineItem, Receipt

CLEAN_TEXT = "KOPI KENANGAN\nEs Kopi Susu 36.000\nSUBTOTAL 48.000\nPB1 5.280\nTOTAL 53.280"
CLEAN_FIELDS = {"merchant_name": "Kopi Kenangan", "subtotal": 48000, "tax": 5280, "total": 53280}


def _signals(**overrides) -> dict[str, FieldSignals]:
    base = {f: FieldSignals(agreement=0.5, grounded=1.0, validated=1.0, self_report=0.9)
            for f in metrics.SCALAR_FIELDS}
    base.update(overrides)
    return base


class TestEscalationReasons:
    def test_clean_read_is_accepted(self):
        r = Receipt(total="53280", subtotal="48000", tax="5280")
        assert cascade.escalation_reasons(r, _signals(), {"total": 0.9}, 0.5) == []

    def test_failed_arithmetic_escalates(self):
        r = Receipt(total="99999", subtotal="48000", tax="5280")
        sig = _signals(total=FieldSignals(0.5, 1.0, 0.0, 0.9))
        assert "arithmetic_failed" in cascade.escalation_reasons(r, sig, {}, 0.5)

    def test_ungrounded_value_escalates(self):
        r = Receipt(merchant_name="Starbucks", total="1")
        sig = _signals(merchant_name=FieldSignals(0.5, 0.0, 0.5, 0.9))
        assert "ungrounded:merchant_name" in cascade.escalation_reasons(r, sig, {}, 0.5)

    def test_low_confidence_on_a_present_field_escalates(self):
        r = Receipt(total="53280")
        assert "low_confidence:total" in cascade.escalation_reasons(
            r, _signals(), {"total": 0.2}, 0.5)

    def test_low_confidence_on_an_absent_field_does_not(self):
        """Otherwise nearly every receipt escalates for lacking a payment_method."""
        r = Receipt(total="53280")
        assert cascade.escalation_reasons(r, _signals(), {"payment_method": 0.1}, 0.5) == []

    def test_empty_extraction_escalates(self):
        assert "empty_extraction" in cascade.escalation_reasons(Receipt(), _signals(), {}, 0.5)

    def test_line_items_alone_are_not_empty(self):
        r = Receipt(line_items=[LineItem(description="Kopi", line_total="1")])
        assert "empty_extraction" not in cascade.escalation_reasons(r, _signals(), {}, 0.5)


# --- the cascade control flow, with model calls faked -------------------------
@pytest.fixture
def image(tmp_path):
    path = tmp_path / "r.png"
    Image.new("RGB", (20, 30), "white").save(path)
    return path


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_A_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("CASCADE_MODEL", "lite")
    monkeypatch.delenv("CASCADE_ESCALATE_BELOW", raising=False)


def _cheap(fields, text=CLEAN_TEXT):
    usage = {"prompt_tokens": 1000, "completion_tokens": 300, "latency_s": 1.0, "model": "lite"}
    return {"transcription": text, "fields": fields, "_confidence": {}}, usage, False


def _full_result():
    return PipelineResult(
        receipt=Receipt(total="53280"), receipt_b=None, transcription=CLEAN_TEXT,
        latency_s=5.0, cost_usd=0.002,
        stages=[StageUsage("model_a", "flash", 1800, 400, 4.0, 0.0015, priced=True),
                StageUsage("model_b", "qwen", 900, 150, 1.0, 0.0005, priced=True)],
    )


class TestExtractCascade:
    def test_clean_cheap_read_never_calls_the_full_pipeline(self, image, env, monkeypatch):
        monkeypatch.setattr(cascade, "run_vlm", lambda *a, **k: _cheap(CLEAN_FIELDS))
        monkeypatch.setattr(cascade, "extract_with_confidence",
                            lambda *a, **k: pytest.fail("escalated a clean receipt"))
        result = cascade.extract_cascade(image, escalate_below=0.0)
        assert result.cascade == {"escalated": False, "reasons": []}
        assert [s.stage for s in result.stages] == ["cascade"]

    def test_escalated_doc_pays_for_both_paths(self, image, env, monkeypatch):
        """The cheap call is sunk cost. Dropping it would flatter the cascade exactly
        in proportion to how often it escalates."""
        monkeypatch.setenv("CASCADE_PRICE_IN_PER_M", "0.10")
        monkeypatch.setenv("CASCADE_PRICE_OUT_PER_M", "0.40")
        bad = dict(CLEAN_FIELDS, total=99999)
        monkeypatch.setattr(cascade, "run_vlm", lambda *a, **k: _cheap(bad))
        monkeypatch.setattr(cascade, "extract_with_confidence", lambda *a, **k: _full_result())
        result = cascade.extract_cascade(image, escalate_below=0.0)
        assert result.cascade["escalated"]
        assert "arithmetic_failed" in result.cascade["reasons"]
        assert [s.stage for s in result.stages] == ["cascade", "model_a", "model_b"]
        cheap_cost = (1000 * 0.10 + 300 * 0.40) / 1_000_000
        assert result.cost_usd == pytest.approx(0.002 + cheap_cost)
        assert result.latency_s == pytest.approx(6.0)

    def test_cheap_path_exception_escalates_instead_of_failing(self, image, env, monkeypatch):
        def boom(*_, **__):
            raise TimeoutError("lite is down")

        monkeypatch.setattr(cascade, "run_vlm", boom)
        monkeypatch.setattr(cascade, "extract_with_confidence", lambda *a, **k: _full_result())
        result = cascade.extract_cascade(image)
        assert result.cascade["escalated"]
        assert result.cascade["reasons"][0].startswith("cheap_path_failed: TimeoutError")
        assert [s.stage for s in result.stages] == ["model_a", "model_b"]  # no cheap cost to add

    def test_invalid_cheap_value_escalates_and_the_cheap_call_stays_billed(
            self, image, env, monkeypatch):
        """Regression: a cheap call that succeeded but returned a schema-invalid value
        raised inside the try, which reset the cheap stage, so the call vanished from the
        escalated document's bill."""
        fields = dict(CLEAN_FIELDS, date="not a date")
        monkeypatch.setattr(cascade, "run_vlm", lambda *a, **k: _cheap(fields))
        monkeypatch.setattr(cascade, "extract_with_confidence", lambda *a, **k: _full_result())
        result = cascade.extract_cascade(image, escalate_below=0.0)
        assert "invalid_output:date" in result.cascade["reasons"]
        assert [s.stage for s in result.stages] == ["cascade", "model_a", "model_b"]

    def test_empty_escalate_below_env_falls_back_to_threshold(self, image, env, monkeypatch):
        """Regression: .env.example ships CASCADE_ESCALATE_BELOW= (empty), and a .get()
        default doesn't apply to an empty string, so float("") failed every document."""
        monkeypatch.setenv("CASCADE_ESCALATE_BELOW", "")
        monkeypatch.setattr(cascade, "run_vlm", lambda *a, **k: _cheap(CLEAN_FIELDS))
        monkeypatch.setattr(cascade, "extract_with_confidence",
                            lambda *a, **k: pytest.fail("escalated a clean receipt"))
        result = cascade.extract_cascade(image, threshold=0.0)
        assert result.cascade == {"escalated": False, "reasons": []}

    def test_cascade_uses_its_own_model_not_model_as(self, image, env, monkeypatch):
        seen = []

        def spy(img, provider, use_cache=True):
            seen.append(provider.model)
            return _cheap(CLEAN_FIELDS)

        monkeypatch.setenv("VLM_A_MODEL", "flash")
        monkeypatch.setattr(cascade, "run_vlm", spy)
        cascade.extract_cascade(image, escalate_below=0.0)
        assert seen == ["lite"]


# --- scorecard cascade block ---------------------------------------------------
class TestCascadeReport:
    def test_splits_accuracy_by_path(self):
        gt = Receipt(total=Decimal("100"))
        right = metrics.compare_document("right", Receipt(total=Decimal("100")), gt)
        wrong = metrics.compare_document("wrong", Receipt(total=Decimal("1")), gt)
        report = cascade_report(
            [right, wrong],
            {"right": {"escalated": True, "reasons": ["low_confidence:total", "low_confidence:tax"]},
             "wrong": {"escalated": False, "reasons": []}},
        )
        assert report["escalation_rate"] == 0.5
        assert report["field_accuracy_escalated"] > report["field_accuracy_accepted"]
        assert report["reasons"] == {"low_confidence": 1}     # counted once per document
        assert report["escalated_docs"] == ["right"]


# --- before/after ---------------------------------------------------------------
def _run(tmp_path, name, *, dar, cost, p50, ids, esc=None, source="pipeline"):
    d = tmp_path / name
    d.mkdir()
    m = {"n_docs": len(ids), "document_accuracy_rate": dar, "micro_f1": 0.9, "macro_f1": 0.8,
         "line_item_f1": 0.7, "hallucination_rate": 0.02,
         "cost_latency": {"cost_per_doc_usd": cost, "tokens_per_doc": 3000.0,
                          "latency": {"p50_s": p50, "p95_s": p50 * 2}}}
    if esc is not None:
        m["cascade"] = {"escalation_rate": esc}
    (d / "scorecard.json").write_text(json.dumps(
        {"meta": {"version": name, "source": source, "set": "test"}, "metrics": m}))
    (d / "per_document.json").write_text(json.dumps([{"doc_id": i} for i in ids]))
    return d


class TestCompare:
    def test_verdicts_follow_metric_direction(self, tmp_path):
        a = cmp.load_run(_run(tmp_path, "base", dar=0.60, cost=0.002, p50=5.0, ids=["a", "b"]))
        b = cmp.load_run(_run(tmp_path, "cas", dar=0.55, cost=0.001, p50=2.0, ids=["a", "b"],
                              esc=0.3, source="cascade"))
        rows = {r["metric"]: r for r in cmp.compare(a, b)["rows"]}
        assert rows["$ / doc (list price)"]["verdict"] == "better"
        assert rows["$ / doc (list price)"]["delta"] == "-50.0%"
        assert rows["Latency p50 (s)"]["verdict"] == "better"
        assert rows["Document accuracy rate"]["verdict"] == "worse"
        assert rows["Document accuracy rate"]["delta"] == "-5.0 pp"
        assert rows["Micro F1"]["verdict"] == "same"
        assert rows["Escalation rate"]["verdict"] == ""        # info only: no direction

    def test_auc_delta_is_not_reported_as_percentage_points(self, tmp_path):
        a_dir = _run(tmp_path, "a", dar=0.5, cost=0.002, p50=5.0, ids=["x"])
        b_dir = _run(tmp_path, "b", dar=0.5, cost=0.002, p50=5.0, ids=["x"])
        for d, auc in ((a_dir, 0.70), (b_dir, 0.75)):
            sc = json.loads((d / "scorecard.json").read_text())
            sc["metrics"]["calibration"] = {"auc": auc, "ece": 0.1}
            (d / "scorecard.json").write_text(json.dumps(sc))
        rows = {r["metric"]: r for r in cmp.compare(cmp.load_run(a_dir), cmp.load_run(b_dir))["rows"]}
        assert rows["Confidence AUC"]["delta"] == "+0.050"

    def test_markdown_states_the_noise_floor(self, tmp_path):
        ids = [f"d{i}" for i in range(40)]
        a = cmp.load_run(_run(tmp_path, "a", dar=0.5, cost=0.002, p50=5.0, ids=ids))
        b = cmp.load_run(_run(tmp_path, "b", dar=0.5, cost=0.001, p50=2.0, ids=ids))
        md = cmp.to_markdown(cmp.compare(a, b))
        assert "one document flips document accuracy by 2.5 pp" in md

    def test_refuses_different_document_sets(self, tmp_path, monkeypatch):
        a = _run(tmp_path, "a", dar=0.5, cost=0.002, p50=5.0, ids=["x", "y"])
        b = _run(tmp_path, "b", dar=0.9, cost=0.001, p50=2.0, ids=["x", "z"])
        monkeypatch.setattr("sys.argv", ["compare", str(a), str(b)])
        with pytest.raises(SystemExit, match="different documents"):
            cmp.main()

    def test_accepts_a_scorecard_path_as_well_as_a_directory(self, tmp_path):
        d = _run(tmp_path, "a", dar=0.5, cost=0.002, p50=5.0, ids=["x"])
        assert cmp.load_run(d / "scorecard.json")[2] == ["x"]
