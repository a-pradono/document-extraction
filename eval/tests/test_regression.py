"""Regression gate (Phase 7). Run: pytest eval/tests -q

Run directories are produced by the real scorecard writer from real comparisons, so these
tests also fail if the scorecard format and the regression reader drift apart.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from eval import metrics, regression
from eval.scorecard import write_scorecard
from extractor.cost import StageUsage, summarize
from extractor.schema import LineItem, Receipt

N_DOCS = 10


def _gt(i: int) -> Receipt:
    return Receipt(merchant_name=f"Shop {i}", total=Decimal(100 + i), tax=Decimal(10),
                   subtotal=Decimal(90 + i),
                   line_items=[LineItem(description="Kopi", line_total=Decimal(90 + i))])


FINGERPRINT = {"ground_truth": "gt1", "documents": "docs1", "prompts": "p1",
               "models": {"model_a": "flash", "model_b": "qwen", "cascade_model": None},
               "threshold": 0.5, "confidence_model": "cm1", "code": "abc123"}


def _write_run(tmp: Path, name: str, preds: dict[int, Receipt], *,
               cost: float = 0.002, p95: float = 6.0, auc: float | None = None,
               fp: dict | None = FINGERPRINT) -> Path:
    results = [metrics.compare_document(f"d{i}", preds.get(i, _gt(i)), _gt(i), latency_s=3.0)
               for i in range(N_DOCS)]
    scorecard = metrics.aggregate(results)
    stages = [[StageUsage("model_a", "vlm", 2500, 500, 3.0, cost, priced=True)]
              for _ in range(N_DOCS)]
    scorecard["cost_latency"] = summarize(stages, [3.0] * (N_DOCS - 1) + [p95])
    if auc is not None:
        scorecard["calibration"] = {"available": True, "auc": auc, "ece": 0.1, "bins": []}
    out = tmp / name
    meta = {"version": name, "set": "test", "source": "pipeline"}
    if fp is not None:
        meta["fingerprint"] = fp
    write_scorecard(out, scorecard, results, meta)
    return out


def _with(i: int, **changes) -> Receipt:
    return _gt(i).model_copy(update=changes)


def _check(tmp_path, base_preds, cand_preds, **kw):
    base = _write_run(tmp_path, "base", base_preds, **kw.pop("base_kw", {}))
    cand = _write_run(tmp_path, "cand", cand_preds, **kw.pop("cand_kw", {}))
    return regression.analyse(base, cand, regression.Gates(**kw)), base, cand


class TestGates:
    def test_identical_runs_pass(self, tmp_path):
        report, base, cand = _check(tmp_path, {}, {})
        assert report.passed and not report.warnings
        assert regression.main([str(base), str(cand)]) == regression.EXIT_OK

    def test_field_regression_across_several_documents_fails(self, tmp_path):
        broken = {i: _with(i, total=Decimal(1)) for i in (1, 2, 3)}
        report, base, cand = _check(tmp_path, {}, broken)
        assert not report.passed
        total = next(c for c in report.field_changes if c["field"] == "total")
        assert total["regressed"] and total["newly_wrong"] == ["d1", "d2", "d3"]
        assert regression.main([str(base), str(cand)]) == regression.EXIT_REGRESSION

    def test_a_single_flipped_document_is_noise_not_a_regression(self, tmp_path):
        """At n~40 one document moves a field's F1 by points; failing on that would make
        the gate cry wolf until people stop reading it."""
        report, _, _ = _check(tmp_path, {}, {4: _with(4, total=Decimal(1))},
                              max_dar_drop=1.0)
        assert not any("total" in f for f in report.failures)
        assert any("treated as noise" in w for w in report.warnings)

    def test_min_docs_is_configurable(self, tmp_path):
        report, _, _ = _check(tmp_path, {}, {4: _with(4, total=Decimal(1))},
                              min_docs=1, max_dar_drop=1.0)
        assert any(f.startswith("total:") for f in report.failures)

    def test_any_new_hallucination_fails_by_default(self, tmp_path):
        """Inventing a value is the failure the system promises never to make."""
        report, _, _ = _check(tmp_path, {}, {7: _with(7, payment_method="VISA")},
                              max_dar_drop=1.0)
        assert report.new_hallucinations == [{"doc_id": "d7", "field": "payment_method",
                                              "value": "VISA"}]
        assert any("hallucinated" in f for f in report.failures)

    def test_existing_hallucination_is_not_new(self, tmp_path):
        both = {7: _with(7, payment_method="VISA")}
        report, _, _ = _check(tmp_path, both, both)
        assert report.new_hallucinations == [] and report.passed

    def test_hallucination_allowance(self, tmp_path):
        report, _, _ = _check(tmp_path, {}, {7: _with(7, payment_method="VISA")},
                              allow_new_hallucinations=1, max_dar_drop=1.0)
        assert not any("hallucinated" in f for f in report.failures)

    def test_document_accuracy_drop_fails(self, tmp_path):
        # Different fields per doc, so no single field crosses --min-docs: DAR alone trips.
        broken = {1: _with(1, total=Decimal(1)), 2: _with(2, tax=Decimal(1)),
                  3: _with(3, merchant_name="Nope")}
        report, _, _ = _check(tmp_path, {}, broken)
        assert any(f.startswith("document accuracy") for f in report.failures)

    def test_cost_rise_fails(self, tmp_path):
        report, _, _ = _check(tmp_path, {}, {}, base_kw={"cost": 0.002}, cand_kw={"cost": 0.003})
        assert any("$/doc" in f for f in report.failures)

    def test_cheaper_but_less_accurate_does_not_ship(self, tmp_path):
        """The case the gate exists for: an optimization that cuts cost 35% while
        breaking a field must fail, however good the cost line looks."""
        report, base, cand = _check(
            tmp_path, {}, {i: _with(i, total=Decimal(1)) for i in (1, 2, 3)},
            base_kw={"cost": 0.002}, cand_kw={"cost": 0.0013})
        rows = {r["metric"]: r for r in report.headline["rows"]}
        assert rows["$ / doc (list price)"]["verdict"] == "better"
        assert not report.passed
        assert regression.main([str(base), str(cand)]) == regression.EXIT_REGRESSION

    def test_cost_fall_passes(self, tmp_path):
        report, _, _ = _check(tmp_path, {}, {}, base_kw={"cost": 0.002}, cand_kw={"cost": 0.001})
        assert report.passed

    def test_p95_rise_fails(self, tmp_path):
        report, _, _ = _check(tmp_path, {}, {}, base_kw={"p95": 4.0}, cand_kw={"p95": 9.0})
        assert any("p95" in f for f in report.failures)

    def test_auc_drop_warns_but_does_not_block(self, tmp_path):
        report, _, _ = _check(tmp_path, {}, {}, base_kw={"auc": 0.85}, cand_kw={"auc": 0.70})
        assert report.passed and any("AUC" in w for w in report.warnings)

    def test_line_item_regression(self, tmp_path):
        broken = {i: _with(i, line_items=[LineItem(description="Teh", line_total=Decimal(1))])
                  for i in (0, 1, 2)}
        report, _, _ = _check(tmp_path, {}, broken)
        li = next(c for c in report.field_changes if c["field"] == "line_items")
        assert li["regressed"] and li["newly_wrong"] == ["d0", "d1", "d2"]


class TestReporting:
    def test_newly_failing_documents_show_the_exact_transition(self, tmp_path):
        broken = {i: _with(i, total=Decimal(1)) for i in (1, 2)}
        report, _, _ = _check(tmp_path, {}, broken)
        doc = next(d for d in report.newly_failing if d["doc_id"] == "d1")
        change = next(c for c in doc["changes"] if c["field"] == "total")
        assert (change["from"], change["to"]) == ("correct", "wrong_value")
        md = regression.to_markdown(report)
        assert "total: correct -> wrong_value" in md and "d1" in md

    def test_improvements_are_reported_as_newly_fixed(self, tmp_path):
        broken = {i: _with(i, total=Decimal(1)) for i in (1, 2)}
        report, _, _ = _check(tmp_path, broken, {})
        assert [d["doc_id"] for d in report.newly_fixed] == ["d1", "d2"]
        assert report.passed

    def test_taxonomy_delta_shows_the_kind_of_change(self, tmp_path):
        report, _, _ = _check(tmp_path, {}, {i: _with(i, total=None) for i in (1, 2, 3)})
        assert report.taxonomy_delta["missing"] == 3
        assert report.taxonomy_delta["correct"] == -3

    def test_markdown_verdict_and_headline(self, tmp_path):
        report, _, _ = _check(tmp_path, {}, {})
        md = regression.to_markdown(report)
        assert md.startswith("# Regression check: PASS") and "## Headline" in md

    def test_json_output(self, tmp_path):
        base = _write_run(tmp_path, "base", {})
        cand = _write_run(tmp_path, "cand", {i: _with(i, total=Decimal(1)) for i in (1, 2, 3)})
        out = tmp_path / "r.json"
        assert regression.main([str(base), str(cand), "--json", str(out)]) == 1
        payload = json.loads(out.read_text())
        assert payload["passed"] is False and payload["newly_failing"]


class TestComparability:
    def _fp(self, **changes):
        fp = json.loads(json.dumps(FINGERPRINT))
        for key, value in changes.items():
            if key.startswith("models."):
                fp["models"][key.split(".", 1)[1]] = value
            else:
                fp[key] = value
        return fp

    def test_relabeled_ground_truth_is_refused(self, tmp_path):
        """Verifying labels fixes some of them. A diff across that fix would report the
        relabeling as a system regression or improvement."""
        base = _write_run(tmp_path, "base", {})
        cand = _write_run(tmp_path, "cand", {}, fp=self._fp(ground_truth="gt2"))
        with pytest.raises(ValueError, match="ground_truth changed"):
            regression.analyse(base, cand, regression.Gates())
        assert regression.main([str(base), str(cand)]) == regression.EXIT_INVALID

    def test_changed_document_bytes_are_refused(self, tmp_path):
        base = _write_run(tmp_path, "base", {})
        cand = _write_run(tmp_path, "cand", {}, fp=self._fp(documents="docs2"))
        assert regression.main([str(base), str(cand)]) == regression.EXIT_INVALID

    def test_label_drift_can_be_allowed_explicitly_with_a_warning(self, tmp_path):
        base = _write_run(tmp_path, "base", {})
        cand = _write_run(tmp_path, "cand", {}, fp=self._fp(ground_truth="gt2"))
        report = regression.analyse(base, cand, regression.Gates(allow_label_drift=True))
        assert any("relabeling" in w for w in report.warnings)
        assert regression.main([str(base), str(cand), "--allow-label-drift"]) == regression.EXIT_OK

    def test_config_changes_are_attributed_in_the_report(self, tmp_path):
        base = _write_run(tmp_path, "base", {})
        cand = _write_run(tmp_path, "cand", {}, fp=self._fp(**{"models.model_a": "flash-lite",
                                                                 "prompts": "p2"}))
        report = regression.analyse(base, cand, regression.Gates())
        assert report.config_changes == {"models.model_a": ("flash", "flash-lite"),
                                         "prompts": ("p1", "p2")}
        md = regression.to_markdown(report)
        assert "What changed between the runs" in md and "flash-lite" in md

    def test_runs_without_fingerprints_warn(self, tmp_path):
        base = _write_run(tmp_path, "base", {}, fp=None)
        cand = _write_run(tmp_path, "cand", {}, fp=None)
        report = regression.analyse(base, cand, regression.Gates())
        assert any("predates fingerprinting" in w for w in report.warnings)


class TestFingerprint:
    def test_relabeling_one_field_changes_the_ground_truth_hash(self, tmp_path):
        from eval import fingerprint
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text('{"total": "100"}')
        b.write_text('{"total": "101"}')
        assert fingerprint.ground_truth_sha(["d1"], lambda _: a) != \
            fingerprint.ground_truth_sha(["d1"], lambda _: b)

    def test_hash_is_order_independent(self, tmp_path):
        from eval import fingerprint
        files = {d: tmp_path / f"{d}.json" for d in ("x", "y")}
        for d, p in files.items():
            p.write_text(f'{{"merchant_name": "{d}"}}')
        assert fingerprint.ground_truth_sha(["x", "y"], files.__getitem__) == \
            fingerprint.ground_truth_sha(["y", "x"], files.__getitem__)

    def test_prompt_edit_changes_the_prompt_hash(self, monkeypatch):
        from eval import fingerprint
        from extractor import pipeline
        before = fingerprint.prompts_sha()
        monkeypatch.setattr(pipeline, "VLM_SYSTEM", pipeline.VLM_SYSTEM + " Be terse.")
        assert fingerprint.prompts_sha() != before

    def test_real_run_records_a_fingerprint(self, tmp_path, monkeypatch):
        from eval import run as eval_run
        from extractor.pipeline import PipelineResult
        from PIL import Image
        data = tmp_path / "data"
        for sub in ("docs", "ground_truth", "sets"):
            (data / sub).mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(data / "docs" / "d1.png")
        (data / "ground_truth" / "d1.json").write_text('{"total": "1"}')
        (data / "sets" / "one.txt").write_text("d1\n")
        for attr, sub in (("DOCS_DIR", "docs"), ("GT_DIR", "ground_truth"), ("SETS_DIR", "sets")):
            monkeypatch.setattr(eval_run, attr, data / sub)
        monkeypatch.setattr(eval_run, "OUT_ROOT", tmp_path / "out")
        monkeypatch.setattr(eval_run, "extract_with_confidence", lambda *a, **k: PipelineResult(
            document=Receipt(total=Decimal("1")), document_b=None, transcription=""))
        path = eval_run.run("one", "pipeline", "fp")
        fp = json.loads(path.read_text())["meta"]["fingerprint"]
        assert {"ground_truth", "documents", "prompts", "models", "threshold"} <= set(fp)
        assert fp["ground_truth"] and fp["documents"]


class TestBreakdown:
    def test_subsets_by_source_and_degraded_tag(self, tmp_path, monkeypatch):
        from eval import run as eval_run
        results = ([metrics.compare_document(f"cord_{i}", _gt(i), _gt(i)) for i in range(4)] +
                   [metrics.compare_document(f"wild_{i}", _with(i, total=Decimal(1)), _gt(i))
                    for i in range(4)])
        sets = tmp_path / "sets"
        sets.mkdir()
        (sets / "degraded.txt").write_text("# tag\nwild_0\nwild_1\nwild_2\n")
        monkeypatch.setattr(eval_run, "SETS_DIR", sets)
        bd = eval_run.breakdown(results)
        assert bd["source:cord"]["document_accuracy_rate"] == 1.0
        assert bd["source:wild"]["document_accuracy_rate"] == 0.0
        assert bd["source:wild"]["error_taxonomy"] == {"wrong_value": 4}
        assert bd["degraded"]["n_docs"] == 3 and bd["not degraded"]["n_docs"] == 5

    def test_tiny_subsets_are_omitted(self, tmp_path, monkeypatch):
        from eval import run as eval_run
        monkeypatch.setattr(eval_run, "SETS_DIR", tmp_path)
        results = ([metrics.compare_document(f"cord_{i}", _gt(i), _gt(i)) for i in range(4)] +
                   [metrics.compare_document("retail_0", _gt(0), _gt(0))])
        assert set(eval_run.breakdown(results)) == {"source:cord"}


class TestInvalidComparisons:
    def test_different_document_sets_exit_2(self, tmp_path):
        base = _write_run(tmp_path, "base", {})
        cand = _write_run(tmp_path, "cand", {})
        per_doc = json.loads((cand / "per_document.json").read_text())
        per_doc[0]["doc_id"] = "other"
        (cand / "per_document.json").write_text(json.dumps(per_doc))
        assert regression.main([str(base), str(cand)]) == regression.EXIT_INVALID

    def test_missing_run_exit_2(self, tmp_path):
        base = _write_run(tmp_path, "base", {})
        (tmp_path / "empty").mkdir()
        assert regression.main([str(base), str(tmp_path / "empty")]) == regression.EXIT_INVALID
