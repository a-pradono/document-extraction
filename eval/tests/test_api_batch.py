"""API and batch runner behaviour (Phase 5). Run: pytest eval/tests -q

Model calls are replaced by a fake extractor: what must hold here — error isolation,
abstention at the boundary, bounded concurrency, the cost/latency summary — holds
regardless of what a model returns.
"""

from __future__ import annotations

import json
import threading
import time
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app as app_module
from api import batch, service
from extractor.cost import StageUsage
from extractor.pipeline import PipelineResult
from extractor.schema import Receipt


def fake_result(**overrides) -> PipelineResult:
    result = PipelineResult(
        receipt=Receipt(merchant_name="Kopi", total=Decimal("53280"), date="2024-03-12"),
        receipt_b=None,
        transcription="KOPI\nTOTAL 53.280",
        confidence={"merchant_name": 0.92, "total": 0.31, "date": 0.88},
        abstained=["total"],
        latency_s=4.2,
        wall_s=4.3,
        stages=[StageUsage("model_a", "vlm", 1800, 400, 3.5, 0.00154, priced=True),
                StageUsage("model_b", "llm", 900, 150, 0.7, 0.00005, priced=True)],
        cost_usd=0.00159,
    )
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


def _jpeg_bytes(tmp_path, name="r.jpg") -> bytes:
    path = tmp_path / name
    Image.new("RGB", (20, 30), "white").save(path)
    return path.read_bytes()


@pytest.fixture(autouse=True)
def no_fitted_model(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIDENCE_MODEL_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.5")
    for key in service.REQUIRED_ENV + service.RECOMMENDED_ENV:
        monkeypatch.setenv(key, "test")
    # Pinned: load_dotenv() runs at import, so a developer's real .env would otherwise
    # decide which extractor these tests exercise.
    monkeypatch.setenv("EXTRACTION_MODE", "pipeline")
    monkeypatch.setenv("CASCADE_MODEL", "test")
    service.confidence_model.cache_clear()
    yield
    service.confidence_model.cache_clear()


# --- the shared service / wire format ---------------------------------------
class TestService:
    def test_abstained_value_is_withheld_from_the_receipt(self, tmp_path):
        """A consumer reading `receipt` must never see a value the system isn't
        confident in; a reviewer can still see what was read under `withheld`."""
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpeg_bytes(tmp_path))
        out = service.extract_document(path, "r", extractor=lambda *a, **k: fake_result())
        assert out.status == "ok"
        assert out.receipt.total is None
        assert out.withheld == {"total": "53280"}
        assert out.receipt.merchant_name == "Kopi"          # confident fields untouched

    def test_withholding_does_not_mutate_the_pipeline_result(self, tmp_path):
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpeg_bytes(tmp_path))
        result = fake_result()
        service.extract_document(path, "r", extractor=lambda *a, **k: result)
        assert result.receipt.total == Decimal("53280")     # eval still needs the raw value

    def test_extractor_exception_becomes_an_error_result(self, tmp_path):
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpeg_bytes(tmp_path))

        def boom(*_, **__):
            raise RuntimeError("model timed out")

        out = service.extract_document(path, "r", extractor=boom)
        assert out.status == "error" and "model timed out" in out.error

    def test_unsupported_type_is_an_error_not_a_crash(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hi")
        out = service.extract_document(path, "notes", extractor=lambda *a, **k: fake_result())
        assert out.status == "error" and "unsupported" in out.error

    def test_threshold_and_usage_are_reported(self, tmp_path):
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpeg_bytes(tmp_path))
        out = service.extract_document(path, "r", extractor=lambda *a, **k: fake_result())
        assert out.threshold == 0.5
        assert out.usage.cost_usd == pytest.approx(0.00159)
        assert [s.stage for s in out.usage.stages] == ["model_a", "model_b"]

    def test_model_b_degradation_surfaces_as_a_warning(self, tmp_path):
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpeg_bytes(tmp_path))
        result = fake_result(errors=["model_b_failed: TimeoutError: slow"])
        out = service.extract_document(path, "r", extractor=lambda *a, **k: result)
        assert out.status == "ok" and out.warnings


# --- HTTP -------------------------------------------------------------------
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "extract_with_confidence", lambda *a, **k: fake_result())
    return TestClient(app_module.app)


class TestConfigPreflight:
    def test_fully_configured(self):
        assert service.config_problems() == {"missing": [], "degraded": []}

    def test_missing_model_a_is_fatal(self, monkeypatch):
        monkeypatch.delenv("VLM_A_API_KEY")
        assert service.config_problems()["missing"] == ["VLM_A_API_KEY"]

    def test_cascade_mode_requires_a_cascade_model(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_MODE", "cascade")
        monkeypatch.delenv("CASCADE_MODEL", raising=False)
        assert "CASCADE_MODEL" in service.config_problems()["missing"]

    def test_unknown_mode_is_a_config_error_not_a_silent_default(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_MODE", "cascde")
        assert any("EXTRACTION_MODE" in m for m in service.config_problems()["missing"])

    def test_cascade_mode_routes_to_the_cascade(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EXTRACTION_MODE", "cascade")
        monkeypatch.setattr(service, "extract_cascade",
                            lambda *a, **k: fake_result(cascade={"escalated": False, "reasons": []}))
        monkeypatch.setattr(service, "extract_with_confidence",
                            lambda *a, **k: pytest.fail("pipeline used in cascade mode"))
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpeg_bytes(tmp_path))
        assert service.extract_document(path, "r").cascade == {"escalated": False, "reasons": []}

    def test_missing_model_b_only_degrades(self, monkeypatch):
        monkeypatch.delenv("LLM_B_MODEL")
        problems = service.config_problems()
        assert problems["missing"] == [] and problems["degraded"] == ["LLM_B_MODEL"]

    def test_empty_string_counts_as_missing(self, monkeypatch):
        """.env.example ships `VLM_A_API_KEY=` — present but empty is not configured."""
        monkeypatch.setenv("VLM_A_API_KEY", "")
        assert "VLM_A_API_KEY" in service.config_problems()["missing"]

    def test_unconfigured_api_refuses_with_503_not_502(self, client, monkeypatch, tmp_path):
        monkeypatch.delenv("VLM_A_BASE_URL")
        resp = client.post("/extract", files={"file": ("r.jpg", _jpeg_bytes(tmp_path), "image/jpeg")})
        assert resp.status_code == 503
        assert client.get("/health").json()["status"] == "misconfigured"

    def test_batch_refuses_before_processing_anything(self, monkeypatch, tmp_path):
        """Regression: an unset key used to be recorded as the same KeyError against
        every single document instead of one clear message up front."""
        folder = _folder(tmp_path, ["a.jpg"])
        monkeypatch.delenv("VLM_A_MODEL")
        monkeypatch.setattr("sys.argv", ["batch", str(folder), "--out", str(tmp_path / "out")])
        with pytest.raises(SystemExit, match="VLM_A_MODEL"):
            batch.main()
        assert not (tmp_path / "out").exists()


class TestApi:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["confidence_model"] == "fallback weights"

    def test_health_reports_degraded_without_model_b(self, client, monkeypatch):
        monkeypatch.delenv("LLM_B_API_KEY")
        body = client.get("/health").json()
        assert body["status"] == "degraded" and body["degraded_config"] == ["LLM_B_API_KEY"]

    def test_extract_returns_the_wire_format(self, client, tmp_path):
        resp = client.post("/extract", files={"file": ("r.jpg", _jpeg_bytes(tmp_path), "image/jpeg")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"] == "r"
        assert body["receipt"]["total"] is None and body["abstained"] == ["total"]
        assert body["receipt"]["date"] == "2024-03-12"
        assert body["usage"]["stages"][0]["prompt_tokens"] == 1800

    def test_money_serializes_without_float_rounding(self, monkeypatch, tmp_path):
        result = fake_result(abstained=[], receipt=Receipt(total=Decimal("0.30")))
        monkeypatch.setattr(service, "extract_with_confidence", lambda *a, **k: result)
        resp = TestClient(app_module.app).post(
            "/extract", files={"file": ("r.jpg", _jpeg_bytes(tmp_path), "image/jpeg")})
        assert resp.json()["receipt"]["total"] == "0.30"

    def test_rejects_unsupported_type(self, client):
        resp = client.post("/extract", files={"file": ("r.gif", b"GIF89a", "image/gif")})
        assert resp.status_code == 415

    def test_rejects_empty_upload(self, client):
        resp = client.post("/extract", files={"file": ("r.jpg", b"", "image/jpeg")})
        assert resp.status_code == 400

    def test_rejects_oversized_upload(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
        resp = client.post("/extract", files={"file": ("r.jpg", b"x" * 11, "image/jpeg")})
        assert resp.status_code == 413

    def test_extraction_failure_is_502_with_a_structured_body(self, monkeypatch, tmp_path):
        def boom(*_, **__):
            raise RuntimeError("provider down")

        monkeypatch.setattr(service, "extract_with_confidence", boom)
        resp = TestClient(app_module.app).post(
            "/extract", files={"file": ("r.jpg", _jpeg_bytes(tmp_path), "image/jpeg")})
        assert resp.status_code == 502
        assert resp.json()["detail"]["status"] == "error"

    def test_upload_temp_file_is_cleaned_up(self, monkeypatch, tmp_path):
        seen = []

        def spy(path, *_, **__):
            seen.append(path)
            assert path.exists()
            return fake_result()

        monkeypatch.setattr(service, "extract_with_confidence", spy)
        TestClient(app_module.app).post(
            "/extract", files={"file": ("r.jpg", _jpeg_bytes(tmp_path), "image/jpeg")})
        assert seen and not seen[0].exists()

    def test_concurrent_extractions_are_capped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app_module, "_slots", threading.BoundedSemaphore(2))
        lock, state = threading.Lock(), {"now": 0, "peak": 0}

        def slow(*_, **__):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return fake_result()

        monkeypatch.setattr(service, "extract_with_confidence", slow)
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpeg_bytes(tmp_path))
        threads = [threading.Thread(target=app_module._extract_bounded, args=(path, "r"))
                   for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert 1 < state["peak"] <= 2

    def test_same_filename_uploads_get_distinct_paths(self, monkeypatch, tmp_path):
        seen = []

        def spy(path, *_, **__):
            seen.append(path)
            return fake_result()

        monkeypatch.setattr(service, "extract_with_confidence", spy)
        c = TestClient(app_module.app)
        for _ in range(2):
            c.post("/extract", files={"file": ("receipt.jpg", _jpeg_bytes(tmp_path), "image/jpeg")})
        assert len(set(seen)) == 2


# --- batch ------------------------------------------------------------------
def _folder(tmp_path, names):
    folder = tmp_path / "in"
    folder.mkdir()
    for name in names:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10)).save(target)
    return folder


class TestBatch:
    def test_one_bad_document_never_kills_the_batch(self, tmp_path):
        folder = _folder(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])

        def doc_fn(path, doc_id):
            if doc_id == "b":
                raise RuntimeError("corrupt image")
            return service.extract_document(path, doc_id, extractor=lambda *a, **k: fake_result())

        s = batch.run_batch(folder, tmp_path / "out", concurrency=2, doc_fn=doc_fn)
        assert (s["succeeded"], s["failed"]) == (2, 1)
        assert s["failures"][0]["doc_id"] == "b"

    def test_writes_every_result_and_a_summary(self, tmp_path):
        folder = _folder(tmp_path, ["a.jpg", "b.png"])
        (folder / "skip.txt").write_text("not a receipt")
        out = tmp_path / "out"
        batch.run_batch(folder, out, doc_fn=lambda p, d: service.extract_document(
            p, d, extractor=lambda *a, **k: fake_result()))
        lines = (out / "results.jsonl").read_text().strip().splitlines()
        assert sorted(json.loads(ln)["doc_id"] for ln in lines) == ["a", "b"]
        summary = json.loads((out / "summary.json").read_text())
        assert summary["cost_latency"]["cost_per_doc_usd"] == pytest.approx(0.00159)
        assert summary["abstention"]["by_field"] == {"total": 2}
        assert "Cost and latency" in (out / "summary.md").read_text()

    def test_concurrency_is_bounded(self, tmp_path):
        folder = _folder(tmp_path, [f"d{i}.jpg" for i in range(10)])
        lock, state = threading.Lock(), {"now": 0, "peak": 0}

        def doc_fn(path, doc_id):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return service.extract_document(path, doc_id, extractor=lambda *a, **k: fake_result())

        batch.run_batch(folder, tmp_path / "out", concurrency=3, doc_fn=doc_fn)
        assert 1 < state["peak"] <= 3

    def test_recursive_ids_keep_same_named_files_distinct(self, tmp_path):
        folder = _folder(tmp_path, ["x/receipt.jpg", "y/receipt.jpg"])
        s = batch.run_batch(folder, tmp_path / "out", recursive=True,
                            doc_fn=lambda p, d: service.extract_document(
                                p, d, extractor=lambda *a, **k: fake_result()))
        lines = (tmp_path / "out" / "results.jsonl").read_text().splitlines()
        assert sorted(json.loads(ln)["doc_id"] for ln in lines) == ["x/receipt", "y/receipt"]
        assert s["succeeded"] == 2

    def test_empty_folder_fails_loudly(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(SystemExit):
            batch.run_batch(tmp_path / "empty", tmp_path / "out")

    def test_summarizes_cascade_escalation(self, tmp_path):
        folder = _folder(tmp_path, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])

        def doc_fn(path, doc_id):
            escalated = doc_id in {"a", "b", "c"}
            result = fake_result(cascade={"escalated": escalated, "reasons": []})
            return service.extract_document(path, doc_id, extractor=lambda *a, **k: result)

        s = batch.run_batch(folder, tmp_path / "out", doc_fn=doc_fn)
        assert s["cascade"] == {"documents": 4, "escalated": 3, "escalation_rate": 0.75}
        assert "escalated 3 of 4" in (tmp_path / "out" / "summary.md").read_text()

    def test_no_cascade_block_in_pipeline_mode(self, tmp_path):
        folder = _folder(tmp_path, ["a.jpg"])
        s = batch.run_batch(folder, tmp_path / "out", doc_fn=lambda p, d: service.extract_document(
            p, d, extractor=lambda *a, **k: fake_result()))
        assert s["cascade"] is None

    def test_counts_single_model_degradation(self, tmp_path):
        folder = _folder(tmp_path, ["a.jpg"])
        degraded = fake_result(errors=["model_b_failed: RateLimitError"])
        s = batch.run_batch(folder, tmp_path / "out", doc_fn=lambda p, d: service.extract_document(
            p, d, extractor=lambda *a, **k: degraded))
        assert s["degraded_to_single_model"] == 1
