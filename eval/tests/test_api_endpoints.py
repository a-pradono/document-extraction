"""/extract/batch, /metrics, and `eval.run --source api` (Phase 8). Run: pytest eval/tests -q

The --source api tests drive the real eval client against the real FastAPI app through
TestClient, so the HTTP contract between harness and service is exercised end to end;
only the model call underneath is faked.
"""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app as app_module
from api import service
from api.telemetry import Telemetry
from eval import run as eval_run
from eval.tests.test_api_batch import fake_result

pytestmark = pytest.mark.usefixtures("configured")


@pytest.fixture
def configured(monkeypatch, tmp_path):
    for key in service.REQUIRED_ENV + service.RECOMMENDED_ENV:
        monkeypatch.setenv(key, "test")
    monkeypatch.setenv("EXTRACTION_MODE", "pipeline")
    monkeypatch.setenv("CONFIDENCE_MODEL_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.5")
    service.confidence_model.cache_clear()
    monkeypatch.setattr(app_module, "telemetry", Telemetry())
    monkeypatch.setattr(service, "extract_with_confidence", lambda *a, **k: fake_result())
    yield
    service.confidence_model.cache_clear()


def _jpg(tmp_path, name="r.jpg") -> bytes:
    path = tmp_path / name
    Image.new("RGB", (20, 30), "white").save(path)
    return path.read_bytes()


@pytest.fixture
def client():
    return TestClient(app_module.app)


# --- /extract/batch ---------------------------------------------------------------
class TestBatchEndpoint:
    def test_returns_results_in_upload_order_with_a_summary(self, client, tmp_path):
        files = [("files", (f"{n}.jpg", _jpg(tmp_path), "image/jpeg")) for n in ("c", "a", "b")]
        body = client.post("/extract/batch", files=files).json()
        assert [r["doc_id"] for r in body["results"]] == ["c", "a", "b"]
        assert body["summary"]["succeeded"] == 3
        assert body["summary"]["cost_latency"]["cost_per_doc_usd"] == pytest.approx(0.00159)

    def test_bad_files_become_error_entries_not_a_failed_request(self, client, tmp_path):
        files = [
            ("files", ("good.jpg", _jpg(tmp_path), "image/jpeg")),
            ("files", ("notes.txt", b"hello", "text/plain")),
            ("files", ("empty.jpg", b"", "image/jpeg")),
        ]
        resp = client.post("/extract/batch", files=files)
        assert resp.status_code == 200
        by_id = {r["doc_id"]: r for r in resp.json()["results"]}
        assert by_id["good"]["status"] == "ok"
        assert by_id["notes"]["status"] == "error" and "unsupported" in by_id["notes"]["error"]
        assert by_id["empty"]["status"] == "error" and "empty" in by_id["empty"]["error"]
        assert resp.json()["summary"]["failed"] == 2

    def test_one_failed_extraction_does_not_sink_the_rest(self, client, tmp_path, monkeypatch):
        def flaky(path, *_, **__):
            if b"FAIL" in path.read_bytes()[-4:]:
                raise RuntimeError("provider timeout")
            return fake_result()

        monkeypatch.setattr(service, "extract_with_confidence", flaky)
        files = [("files", ("ok.jpg", _jpg(tmp_path), "image/jpeg")),
                 ("files", ("bad.jpg", _jpg(tmp_path) + b"FAIL", "image/jpeg"))]
        results = {r["doc_id"]: r for r in client.post("/extract/batch", files=files).json()["results"]}
        assert results["ok"]["status"] == "ok"
        assert results["bad"]["status"] == "error" and "provider timeout" in results["bad"]["error"]

    def test_duplicate_filenames_stay_distinguishable(self, client, tmp_path):
        files = [("files", ("receipt.jpg", _jpg(tmp_path), "image/jpeg")) for _ in range(3)]
        ids = [r["doc_id"] for r in client.post("/extract/batch", files=files).json()["results"]]
        assert ids == ["receipt", "receipt#2", "receipt#3"]

    def test_too_many_files_is_413(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "MAX_BATCH_FILES", 2)
        files = [("files", (f"{i}.jpg", _jpg(tmp_path), "image/jpeg")) for i in range(3)]
        assert client.post("/extract/batch", files=files).status_code == 413

    def test_unconfigured_is_503(self, client, tmp_path, monkeypatch):
        monkeypatch.delenv("VLM_A_API_KEY")
        files = [("files", ("a.jpg", _jpg(tmp_path), "image/jpeg"))]
        assert client.post("/extract/batch", files=files).status_code == 503

    def test_withholding_applies_to_batch_results_too(self, client, tmp_path):
        files = [("files", ("a.jpg", _jpg(tmp_path), "image/jpeg"))]
        result = client.post("/extract/batch", files=files).json()["results"][0]
        assert result["document"]["total"] is None and result["withheld"] == {"total": "53280"}


# --- /metrics ------------------------------------------------------------------------
class TestMetricsEndpoint:
    def test_starts_empty(self, client):
        m = client.get("/metrics").json()
        assert m["documents"]["processed"] == 0 and m["cost"]["per_doc_usd"] == 0.0

    def test_counts_single_and_batch_extractions(self, client, tmp_path):
        client.post("/extract", files={"file": ("a.jpg", _jpg(tmp_path), "image/jpeg")})
        client.post("/extract/batch", files=[
            ("files", ("b.jpg", _jpg(tmp_path), "image/jpeg")),
            ("files", ("c.txt", b"x", "text/plain")),
        ])
        m = client.get("/metrics").json()
        assert m["requests"] == 2
        assert (m["documents"]["succeeded"], m["documents"]["failed"]) == (2, 1)
        assert m["abstention_rate"] == pytest.approx(2 / 6, abs=1e-4)  # 1 of 3 fields, twice
        assert m["cost"]["total_usd"] == pytest.approx(2 * 0.00159)
        assert m["latency_window"]["model"]["n"] == 2

    def test_rejected_requests_are_counted_separately(self, client):
        client.post("/extract", files={"file": ("a.gif", b"GIF89a", "image/gif")})
        m = client.get("/metrics").json()
        assert m["rejected_requests"] == 1 and m["documents"]["processed"] == 0

    def test_escalation_rate_in_cascade_mode(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("EXTRACTION_MODE", "cascade")
        monkeypatch.setenv("CASCADE_MODEL", "lite")
        results = iter([fake_result(cascade={"escalated": True, "reasons": ["x"]}),
                        fake_result(cascade={"escalated": False, "reasons": []})])
        monkeypatch.setattr(service, "extract_cascade", lambda *a, **k: next(results))
        for n in ("a", "b"):
            client.post("/extract", files={"file": (f"{n}.jpg", _jpg(tmp_path), "image/jpeg")})
        assert client.get("/metrics").json()["cascade"]["escalation_rate"] == 0.5

    def test_telemetry_window_is_bounded(self):
        t = Telemetry(window=3)
        for _ in range(10):
            t.record(service.from_result("d", fake_result(), 0.5))
        assert t.snapshot()["latency_window"]["model"]["n"] == 3
        assert t.snapshot()["documents"]["succeeded"] == 10     # counters are not windowed


# --- eval.run --source api -------------------------------------------------------------
@pytest.fixture
def api_source(monkeypatch, client):
    monkeypatch.setattr(eval_run, "_api_client", client)
    return client


class TestEvalSourceApi:
    def test_withheld_values_are_restored_for_scoring(self, api_source, tmp_path):
        """The API nulls abstained fields for consumers; the harness must score what was
        actually read, or an API run disagrees with a local run of the same system."""
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpg(tmp_path))
        ext = eval_run.extract_one(path, "api", None, 0.5, True)
        assert ext.document.total == Decimal("53280")
        assert ext.abstained == ["total"]
        assert ext.confidence["total"] == 0.31

    def test_usage_and_stages_survive_the_round_trip(self, api_source, tmp_path):
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpg(tmp_path))
        ext = eval_run.extract_one(path, "api", None, 0.5, True)
        assert [s.stage for s in ext.stages] == ["model_a", "model_b"]
        assert ext.cost_usd == pytest.approx(0.00159) and ext.latency_s == 4.2

    def test_extraction_failure_raises_so_the_doc_is_recorded_as_failed(
            self, api_source, tmp_path, monkeypatch):
        def boom(*_, **__):
            raise RuntimeError("provider down")

        monkeypatch.setattr(service, "extract_with_confidence", boom)
        path = tmp_path / "r.jpg"
        path.write_bytes(_jpg(tmp_path))
        with pytest.raises(RuntimeError, match="provider down"):
            eval_run.extract_one(path, "api", None, 0.5, True)

    def test_preflight_uses_the_servers_threshold(self, api_source, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.73")
        assert eval_run.api_preflight()["threshold"] == 0.73

    def test_preflight_refuses_a_misconfigured_server(self, api_source, monkeypatch):
        monkeypatch.delenv("VLM_A_MODEL")
        with pytest.raises(SystemExit, match="misconfigured"):
            eval_run.api_preflight()

    def test_preflight_refuses_an_unreachable_server(self, monkeypatch):
        monkeypatch.setattr(eval_run, "_api_client",
                            httpx.Client(base_url="http://127.0.0.1:9", timeout=0.5))
        with pytest.raises(SystemExit, match="not reachable"):
            eval_run.api_preflight()

    def test_full_run_writes_a_scorecard_with_server_meta(self, api_source, tmp_path, monkeypatch):
        import json
        import shutil

        data = tmp_path / "data"
        (data / "docs").mkdir(parents=True)
        (data / "ground_truth").mkdir()
        (data / "sets").mkdir()
        (data / "docs" / "d1.jpg").write_bytes(_jpg(tmp_path))
        (data / "ground_truth" / "d1.json").write_text(json.dumps({"total": "53280"}))
        (data / "sets" / "mini.txt").write_text("d1\n")
        monkeypatch.setattr(eval_run, "DOCS_DIR", data / "docs")
        monkeypatch.setattr(eval_run, "GT_DIR", data / "ground_truth")
        monkeypatch.setattr(eval_run, "SETS_DIR", data / "sets")
        monkeypatch.setattr(eval_run, "OUT_ROOT", tmp_path / "out")
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.4")

        path = eval_run.run("mini", "api", "apitest")
        sc = json.loads(path.read_text())
        assert sc["meta"]["source"] == "api" and sc["meta"]["threshold"] == 0.4
        assert sc["meta"]["api"]["mode"] == "pipeline"
        # total was abstained by the server yet correct: restored and scored as correct.
        total = next(c for c in json.loads((path.parent / "per_document.json").read_text())[0]
                     ["fields"] if c["field"] == "total")
        assert total["label"] == "correct"
        shutil.rmtree(tmp_path / "out")
