"""Offline end-to-end check: the whole system over real HTTP, no API keys.

    python -m agent.offline_e2e            # exit 0 = every check passed

Starts the stub model server (agent/stub_provider.py) and the real extraction API in
background threads on free local ports, points the real code at the stub, and checks
behaviour that unit tests cannot see because they mock above the network:

  pipeline      both models called, every printed value GROUNDED (money "53.280", date
                "12/03/2024", currency "Rp"), cost priced per stage
  cache         a second run makes zero model calls but keeps its cost
  multi-page    a 3-page PDF reaches the model as 3 images in ONE call
  truncation    a max_tokens stop fails loudly, not as a JSON parse error
  cascade       a consistent cheap read is accepted; a contradictory one escalates and
                is billed for both paths
  api           /extract withholds abstained fields, /extract/batch isolates a bad
                file, /metrics counts what happened
  eval + gate   eval.run --source api scores through HTTP, and the regression gate
                passes a run compared with itself

Everything is written to a temp directory. It never touches outputs/, .cache/ or .env.
Each check exists because the thing it checks was broken at some point while this
system was built; see the README's agentic-development section.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import httpx
import uvicorn

REPO = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = next(iter(sorted((REPO / "data" / "docs").glob("cord_*.jpg"))), None)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _serve(app, port: int):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError(f"server on :{port} did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


class Checks:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def run(self, name: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn() or ""
            self.results.append((name, True, detail))
            print(f"  PASS  {name}  {detail}")
        except Exception as exc:  # noqa: BLE001 — a failed check is reported, not raised
            self.results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}  {type(exc).__name__}: {exc}")
            if os.environ.get("E2E_VERBOSE"):
                traceback.print_exc()

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.results)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    if SAMPLE_IMAGE is None:
        print("no data/docs/cord_*.jpg to test with - run the dataset converters first")
        return 2

    work = Path(tempfile.mkdtemp(prefix="offline_e2e_"))
    stub_port, api_port = _free_port(), _free_port()
    stub_url = f"http://127.0.0.1:{stub_port}/v1"
    os.environ.update({
        "VLM_A_BASE_URL": stub_url, "VLM_A_MODEL": "stub-vlm", "VLM_A_API_KEY": "stub",
        "LLM_B_BASE_URL": stub_url, "LLM_B_MODEL": "stub-llm", "LLM_B_API_KEY": "stub",
        "VLM_A_PRICE_IN_PER_M": "0.30", "VLM_A_PRICE_OUT_PER_M": "2.50",
        "LLM_B_PRICE_IN_PER_M": "0.04", "LLM_B_PRICE_OUT_PER_M": "0.10",
        "CASCADE_PRICE_IN_PER_M": "0.10", "CASCADE_PRICE_OUT_PER_M": "0.40",
        "CASCADE_ESCALATE_BELOW": "0.0", "EXTRACTION_MODE": "pipeline",
        "CONFIDENCE_MODEL_PATH": str(work / "no_model.json"), "CONFIDENCE_THRESHOLD": "0.5",
        "PROVIDER_MAX_RETRIES": "0", "LOG_LEVEL": "WARNING",
    })

    # Imported after the environment is set; module-level paths are then redirected.
    from agent import stub_provider
    from api import app as api_app
    from api import service
    from extractor import cascade, pipeline
    from extractor.provider import TruncatedResponse
    from eval import regression
    from eval import run as eval_run

    pipeline.CACHE_DIR = work / "cache"
    service.confidence_model.cache_clear()
    checks = Checks()
    print(f"offline e2e  (work dir {work})")

    with _serve(stub_provider.create_app(), stub_port), _serve(api_app.app, api_port) as api_url:
        stub = httpx.Client(base_url=f"http://127.0.0.1:{stub_port}", timeout=10)
        calls = lambda: stub.get("/calls").json()  # noqa: E731

        def check_pipeline() -> str:
            stub_provider.reset()
            r = pipeline.extract_with_confidence(SAMPLE_IMAGE)
            expect([s.stage for s in r.stages] == ["model_a", "model_b"], f"stages {r.stages}")
            expect(r.cost_usd > 0 and all(s.priced for s in r.stages), "stages not priced")
            ungrounded = [f for f in ("subtotal", "tax", "total", "date", "currency")
                          if r.signals[f].grounded != 1.0]
            expect(not ungrounded, f"printed values scored ungrounded: {ungrounded}")
            expect(r.signals["date"].agreement == 0.0, "models disagree on date; agreement must say so")
            expect("line_items" in r.confidence, "table has no confidence")
            return f"${r.cost_usd:.5f}, all printed values grounded, table scored"

        def check_cache() -> str:
            stub_provider.reset()
            first = pipeline.extract_with_confidence(SAMPLE_IMAGE)
            before = calls()["total"]
            second = pipeline.extract_with_confidence(SAMPLE_IMAGE)
            expect(calls()["total"] == before, f"{calls()['total'] - before} new model calls on a rerun")
            expect(all(s.cached for s in second.stages), "rerun stages not marked cached")
            expect(second.cost_usd == first.cost_usd, "cached run lost its cost")
            return "rerun: 0 model calls, cost preserved"

        def check_multipage() -> str:
            from PIL import Image
            pages = [Image.new("RGB", (400, 600), (255, 255 - 30 * i, 255)) for i in range(3)]
            pdf = work / "three_pages.pdf"
            pages[0].save(pdf, save_all=True, append_images=pages[1:])
            stub_provider.reset()
            pipeline.extract_with_confidence(pdf, use_cache=False)
            seen = calls()
            expect(seen["by_model"].get("stub-vlm") == 1, f"VLM calls {seen['by_model']}")
            expect(seen["images_per_request"][0] == 3, f"images sent {seen['images_per_request']}")
            return "3 pages -> 1 VLM call carrying 3 images"

        def check_truncation() -> str:
            from extractor.provider import Provider
            p = Provider(base_url=stub_url, api_key="stub", model="stub-truncate")
            try:
                p.complete([{"role": "user", "content": "x"}], max_tokens=8192)
            except TruncatedResponse:
                return "length stop raised TruncatedResponse"
            raise AssertionError("truncated output was accepted")

        def check_cascade() -> str:
            os.environ["CASCADE_MODEL"] = "stub-lite-clean"
            clean = cascade.extract_cascade(SAMPLE_IMAGE, use_cache=False)
            expect(clean.cascade == {"escalated": False, "reasons": []},
                   f"consistent cheap read escalated: {clean.cascade}")
            expect([s.stage for s in clean.stages] == ["cascade"], "accepted doc billed for more")
            os.environ["CASCADE_MODEL"] = "stub-lite-broken"
            broken = cascade.extract_cascade(SAMPLE_IMAGE, use_cache=False)
            expect(broken.cascade.get("escalated") and
                   "arithmetic_failed" in broken.cascade["reasons"], f"not escalated: {broken.cascade}")
            expect([s.stage for s in broken.stages] == ["cascade", "model_a", "model_b"],
                   "escalated doc not billed for both paths")
            return (f"clean accepted (${clean.cost_usd:.5f}), contradiction escalated "
                    f"(${broken.cost_usd:.5f})")

        def check_api() -> str:
            api = httpx.Client(base_url=api_url, timeout=60)
            expect(api.get("/health").json()["status"] == "ok", "health not ok")
            files = [("files", (SAMPLE_IMAGE.name, SAMPLE_IMAGE.read_bytes(), "image/jpeg")),
                     ("files", ("notes.txt", b"not a receipt", "text/plain"))]
            body = api.post("/extract/batch", files=files).json()
            statuses = [r["status"] for r in body["results"]]
            expect(statuses == ["ok", "error"], f"batch statuses {statuses}")
            # A threshold no field can reach, so withholding is actually exercised. With the
            # normal threshold nothing abstained and this check passed vacuously.
            os.environ["CONFIDENCE_THRESHOLD"] = "0.99"
            try:
                one = api.post("/extract", files={"file": (SAMPLE_IMAGE.name, SAMPLE_IMAGE.read_bytes())}).json()
            finally:
                os.environ["CONFIDENCE_THRESHOLD"] = "0.5"
            expect("total" in one["abstained"] and "line_items" in one["abstained"],
                   f"nothing abstained at threshold 0.99: {one['abstained']}")
            for field in one["abstained"]:
                expect(field in one["withheld"], f"abstained {field} not reported as withheld")
                value = one["receipt"].get(field)
                expect(value in (None, []), f"abstained {field} leaked into receipt: {value!r}")
            metrics = api.get("/metrics").json()
            expect(metrics["documents"]["succeeded"] >= 2 and metrics["documents"]["failed"] >= 1,
                   f"metrics {metrics['documents']}")
            return (f"batch isolated the bad file; {len(one['abstained'])} low-confidence fields "
                    f"withheld incl. the table; metrics counted")

        def check_eval_and_gate() -> str:
            data = work / "data"
            (data / "docs").mkdir(parents=True)
            (data / "ground_truth").mkdir()
            (data / "sets").mkdir()
            ids = []
            for src in sorted((REPO / "data" / "docs").glob("cord_*.jpg"))[:3]:
                (data / "docs" / src.name).write_bytes(src.read_bytes())
                gt = REPO / "data" / "ground_truth" / f"{src.stem}.json"
                (data / "ground_truth" / gt.name).write_text(gt.read_text(encoding="utf-8"), encoding="utf-8")
                ids.append(src.stem)
            (data / "sets" / "e2e.txt").write_text("\n".join(ids) + "\n")
            eval_run.DOCS_DIR, eval_run.GT_DIR, eval_run.SETS_DIR = (
                data / "docs", data / "ground_truth", data / "sets")
            eval_run.OUT_ROOT = work / "outputs"
            eval_run._api_client = httpx.Client(base_url=api_url, timeout=60)

            with contextlib.redirect_stdout(open(os.devnull, "w")), \
                    contextlib.redirect_stderr(open(os.devnull, "w")):
                first = eval_run.run("e2e", "api", "e2e_a")
                time.sleep(1.1)                       # run dirs are stamped to the second
                second = eval_run.run("e2e", "api", "e2e_b")
            meta = json.loads(first.read_text())["meta"]
            expect(meta["source"] == "api" and meta["api"]["model_a"] == "stub-vlm", f"meta {meta}")
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                code = regression.main([str(first.parent), str(second.parent)])
            expect(code == regression.EXIT_OK, f"identical runs failed the regression gate (exit {code})")
            return f"{len(ids)} docs scored over HTTP; regression gate passes identical runs"

        for name, fn in [("pipeline", check_pipeline), ("cache", check_cache),
                         ("multi-page", check_multipage), ("truncation", check_truncation),
                         ("cascade", check_cascade), ("api", check_api),
                         ("eval + gate", check_eval_and_gate)]:
            checks.run(name, fn)

    passed = sum(ok for _, ok, _ in checks.results)
    print(f"\n{passed}/{len(checks.results)} checks passed")
    return 0 if checks.ok else 1


if __name__ == "__main__":
    sys.exit(main())
