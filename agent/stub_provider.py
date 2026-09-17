"""A deterministic OpenAI-compatible model server for testing without API keys.

    python -m agent.stub_provider --port 8799

Born during development: with no API keys available, the only way to see the real
provider -> cache -> grounding -> confidence -> HTTP path run end to end was to give it
something to talk to. Unit tests mock at function boundaries; this mocks at the network
boundary, so everything above the socket is the production code.

It answers chat.completions for model names that select behaviour:

    stub-vlm            reads a Kopi Kenangan receipt; every value is printed in its
                        own transcription, so a correct system grounds all of it
    stub-llm            text re-extraction that DISAGREES with stub-vlm on the date, so
                        agreement varies per field
    stub-lite-clean     cheap VLM whose read is fully consistent -> cascade must accept
    stub-lite-broken    cheap VLM whose total contradicts subtotal + tax -> must escalate
    stub-truncate       stops at max_tokens (finish_reason "length")

GET /calls reports how many calls each model received and how many images each request
carried, which is how a check proves "cache hit, zero new calls" or "3 pages, 1 call".
"""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time

from fastapi import FastAPI, Request

TRANSCRIPTION = ("KOPI KENANGAN\nJl. Sudirman No. 45\nTgl 12/03/2024\n"
                 "Es Kopi Susu 2 18.000 36.000\nCroissant 1 12.000 12.000\n"
                 "SUBTOTAL 48.000\nPB1 5.280\nTOTAL Rp 53.280\nQRIS")

FIELDS = {
    "merchant_name": "Kopi Kenangan", "merchant_address": "Jl. Sudirman No. 45",
    "date": "2024-03-12", "currency": "IDR", "subtotal": 48000, "tax": 5280,
    "total": 53280, "payment_method": "QRIS",
    "line_items": [
        {"description": "Es Kopi Susu", "quantity": 2, "unit_price": 18000, "line_total": 36000},
        {"description": "Croissant", "quantity": 1, "unit_price": 12000, "line_total": 12000},
    ],
}
SELF_REPORT = {f: 0.9 for f in FIELDS}

_lock = threading.Lock()
CALLS: dict = {"total": 0, "by_model": {}, "images_per_request": []}


def reset() -> None:
    with _lock:
        CALLS.update(total=0, by_model={}, images_per_request=[])


def _images(body: dict) -> int:
    return sum(1 for m in body.get("messages", []) if isinstance(m.get("content"), list)
               for b in m["content"] if b.get("type") == "image_url")


def _answer(model: str) -> tuple[dict | str, tuple[int, int], str]:
    fields = copy.deepcopy(FIELDS)
    if model == "stub-llm":
        return dict(fields, date="2024-03-21"), (950, 160), "stop"
    if model == "stub-truncate":
        return '{"transcription": "KOPI KEN', (1800, 8192), "length"
    vlm = {"transcription": TRANSCRIPTION, "fields": fields, "_confidence": SELF_REPORT}
    if model == "stub-lite-broken":
        vlm["fields"]["total"] = 99999            # fails subtotal + tax == total
    tokens = (1800, 380) if model.startswith("stub-lite") else (1800, 420)
    return vlm, tokens, "stop"


def create_app() -> FastAPI:
    app = FastAPI(title="stub provider")

    @app.post("/v1/chat/completions")
    async def completions(req: Request):
        body = await req.json()
        model = body["model"]
        with _lock:
            CALLS["total"] += 1
            CALLS["by_model"][model] = CALLS["by_model"].get(model, 0) + 1
            CALLS["images_per_request"].append(_images(body))
        payload, (tok_in, tok_out), finish = _answer(model)
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return {
            "id": f"stub-{CALLS['total']}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": tok_in, "completion_tokens": tok_out,
                      "total_tokens": tok_in + tok_out},
        }

    @app.get("/calls")
    def calls():
        with _lock:
            return copy.deepcopy(CALLS)

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()
    # 127.0.0.1, not localhost: on Windows the client tries IPv6 first and waits ~4s.
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
