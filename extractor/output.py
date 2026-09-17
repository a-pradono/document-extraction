"""Turning raw model output into a Receipt without losing a document to one bad value.

Two layers, because models fail in two different ways:

1. The text is not JSON at all (a stray sentence, an unclosed brace). One TEXT-ONLY repair
   call sends the malformed output back and asks for the same content as valid JSON. No
   images go with it, so it costs a small fraction of a vision call. Output that was cut
   off at max_tokens is never repaired: the missing half would be invented, so
   TruncatedResponse propagates instead.

2. The JSON parses but a value breaks the schema (a date of "12 Maret", a total of
   "N/A"). Validating the whole object would throw away every other correct field with
   it. Instead each field is validated on its own: valid fields are kept, invalid ones
   become null AND are marked invalid, and the pipeline then forces their confidence to 0.
   They are abstained on, not silently dropped, so "the model produced garbage here"
   stays visible. A table with an invalid row is marked invalid as a whole: shipping it
   one row short would look complete while being wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dc_field
from typing import Any

from pydantic import ValidationError

from extractor.provider import LLMResponse, Provider
from extractor.schema import LineItem, Receipt

REPAIR_SYSTEM = ("You fix malformed JSON. Return the SAME content as one valid JSON object. "
                 "Do not add, remove, or change any value. Return only the JSON.")


class ModelOutputError(ValueError):
    """The model's output could not be parsed as JSON, even after one repair attempt."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def parse_json_object(text: str) -> dict:
    cleaned = _strip_fences(text)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)     # an object wrapped in prose
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise json.JSONDecodeError("top-level JSON value is not an object", cleaned, 0)
    return value


def parse_or_repair(resp: LLMResponse, provider: Provider, usage: dict) -> dict:
    """Parse the response, spending at most one text-only repair call.

    `usage` is updated in place so the repair's tokens and latency land on the bill.
    """
    try:
        return parse_json_object(resp.text)
    except json.JSONDecodeError as first_error:
        try:
            fix = provider.complete(
                [{"role": "system", "content": REPAIR_SYSTEM},
                 {"role": "user", "content": resp.text}],
                json_mode=True,
                max_tokens=max(1024, resp.completion_tokens * 2),
            )
        except Exception as exc:  # noqa: BLE001 — report the original failure, with context
            raise ModelOutputError(f"invalid JSON ({first_error}); repair call failed: {exc}") from exc
        usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + fix.prompt_tokens
        usage["completion_tokens"] = usage.get("completion_tokens", 0) + fix.completion_tokens
        usage["latency_s"] = usage.get("latency_s", 0.0) + fix.latency_s
        usage["repaired"] = True
        try:
            return parse_json_object(fix.text)
        except json.JSONDecodeError as exc:
            raise ModelOutputError(f"invalid JSON even after one repair: {exc}") from exc


@dataclass
class Salvaged:
    document: BaseModel
    invalid: list[str] = dc_field(default_factory=list)     # field names forced to abstain
    notes: list[str] = dc_field(default_factory=list)


def salvage_receipt(fields: Any, model: type[BaseModel] = Receipt) -> Salvaged:
    """Validate field by field so one bad value can't cost the whole document.

    Works for any document model. A nested object (an invoice's `vendor`) is validated as
    a unit, so an invalid party is marked invalid as `vendor` and every `vendor.*` field
    it contains is abstained."""
    if fields is None:
        return Salvaged(model())
    if not isinstance(fields, dict):
        return Salvaged(model(), invalid=["*"],
                        notes=[f"fields was {type(fields).__name__}, not an object"])
    try:
        return Salvaged(model.model_validate(fields))
    except ValidationError:
        pass

    kept: dict[str, Any] = {}
    out = Salvaged(model())
    for name in model.model_fields:
        if name not in fields or name == "line_items":
            continue
        try:
            model.model_validate({name: fields[name]})
            kept[name] = fields[name]
        except ValidationError as exc:
            out.invalid.append(name)
            out.notes.append(f"invalid {name}={fields[name]!r}: {exc.errors()[0]['msg']}")

    rows = fields.get("line_items")
    if rows is not None:
        if not isinstance(rows, list):
            out.invalid.append("line_items")
            out.notes.append(f"line_items was {type(rows).__name__}, not a list")
        else:
            good = []
            for i, row in enumerate(rows):
                try:
                    good.append(LineItem.model_validate(row))
                except ValidationError as exc:
                    out.notes.append(f"invalid line_items[{i}]: {exc.errors()[0]['msg']}")
            if len(good) < len(rows):
                out.invalid.append("line_items")
            kept["line_items"] = good

    out.document = model.model_validate(kept)
    return out


def is_invalid(field: str, invalid: list[str]) -> bool:
    """`vendor.name` is invalid if `vendor` as a whole was, or if everything was."""
    return "*" in invalid or field in invalid or field.split(".", 1)[0] in invalid
