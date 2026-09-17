"""Typed, validated receipt schema (Phase 1).

Design rules:
- Money is Decimal, never float.
- Absent fields are None. The extractor must NEVER invent a value that is not on
  the document; it returns null instead.
- `extra="ignore"` keeps the walking skeleton robust to stray keys from the model.
  Switch to "forbid" later if you want hallucinated extra keys to hard-fail.

Phase 4 wraps each field with per-field confidence/abstention/evidence. That lives
in a separate ConfidenceReport keyed by field name so this typed schema stays clean
and easy to validate against ground truth.
"""

from __future__ import annotations

from datetime import date as date_type
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    line_total: Optional[Decimal] = None


class Receipt(BaseModel):
    """A single receipt. Any field not visibly present must be null."""

    model_config = ConfigDict(extra="ignore")

    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    # `date_type` alias, not a bare `date`: the field name shadows the imported type in
    # the class namespace, so `Optional[date]` silently resolves to NoneType and the
    # field can then only ever be null.
    date: Optional[date_type] = None            # normalize to YYYY-MM-DD
    currency: Optional[str] = None              # ISO 4217 where possible, e.g. "IDR"
    subtotal: Optional[Decimal] = None
    # Charges and reductions printed between subtotal and total. Without them the
    # reconciliation subtotal + tax == total is simply false on any receipt with a service
    # charge (12% of CORD) or a tip, so the strongest confidence signal fails on correct
    # extractions.
    service_charge: Optional[Decimal] = Field(
        default=None, description="Service charge, gratuity or tip: an amount ADDED on top of the subtotal.")
    discount: Optional[Decimal] = Field(
        default=None, description="Discount or voucher SUBTRACTED from the bill, as a positive amount.")
    tax: Optional[Decimal] = None
    total: Optional[Decimal] = None
    payment_method: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
