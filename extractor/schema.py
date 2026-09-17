"""Typed, validated document schemas: Receipt (primary) and Invoice (second type).

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


class Party(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    address: Optional[str] = None
    tax_id: Optional[str] = None


class Invoice(BaseModel):
    """An invoice. Reuses LineItem so line-item matching and validation are unchanged.

    The graded part is ROLE RESOLUTION: vendor and customer look alike on the page (two
    names, two addresses) and only position and labelling ("Bill To", "Remit to", "c/o")
    say which is which. A swap is scored as `swapped_field`.
    """

    model_config = ConfigDict(extra="ignore")

    invoice_number: Optional[str] = None
    issue_date: Optional[date_type] = None
    due_date: Optional[date_type] = None
    vendor: Party = Field(
        default_factory=Party,
        description="The party that ISSUED the invoice and is to be paid (seller/supplier).")
    customer: Party = Field(
        default_factory=Party,
        description="The party BILLED, who is to pay (buyer, 'Bill To'). A 'c/o' or "
                    "'Attention' person belongs to the customer, not the vendor.")
    currency: Optional[str] = None
    subtotal: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    total: Optional[Decimal] = None
    purchase_order: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)


DOCUMENT_MODELS: dict[str, type[BaseModel]] = {"receipt": Receipt, "invoice": Invoice}


def document_type_of(doc_id: str) -> str:
    """The one place a document's type is decided: invoice ids are prefixed `inv_`."""
    return "invoice" if doc_id.startswith("inv_") else "receipt"


def model_for(doc_id_or_type: str) -> type[BaseModel]:
    key = doc_id_or_type if doc_id_or_type in DOCUMENT_MODELS else document_type_of(doc_id_or_type)
    return DOCUMENT_MODELS[key]
