"""The second document type: invoices, role resolution, and never pooling types.
Run: pytest eval/tests -q
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image

from api.schemas import from_result
from eval import metrics, taxonomy
from eval.run import breakdown  # noqa: F401  (import check: run module loads with invoices)
from extractor.cascade import escalation_reasons
from extractor.confidence import group_of, validate_arithmetic
from extractor.output import is_invalid, salvage_receipt
from extractor.pipeline import PipelineResult, build_signals
from extractor.schema import Invoice, Receipt, document_type_of, model_for

VENDOR = {"name": "Correll Law Group", "address": "250 Park Avenue, New York, NY 10177"}
CUSTOMER = {"name": "Wayne LaPierre", "address": "11250 Waples Mill Road, Fairfax, VA 22030"}


def invoice(**changes) -> Invoice:
    base = {"invoice_number": "2020.03.05", "issue_date": "2021-01-13", "vendor": VENDOR,
            "customer": CUSTOMER, "total": "8840.00",
            "line_items": [{"description": "Legal services", "line_total": "8840.00"}]}
    base.update(changes)
    return Invoice.model_validate(base)


class TestDocumentType:
    def test_type_comes_from_the_id_prefix(self):
        assert document_type_of("inv_0003") == "invoice"
        assert document_type_of("cord_0003") == "receipt"
        assert model_for("inv_0001") is Invoice and model_for("wild_0001") is Receipt

    def test_invoice_dates_resolve_to_real_dates(self):
        """The same field-name shadowing trap that once made Receipt.date always null."""
        inv = invoice(due_date="2021-02-12")
        assert inv.issue_date.isoformat() == "2021-01-13" and inv.due_date.year == 2021

    def test_invoice_reuses_line_item(self):
        assert Invoice.model_fields["line_items"].annotation == Receipt.model_fields["line_items"].annotation


class TestRoleResolutionScoring:
    def test_nested_party_fields_are_scored(self):
        result = metrics.compare_document("inv", invoice(), invoice())
        fields = {c.field for c in result.comparisons}
        assert {"vendor.name", "customer.name", "vendor.address", "customer.address"} <= fields
        assert result.document_correct

    def test_swapped_parties_are_labeled_swapped_field(self):
        """The graded failure for invoices: vendor and customer returned the wrong way round."""
        swapped = invoice(vendor=CUSTOMER, customer=VENDOR)
        result = metrics.compare_document("inv", swapped, invoice())
        labels = {c.field: c.label for c in result.comparisons}
        assert labels["vendor.name"] == taxonomy.SWAPPED_FIELD
        assert labels["customer.name"] == taxonomy.SWAPPED_FIELD
        assert labels["vendor.address"] == taxonomy.SWAPPED_FIELD

    def test_a_missing_customer_is_missing_not_swapped(self):
        result = metrics.compare_document("inv", invoice(customer={}), invoice())
        labels = {c.field: c.label for c in result.comparisons}
        assert labels["customer.name"] == taxonomy.MISSING

    def test_receipt_and_invoice_cannot_be_compared(self):
        with pytest.raises(TypeError, match="own type"):
            metrics.compare_document("x", Receipt(total="1"), invoice())

    def test_party_fields_use_the_string_confidence_group(self):
        assert group_of("vendor.name") == "string" and group_of("issue_date") == "date"


class TestInvoiceValidation:
    def test_due_before_issue_fails_both_dates(self):
        v = validate_arithmetic(invoice(issue_date="2025-11-24", due_date="2024-07-20"))
        assert v["issue_date"] == v["due_date"] == 0.0

    def test_ordered_dates_pass(self):
        v = validate_arithmetic(invoice(issue_date="2020-01-09", due_date="2020-02-08"))
        assert v["issue_date"] == v["due_date"] == 1.0

    def test_same_name_on_both_parties_fails_role_resolution(self):
        v = validate_arithmetic(invoice(customer={"name": "correll law group!"}))
        assert v["vendor.name"] == v["customer.name"] == 0.0

    def test_distinct_names_earn_no_credit(self):
        """Two different names can still be the wrong way round."""
        assert "vendor.name" not in validate_arithmetic(invoice())

    def test_printed_arithmetic_error_is_caught(self):
        """inv_0004 prints rows of 8.05 + 48.51 but a subtotal of 36.59."""
        doc = invoice(subtotal="36.59", total="36.59",
                      line_items=[{"description": "a", "line_total": "8.05"},
                                  {"description": "b", "line_total": "48.51"}])
        assert validate_arithmetic(doc)["line_items"] == 0.0

    def test_cheap_path_escalates_on_inconsistent_invoice(self):
        doc = invoice(issue_date="2025-11-24", due_date="2024-07-20")
        signals = build_signals(doc, None, "INVOICE Correll Law Group 13/01/2021 8,840.00", {})
        assert "consistency_failed" in escalation_reasons(doc, signals, {}, 0.0)


class TestSignalsAndSalvage:
    TEXT = ("Correll Law Group 250 Park Avenue, New York, NY 10177 INVOICE NUMBER: 2020.03.05 "
            "INVOICE DATE: 1/13/2021 TO: Wayne LaPierre 11250 Waples Mill Road, Fairfax, VA 22030 "
            "Legal services $8,840.00 AMOUNT DUE: $8,840.00")

    def test_party_names_are_grounded_in_the_transcription(self):
        signals = build_signals(invoice(), None, self.TEXT, {})
        assert signals["vendor.name"].grounded == 1.0
        assert signals["customer.name"].grounded == 1.0
        assert signals["issue_date"].grounded == 1.0

    def test_invented_party_is_ungrounded(self):
        signals = build_signals(invoice(customer={"name": "Acme Holdings"}), None, self.TEXT, {})
        assert signals["customer.name"].grounded == 0.0

    def test_invalid_nested_party_is_abstained_as_a_whole(self):
        s = salvage_receipt({"invoice_number": "1", "vendor": "not an object", "total": "5"}, Invoice)
        assert s.document.invoice_number == "1" and s.document.total == Decimal("5")
        assert "vendor" in s.invalid
        assert is_invalid("vendor.name", s.invalid) and not is_invalid("customer.name", s.invalid)


class TestApiBoundary:
    def test_abstained_party_field_is_withheld_in_place(self):
        result = PipelineResult(document=invoice(), document_b=None, transcription="",
                                confidence={"customer.name": 0.1}, abstained=["customer.name"])
        out = from_result("inv_0008", result, 0.5, "invoice")
        assert out.document_type == "invoice"
        assert out.document.customer.name is None
        assert out.document.customer.address == CUSTOMER["address"]      # siblings untouched
        assert out.withheld == {"customer.name": "Wayne LaPierre"}

    def test_api_accepts_and_validates_document_type(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        from api import app as app_module
        from api import service

        for key in service.REQUIRED_ENV + service.RECOMMENDED_ENV:
            monkeypatch.setenv(key, "test")
        monkeypatch.setenv("EXTRACTION_MODE", "pipeline")
        seen = {}

        def fake(path, **kwargs):
            seen.update(kwargs)
            return PipelineResult(document=invoice(), document_b=None, transcription="")

        monkeypatch.setattr(service, "extract_with_confidence", fake)
        page = tmp_path / "inv.png"
        Image.new("RGB", (20, 30), "white").save(page)
        client = TestClient(app_module.app)

        body = client.post("/extract", files={"file": ("inv.png", page.read_bytes(), "image/png")},
                           data={"document_type": "invoice"}).json()
        assert seen["doc_type"] == "invoice" and body["document_type"] == "invoice"
        assert body["document"]["vendor"]["name"] == "Correll Law Group"

        bad = client.post("/extract", files={"file": ("inv.png", page.read_bytes(), "image/png")},
                          data={"document_type": "passport"})
        assert bad.status_code == 400

    def test_eval_client_restores_withheld_nested_fields(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        from api import app as app_module
        from api import service
        from eval import run as eval_run

        for key in service.REQUIRED_ENV + service.RECOMMENDED_ENV:
            monkeypatch.setenv(key, "test")
        monkeypatch.setenv("EXTRACTION_MODE", "pipeline")
        monkeypatch.setattr(service, "extract_with_confidence", lambda *a, **k: PipelineResult(
            document=invoice(), document_b=None, transcription="",
            confidence={"customer.name": 0.1}, abstained=["customer.name"]))
        monkeypatch.setattr(eval_run, "_api_client", TestClient(app_module.app))
        page = tmp_path / "inv.png"
        Image.new("RGB", (20, 30), "white").save(page)
        ext = eval_run.extract_one(page, "api", None, 0.5, True, "invoice")
        assert isinstance(ext.document, Invoice) and ext.document.customer.name == "Wayne LaPierre"


class TestReporting:
    def _run(self, tmp_path, monkeypatch):
        from eval import run as eval_run
        data = tmp_path / "data"
        for sub in ("docs", "ground_truth", "sets"):
            (data / sub).mkdir(parents=True)
        for doc_id in ("cord_0001", "cord_0002", "cord_0003"):
            Image.new("RGB", (10, 10)).save(data / "docs" / f"{doc_id}.png")
            (data / "ground_truth" / f"{doc_id}.json").write_text('{"total": "1"}')
        for doc_id in ("inv_0001", "inv_0002", "inv_0003"):
            Image.new("RGB", (10, 10)).save(data / "docs" / f"{doc_id}.png")
            (data / "ground_truth" / f"{doc_id}.json").write_text(invoice().model_dump_json())
        (data / "sets" / "mixed.txt").write_text("cord_0001\ncord_0002\ncord_0003\ninv_0001\ninv_0002\ninv_0003\n")
        for attr, sub in (("DOCS_DIR", "docs"), ("GT_DIR", "ground_truth"), ("SETS_DIR", "sets")):
            monkeypatch.setattr(eval_run, attr, data / sub)
        monkeypatch.setattr(eval_run, "OUT_ROOT", tmp_path / "out")

        def fake(path, *, doc_type="receipt", **_):
            doc = invoice(vendor=CUSTOMER, customer=VENDOR) if doc_type == "invoice" else Receipt(total="1")
            return PipelineResult(document=doc, document_b=None, transcription="")

        monkeypatch.setattr(eval_run, "extract_with_confidence", fake)
        path = eval_run.run("mixed", "pipeline", "types")
        return json.loads(path.read_text())["metrics"]

    def test_invoices_are_never_pooled_into_the_receipt_headline(self, tmp_path, monkeypatch):
        m = self._run(tmp_path, monkeypatch)
        assert m["headline_document_type"] == "receipt"
        assert m["n_docs"] == 3 and m["document_accuracy_rate"] == 1.0      # receipts only
        inv = m["by_document_type"]["invoice"]
        assert inv["n_docs"] == 3 and inv["document_accuracy_rate"] == 0.0
        assert inv["error_taxonomy"]["swapped_field"] > 0                  # role swaps reported

    def test_calibration_excludes_invoices(self, tmp_path, monkeypatch):
        from eval import run as eval_run
        calls = []
        monkeypatch.setattr(eval_run, "_doc_ids", lambda _: ["cord_0001", "inv_0001", "inv_0002"])
        monkeypatch.setattr(eval_run, "extract_one", lambda path, *a, **k: calls.append(path) or (_ for _ in ()).throw(RuntimeError("stop")))
        monkeypatch.setattr(eval_run, "_find_doc", lambda d: Path(d))
        monkeypatch.setattr(eval_run, "_find_gt", lambda d: tmp_path / "gt.json")
        (tmp_path / "gt.json").write_text('{"total": "1"}')
        with pytest.raises(SystemExit):
            eval_run.calibrate(0.9)
        assert calls == [Path("cord_0001")]


class TestSplitStability:
    def test_adding_a_source_never_moves_another_sources_documents(self, tmp_path, monkeypatch):
        """Regression: one shared RNG meant adding invoices reshuffled the WildReceipt split."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        import assign_splits

        def split(ids):
            gt, sets, docs = tmp_path / "gt", tmp_path / "sets", tmp_path / "docs"
            for d in (gt, sets, docs):
                d.mkdir(exist_ok=True)
                for f in d.iterdir():
                    f.unlink()
            for doc_id in ids:
                (gt / f"{doc_id}.json").write_text("{}")
            monkeypatch.setattr(assign_splits, "GT_DIR", gt)
            monkeypatch.setattr(assign_splits, "SETS_DIR", sets)
            monkeypatch.setattr(assign_splits, "DOCS_DIR", docs)
            monkeypatch.setattr("sys.argv", ["assign_splits"])
            assign_splits.main()
            read = lambda name: [l for l in (sets / f"{name}.txt").read_text().splitlines() if l and not l.startswith("#")]  # noqa: E731
            return read("calibration"), read("test"), read("smoke")

        receipts = [f"cord_{i:04d}" for i in range(1, 26)] + [f"wild_{i:04d}" for i in range(1, 26)]
        before = split(receipts)
        after = split(receipts + [f"inv_{i:04d}" for i in range(1, 11)])
        strip = lambda ids: [d for d in ids if not d.startswith("inv_")]  # noqa: E731
        assert [strip(x) for x in after] == [strip(x) for x in before]
