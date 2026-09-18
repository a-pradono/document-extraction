"""The review dashboard (eval/dashboard.py). Run: pytest eval/tests -q

It is the artifact a reviewer actually looks at, so the things that must not break are
the ones that would mislead them: a document filed under the wrong status, an abstained
value shown as if it were asserted, and an arithmetic verdict the page invented itself.
"""

from __future__ import annotations

import json
import re

import pytest
from PIL import Image

from api.schemas import from_result
from eval import dashboard
from extractor.confidence import FieldSignals
from extractor.pipeline import PipelineResult
from extractor.schema import Invoice, Receipt

CLEAN = {"merchant_name": "Kopi Kenangan", "date": "2024-03-12", "currency": "IDR",
         "subtotal": "48000", "tax": "5280", "total": "53280",
         "line_items": [{"description": "Es Kopi Susu", "quantity": "2",
                         "unit_price": "18000", "line_total": "36000"},
                        {"description": "Croissant", "quantity": "1",
                         "unit_price": "12000", "line_total": "12000"}]}


def payload(document=None, *, abstained=(), confidence=None, signals=None,
            doc_type="receipt", doc_id="cord_0007"):
    result = PipelineResult(
        document=document if document is not None else Receipt(**CLEAN),
        document_b=None, transcription="KOPI KENANGAN", confidence=confidence or {},
        abstained=list(abstained), signals=signals or {})
    return json.loads(from_result(doc_id, result, 0.6, doc_type).model_dump_json())


def row(html: str, label: str) -> str:
    match = re.search(rf"<tr[^>]*>(?:(?!</tr>).)*?<td class=fl>{label}</td>.*?</tr>",
                      html, re.S)
    assert match, f"no row labelled {label}"
    return match.group(0)


class TestStatus:
    def test_a_failed_check_beats_an_abstention(self):
        """A document with both must file under `failed`: the arithmetic failure is the
        finding, and hiding it behind the softer label would bury it."""
        broken = dict(CLEAN, total="53780")
        p = payload(Receipt(**broken), abstained=["merchant_name"])
        assert dashboard._status_of(p) == "failed"

    def test_abstention_alone_is_abstained(self):
        assert dashboard._status_of(payload(abstained=["merchant_name"])) == "abstained"

    def test_everything_passing_is_clean(self):
        assert dashboard._status_of(payload()) == "clean"

    def test_an_errored_document_is_failed(self):
        assert dashboard._status_of({"status": "error", "document": None}) == "failed"

    def test_a_check_that_did_not_apply_is_not_a_failure(self):
        """null means "not checked" - counting it as a failure would punish a receipt
        for not printing a subtotal."""
        p = payload(Receipt(total="10"))
        assert None in p["validation"].values()
        assert dashboard._status_of(p) == "clean"


class TestRows:
    def test_an_abstained_field_keeps_its_value_and_is_marked(self):
        p = payload(abstained=["merchant_name"], confidence={"merchant_name": 0.2})
        cell = row(dashboard.card.__wrapped__(p, None) if False else _fields(p), "Merchant")
        assert "Kopi Kenangan" in cell and "abstained" in cell and "st-abstained" in cell

    def test_an_absent_field_is_a_dash_not_blank(self):
        cell = row(_fields(payload()), "Payment method")
        assert dashboard.DASH in cell and "absent" in cell

    def test_invoice_parties_are_one_row_each(self):
        inv = Invoice(invoice_number="1", total="10",
                      vendor={"name": "Correll Law Group", "address": "250 Park Avenue"},
                      customer={"name": "Wayne LaPierre"})
        rows = dashboard._rows(payload(inv, doc_type="invoice", doc_id="inv_0008"))
        vendor = next(r for r in rows if r["label"] == "Vendor")
        assert "Correll Law Group" in vendor["html"] and "250 Park Avenue" in vendor["html"]
        assert not any(r["label"].startswith("vendor.") for r in rows)

    def test_a_party_is_abstained_when_any_of_its_fields_is(self):
        inv = Invoice(invoice_number="1", total="10", vendor={"name": "Acme"},
                      customer={"name": "Wayne LaPierre"})
        rows = dashboard._rows(payload(inv, doc_type="invoice", doc_id="inv_0008",
                                       abstained=["customer.name"]))
        customer = next(r for r in rows if r["label"] == "Customer")
        assert customer["status"] == "abstained"

    def test_party_confidence_is_the_weakest_of_its_fields(self):
        inv = Invoice(invoice_number="1", total="10",
                      vendor={"name": "Acme", "address": "1 Road"}, customer={"name": "B"})
        rows = dashboard._rows(payload(inv, doc_type="invoice", doc_id="inv_0001",
                                       confidence={"vendor.name": 0.9, "vendor.address": 0.4}))
        assert next(r for r in rows if r["label"] == "Vendor")["conf"] == 0.4


class TestArithmeticAndChips:
    def test_a_passing_sum_says_it_matches(self):
        assert "matches subtotal" in dashboard._line_items(payload())

    def test_a_failing_sum_shows_the_payloads_discrepancy(self):
        p = payload()
        p["validation"]["line_items_sum_to_subtotal"] = False
        p["validation_notes"]["line_items_sum_to_subtotal"] = "rows sum to 1 != 2 - off by 1"
        html = dashboard._line_items(p)
        assert "off by 1" in html and "sum-bad" in html and "matches subtotal" not in html

    def test_inapplicable_checks_are_counted_not_shown_as_passing(self):
        chips = dashboard._chips(payload(Receipt(total="10")))
        assert "did not apply" in chips
        assert "ch-ok" not in chips.split("did not apply")[0] or "line items" not in chips


class TestPage:
    def _html(self, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        for doc_id in ("cord_0001", "inv_0001"):
            Image.new("RGB", (40, 60), "white").save(docs / f"{doc_id}.png")
        monkeypatch.setattr(dashboard.eval_run, "DOCS_DIR", docs)
        monkeypatch.setattr(dashboard.eval_run, "_load_model", lambda: (None, True))

        def fake(path, source, model, threshold, use_cache, doc_type="receipt"):
            doc = (Invoice(invoice_number="1", total="10", vendor={"name": "A"},
                           customer={"name": "B"}) if doc_type == "invoice"
                   else Receipt(**CLEAN))
            return dashboard.eval_run.Extraction(
                document=doc, latency_s=1.0, transcription="T",
                signals={"total": FieldSignals(1.0, 1.0, 1.0, 0.9)})

        monkeypatch.setattr(dashboard.eval_run, "extract_one", fake)
        return dashboard.build(["cord_0001", "inv_0001"], title="test")

    def test_it_renders_both_types_with_filters_and_counts(self, tmp_path, monkeypatch):
        html = self._html(tmp_path, monkeypatch)
        assert html.count("<article") == 2
        assert 'data-type="receipt"' in html and 'data-type="invoice"' in html
        assert "data-type-filter=invoice" in html and "data-status=failed" in html
        assert 'data-count="2"' in html                       # 2 documents

    def test_it_is_self_contained(self, tmp_path, monkeypatch):
        """Opened over file://, often with no network: every asset must be inline."""
        html = self._html(tmp_path, monkeypatch)
        assert not re.findall(r'(?:src|href)="https?://', html)
        assert "data:image/jpeg;base64," in html

    def test_it_shows_no_ground_truth(self, tmp_path, monkeypatch):
        html = self._html(tmp_path, monkeypatch).lower()
        assert "ground truth" not in html and "expected" not in html

    def test_one_failing_document_does_not_sink_the_page(self, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        Image.new("RGB", (40, 60)).save(docs / "cord_0001.png")
        Image.new("RGB", (40, 60)).save(docs / "cord_0002.png")
        monkeypatch.setattr(dashboard.eval_run, "DOCS_DIR", docs)
        monkeypatch.setattr(dashboard.eval_run, "_load_model", lambda: (None, True))

        def flaky(path, *a, **k):
            if path.stem == "cord_0001":
                raise RuntimeError("RateLimitError: 429")
            return dashboard.eval_run.Extraction(document=Receipt(**CLEAN), latency_s=0.1)

        monkeypatch.setattr(dashboard.eval_run, "extract_one", flaky)
        html = dashboard.build(["cord_0001", "cord_0002"], title="t")
        assert html.count("<article") == 2 and "RateLimitError" in html


class TestThumbnails:
    def test_a_wide_image_is_downscaled(self, tmp_path):
        path = tmp_path / "big.png"
        Image.new("RGB", (4000, 2000), "white").save(path)
        uri, pages = dashboard.thumbnail(path)
        assert uri.startswith("data:image/jpeg;base64,") and pages == 1
        assert len(uri) < 400_000

    def test_a_pdf_reports_its_page_count(self):
        pytest.importorskip("pypdfium2")
        from pathlib import Path
        pdfs = sorted(Path("data/docs").glob("inv_*.pdf"))
        if not pdfs:
            pytest.skip("no invoices fetched")
        uri, pages = dashboard.thumbnail(pdfs[0])
        assert uri.startswith("data:image/jpeg;base64,") and pages >= 1


def _fields(p) -> str:
    """The field table alone, for row-level assertions."""
    return "".join(
        f'<tr class=st-{r["status"]}><td class=fl>{r["label"]}</td>'
        f'<td class=fv>{r["html"]}</td><td class=fc></td>'
        f'<td class=fs>{"" if r["status"] == "ok" else r["status"]}</td></tr>'
        for r in dashboard._rows(p))


class TestConfidenceBands:
    """High >= 0.80, medium 0.50-0.79, low < 0.50, on the mean per-field score."""

    def test_the_boundaries_fall_the_way_the_labels_read(self):
        assert dashboard._confidence_of({"confidence": {"a": 0.80}})[1] == "high"
        assert dashboard._confidence_of({"confidence": {"a": 0.799}})[1] == "medium"
        assert dashboard._confidence_of({"confidence": {"a": 0.50}})[1] == "medium"
        assert dashboard._confidence_of({"confidence": {"a": 0.499}})[1] == "low"

    def test_the_band_uses_the_mean_the_badge_shows(self):
        mean, band = dashboard._confidence_of({"confidence": {"a": 0.9, "b": 0.5}})
        assert mean == pytest.approx(0.7) and band == "medium"

    def test_a_document_with_no_scores_has_no_band(self):
        """A failed extraction must not land in 'low' - it was never scored at all."""
        assert dashboard._confidence_of({"status": "error"}) == (None, "none")

    def test_the_card_carries_the_band_and_the_toolbar_offers_it(self, tmp_path, monkeypatch):
        docs = tmp_path / "docs"
        docs.mkdir()
        Image.new("RGB", (40, 60)).save(docs / "cord_0001.png")
        monkeypatch.setattr(dashboard.eval_run, "DOCS_DIR", docs)
        monkeypatch.setattr(dashboard.eval_run, "_load_model", lambda: (None, True))
        monkeypatch.setattr(dashboard.eval_run, "extract_one", lambda *a, **k:
                            dashboard.eval_run.Extraction(
                                document=Receipt(**CLEAN), latency_s=0.1,
                                confidence={"total": 0.95, "tax": 0.91}))
        html = dashboard.build(["cord_0001"], title="t")
        assert 'data-conf="high"' in html
        assert "data-conf=low" in html and "data-conf=medium" in html
        assert "conf-high" in html                       # badge coloured by band


class TestSingleModelDocuments:
    """A rate-limited Model B leaves a card that looks complete: three signals still
    score, and the missing `agreement` reads as a neutral 0.50. It has to be labelled."""

    def _payload(self, stages):
        from extractor.cost import StageUsage
        result = PipelineResult(document=Receipt(**CLEAN), document_b=None,
                                transcription="T", confidence={"total": 0.9},
                                stages=[StageUsage(stage=s, model="m") for s in stages])
        return json.loads(from_result("cord_0001", result, 0.5).model_dump_json())

    def test_both_models_is_not_flagged(self):
        assert dashboard._single_model(self._payload(["model_a", "model_b"])) is False

    def test_model_a_alone_is_flagged(self):
        assert dashboard._single_model(self._payload(["model_a"])) is True

    def test_a_document_that_never_ran_is_not_flagged_as_single_model(self):
        """It failed outright; calling that 'one model' would misdescribe it."""
        assert dashboard._single_model({"status": "error", "document": None}) is False

    def test_the_card_says_so_in_words(self, tmp_path, monkeypatch):
        from extractor.cost import StageUsage

        docs = tmp_path / "docs"
        docs.mkdir()
        Image.new("RGB", (40, 60)).save(docs / "cord_0001.png")
        monkeypatch.setattr(dashboard.eval_run, "DOCS_DIR", docs)
        monkeypatch.setattr(dashboard.eval_run, "_load_model", lambda: (None, True))
        monkeypatch.setattr(dashboard.eval_run, "extract_one", lambda *a, **k:
                            dashboard.eval_run.Extraction(
                                document=Receipt(**CLEAN), latency_s=0.1,
                                confidence={"total": 0.9},
                                stages=[StageUsage(stage="model_a", model="vlm")]))
        html = dashboard.build(["cord_0001"], title="t")
        assert "one model only" in html and "rests on three signals" in html
