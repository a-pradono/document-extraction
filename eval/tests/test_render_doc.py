"""The per-document card (docs/OUTPUT.md section 3). Run: pytest eval/tests -q

The card is what a human looks at after uploading one document, so the things that must
never break are the ones a reader would misread: absent vs abstained, and an arithmetic
check that quietly claims to pass.
"""

from __future__ import annotations

import json
import re

from api.schemas import from_result
from eval.render_doc import main, render
from extractor.confidence import FieldSignals
from extractor.pipeline import PipelineResult
from extractor.schema import Invoice, Receipt

CLEAN = {"merchant_name": "Kopi Kenangan", "subtotal": "48000", "tax": "5280",
         "total": "53280", "currency": "IDR",
         "line_items": [{"description": "Es Kopi Susu", "quantity": "2",
                         "unit_price": "18000", "line_total": "36000"},
                        {"description": "Croissant", "quantity": "1",
                         "unit_price": "12000", "line_total": "12000"}]}


def payload(document=None, *, abstained=(), confidence=None, signals=None,
            transcription="KOPI KENANGAN\nJl. Sudirman", threshold=0.6, doc_type="receipt"):
    result = PipelineResult(
        document=document if document is not None else Receipt(**CLEAN),
        document_b=None, transcription=transcription,
        confidence=confidence or {}, abstained=list(abstained), signals=signals or {})
    return json.loads(from_result("cord_0007", result, threshold, doc_type).model_dump_json())


def row(html: str, field: str) -> str:
    match = re.search(rf"<tr[^>]*><td>{re.escape(field)}</td>.*?</tr>", html, re.S)
    assert match, f"no row for {field}"
    return match.group(0)


class TestNullVersusAbstained:
    def test_a_null_field_renders_as_a_dash_and_is_not_marked_abstained(self):
        """A blank cell reads as a rendering bug; 'abstained' on an absent field reads as
        a system that declined to answer when there was nothing to answer."""
        html = render(payload())
        cell = row(html, "payment_method")
        assert "&mdash;" in cell and "abstained" not in cell and "absent" in cell

    def test_an_abstained_field_keeps_its_value_and_its_styling(self):
        html = render(payload(abstained=["merchant_name"], confidence={"merchant_name": 0.2}))
        cell = row(html, "merchant_name")
        assert "abstained" in cell and "Kopi Kenangan" in cell and "class=abst" in cell

    def test_the_two_statuses_are_visually_distinct(self):
        html = render(payload(abstained=["merchant_name"]))
        assert "class=abst" in row(html, "merchant_name")
        assert "class=absent" in row(html, "payment_method")

    def test_an_abstained_table_is_shown_for_review_but_marked_not_asserted(self):
        html = render(payload(abstained=["line_items"]))
        assert "Es Kopi Susu" in html and "withheld" in html

    def test_signals_are_exposed_on_the_row(self):
        html = render(payload(signals={"total": FieldSignals(1.0, 1.0, 1.0, 0.95)}))
        assert "agreement: 1.00" in row(html, "total")


class TestArithmeticFooter:
    def test_a_passing_check_says_it_matches(self):
        html = render(payload())
        assert "matches subtotal" in html and "48,000.00" in html

    def test_a_failing_check_shows_the_discrepancy_not_a_silent_pass(self):
        broken = dict(CLEAN, subtotal="47500", total="52780")
        html = render(payload(Receipt(**broken)))
        assert "off by 500" in html
        assert "matches subtotal" not in html

    def test_the_verdict_is_the_payloads_not_recomputed(self):
        """The renderer must never form its own opinion: if it did, the page and the API
        could disagree about whether a document reconciled."""
        p = payload()
        p["validation"]["line_items_sum_to_subtotal"] = False
        p["validation_notes"]["line_items_sum_to_subtotal"] = "rows sum to 1 != 2 - off by 1"
        html = render(p)
        assert "off by 1" in html and "matches subtotal" not in html

    def test_an_unavailable_check_is_not_rendered_as_a_pass(self):
        html = render(payload(Receipt(line_items=CLEAN["line_items"])))
        assert "no subtotal to check against" in html
        assert "date_parses n/a" in html or "n/a" in html


class TestValidationPanel:
    def test_every_check_becomes_a_chip(self):
        html = render(payload())
        for name in ("subtotal_plus_tax_equals_total", "line_items_sum_to_subtotal",
                     "rows_multiply_out", "currency_is_iso"):
            assert f"{name} " in html

    def test_invoice_role_check_appears_and_fails_when_both_parties_match(self):
        inv = Invoice(invoice_number="1", vendor={"name": "Acme"}, customer={"name": "acme!"},
                      total="10")
        html = render(payload(inv, doc_type="invoice"))
        assert "vendor_differs_from_customer" in html
        assert "role resolution failed" in html


class TestPageLevel:
    def test_the_transcription_is_present_but_collapsed(self):
        html = render(payload())
        assert "<details>" in html and "Jl. Sudirman" in html

    def test_no_external_assets(self):
        """It is opened over file://, often with no network."""
        html = render(payload())
        assert "http://" not in html and "https://" not in html

    def test_an_errored_payload_with_no_fields_renders(self):
        p = {"doc_id": "bad_0001", "status": "error", "document": None,
             "error": "TruncatedResponse: stopped at length", "warnings": ["retry failed"]}
        html = render(p)
        assert "TruncatedResponse" in html and "retry failed" in html

    def test_ground_truth_is_not_shown(self):
        html = render(payload())
        assert "ground truth" not in html.lower() and "expected" not in html.lower()

    def test_cli_writes_a_file(self, tmp_path, capsys):
        src = tmp_path / "extraction.json"
        src.write_text(json.dumps(payload()), encoding="utf-8")
        main([str(src)])
        out = tmp_path / "extraction.html"
        assert out.exists() and "Fields" in out.read_text(encoding="utf-8")
        assert "wrote" in capsys.readouterr().out


class TestRunWiring:
    def test_a_run_writes_one_card_per_document_and_links_them(self, tmp_path, monkeypatch):
        from PIL import Image

        from eval import run as eval_run
        data = tmp_path / "data"
        for sub in ("docs", "ground_truth", "sets"):
            (data / sub).mkdir(parents=True)
        for doc_id in ("cord_0001", "cord_0002"):
            Image.new("RGB", (10, 10)).save(data / "docs" / f"{doc_id}.png")
            (data / "ground_truth" / f"{doc_id}.json").write_text(json.dumps(CLEAN))
        (data / "sets" / "s.txt").write_text("cord_0001\ncord_0002\n")
        for attr, sub in (("DOCS_DIR", "docs"), ("GT_DIR", "ground_truth"), ("SETS_DIR", "sets")):
            monkeypatch.setattr(eval_run, attr, data / sub)
        monkeypatch.setattr(eval_run, "OUT_ROOT", tmp_path / "out")
        monkeypatch.setattr(eval_run, "extract_with_confidence", lambda path, **k: PipelineResult(
            document=Receipt(**CLEAN), document_b=None, transcription="KOPI"))

        path = eval_run.run("s", "pipeline", "cards")
        cards = sorted((path.parent / "documents").glob("*.html"))
        assert [c.stem for c in cards] == ["cord_0001", "cord_0002"]
        # the payload is kept beside the card, so it can be re-rendered without re-running
        payloads = sorted(p.stem for p in (path.parent / "documents").glob("*.json"))
        assert payloads == ["cord_0001", "cord_0002"]
        scorecard = (path.parent / "scorecard.html").read_text(encoding="utf-8")
        assert 'href="documents/cord_0001.html"' in scorecard
