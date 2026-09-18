"""The verification review server (scripts/review_server.py). Run: pytest eval/tests -q

This tool decides what a human is shown before they certify a label, so the failure that
matters is showing them the wrong thing: a document out of order, a disputed sum left
off the screen, or a verdict recorded against the wrong id.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import review_server  # noqa: E402


@pytest.fixture()
def app(tmp_path, monkeypatch):
    gt, docs = tmp_path / "gt", tmp_path / "docs"
    gt.mkdir(), docs.mkdir()

    labels = {
        # rows sum to 87.30, but the subtotal says 97.30: wild_0019's real defect
        "wild_0019": {"merchant_name": "The Blue Ginger", "subtotal": "97.30",
                      "tax": "6.72", "total": "102.75",
                      "line_items": [{"description": "Fish Head", "line_total": "87.30"}]},
        "cord_0001": {"merchant_name": None, "subtotal": "10", "tax": "1", "total": "11",
                      "line_items": [{"description": "Kopi", "line_total": "10"}]},
    }
    for doc_id, label in labels.items():
        (gt / f"{doc_id}.json").write_text(json.dumps(label), encoding="utf-8")
        Image.new("RGB", (60, 90), "white").save(docs / f"{doc_id}.png")
    prov = {
        "wild_0019": {"verified": False, "hand_labeled_fields": ["merchant_name"],
                      "label_notes": "currency printed only as '$'. CHECK."},
        "cord_0001": {"verified": False, "hand_labeled_fields": ["merchant_name"]},
    }
    (gt / "_provenance.json").write_text(json.dumps(prov), encoding="utf-8")

    monkeypatch.setattr(review_server, "GT_DIR", gt)
    monkeypatch.setattr(review_server, "DOCS_DIR", docs)
    import gt_common
    import verify_gt
    monkeypatch.setattr(gt_common, "GT_DIR", gt)
    monkeypatch.setattr(gt_common, "PROVENANCE", gt / "_provenance.json")
    monkeypatch.setattr(verify_gt, "GT_DIR", gt)
    monkeypatch.setattr(verify_gt, "PROVENANCE", gt / "_provenance.json", raising=False)
    monkeypatch.setattr(review_server, "audit_all", lambda: [
        {"doc_id": "wild_0019", "flags": [
            {"field": "total", "severity": "error",
             "message": "subtotal 97.30 + tax 6.72 = 104.02, label says 102.75"},
            {"field": "line_items", "severity": "todo", "message": "ignore me"}]}])
    return TestClient(review_server.build_app(only_unverified=True)), gt


class TestWhatTheReviewerSees:
    def test_the_disputed_arithmetic_is_on_the_screen(self, app):
        client, _ = app
        docs = client.get("/api/docs").json()["documents"]
        wild = next(d for d in docs if d["doc_id"] == "wild_0019")
        assert wild["disputes"][0]["message"].startswith("subtotal 97.30 + tax 6.72")

    def test_todo_flags_are_not_shown_as_disputes(self, app):
        """'needs a hand label' is not a claim that the arithmetic is wrong."""
        client, _ = app
        wild = next(d for d in client.get("/api/docs").json()["documents"]
                    if d["doc_id"] == "wild_0019")
        assert all(f["severity"] != "todo" for f in wild["disputes"])

    def test_the_row_sum_is_computed_so_the_gap_is_visible(self, app):
        client, _ = app
        wild = next(d for d in client.get("/api/docs").json()["documents"]
                    if d["doc_id"] == "wild_0019")
        assert wild["rows_sum"] == "87.30"          # against a labelled subtotal of 97.30

    def test_the_drafters_check_note_travels_with_the_document(self, app):
        client, _ = app
        wild = next(d for d in client.get("/api/docs").json()["documents"]
                    if d["doc_id"] == "wild_0019")
        assert wild["check"] is True and "CHECK" in wild["note"]

    def test_check_documents_come_first(self, app):
        client, _ = app
        ids = [d["doc_id"] for d in client.get("/api/docs").json()["documents"]]
        assert ids[0] == "wild_0019"

    def test_an_absent_field_is_marked_absent_not_omitted(self, app):
        client, _ = app
        cord = next(d for d in client.get("/api/docs").json()["documents"]
                    if d["doc_id"] == "cord_0001")
        merchant = next(f for f in cord["fields"] if f["field"] == "merchant_name")
        assert merchant["value"] is None and merchant["drafted"] is True


class TestVerdicts:
    def test_a_verdict_is_written_to_provenance(self, app):
        client, gt = app
        r = client.post("/api/verdict", json={"doc_id": "wild_0019", "verified": True,
                                             "note": "checked against the image"})
        assert r.status_code == 200
        prov = json.loads((gt / "_provenance.json").read_text(encoding="utf-8"))
        assert prov["wild_0019"]["verified"] is True
        assert prov["wild_0019"]["verified_note"] == "checked against the image"

    def test_needs_fixing_is_recorded_as_unverified(self, app):
        client, gt = app
        client.post("/api/verdict", json={"doc_id": "wild_0019", "verified": False,
                                          "note": "subtotal should be 87.30"})
        prov = json.loads((gt / "_provenance.json").read_text(encoding="utf-8"))
        assert prov["wild_0019"]["verified"] is False
        assert "87.30" in prov["wild_0019"]["verified_note"]

    def test_an_unknown_document_is_refused(self, app):
        client, _ = app
        assert client.post("/api/verdict",
                           json={"doc_id": "nope_0001", "verified": True}).status_code == 404

    def test_the_queue_only_holds_unverified_documents(self, app):
        client, _ = app
        client.post("/api/verdict", json={"doc_id": "wild_0019", "verified": True})
        ids = [d["doc_id"] for d in client.get("/api/docs").json()["documents"]]
        assert "wild_0019" not in ids and "cord_0001" in ids


class TestImages:
    def test_the_image_is_served_as_jpeg(self, app):
        client, _ = app
        r = client.get("/api/image/cord_0001")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"

    def test_a_missing_file_is_a_404_not_a_crash(self, app):
        client, _ = app
        assert client.get("/api/image/nope_0001").status_code == 404
