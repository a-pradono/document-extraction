# Dataset acquisition 

Build `scripts/fetch_datasets.sh` and the converters described below. Read this whole
file first; the **Non-negotiables** at the bottom override any convenience shortcut.

Target: **~65 hand-verified documents** across three sources. Small and trustworthy
beats large and noisy — "ground truth you can trust" is the top-weighted grading
criterion. Do NOT scale past ~65.

| Source | Docs | Role | Auth needed |
|---|---:|---|---|
| CORD (v2) | 25 | Structured backbone: line items, subtotal→total arithmetic | No (HF) |
| WildReceipt | 25 | Template diversity, photographed "in the wild" | No (direct) |
| OCR Receipts (retail) | 15 | Messiest tier, photographed | Yes (Kaggle) |

---

## 1. CORD — easiest path is Hugging Face, not Kaggle

Prefer the HF mirror: no auth, and `datasets` handles extraction.

- **HF (preferred):** dataset id `naver-clova-ix/cord-v2`
- **Kaggle (alternative):** https://www.kaggle.com/datasets/lonelvino/cord-1000

```python
from datasets import load_dataset
ds = load_dataset("naver-clova-ix/cord-v2", split="test")   # 100 docs in test split
```

Each row has an `image` plus a `ground_truth` JSON string containing `gt_parse` with
`menu` (line items: `nm`, `cnt`, `price`), `sub_total`, and `total`. Indonesian
receipts, so currency is IDR — set `OCR_LANG=id` if you ever revive OCR.

**Sampling:** do NOT take the first 25. Stratify by line-item count so you get short
2-item receipts AND long 15-item ones; otherwise you build an accidentally-easy eval
set and the metrics flatter you. Also spread across distinct merchants.

## 2. WildReceipt — direct download, no auth

```bash
wget -nv -c https://download.openmmlab.com/mmocr/data/wildreceipt.tar
mkdir -p data/raw && tar -xf wildreceipt.tar --directory data/raw
rm -f wildreceipt.tar
```

Extracts to `wildreceipt/` containing `class_list.txt`, `dict.txt`, `image_files/`,
`train.txt`, `test.txt`. 1,740 images, ~69k text boxes, 25 categories (12 keys +
12 values + "Ignore"). Annotation lines are JSON:

```json
{"file_name": "a.jpeg", "height": 348, "width": 348,
 "annotations": [{"box": [...], "text": "CHOEUN", "label": 1}]}
```

`class_list.txt` maps label ints to names (e.g. `1 Store_name_value`). You want the
`*_value` classes: `Store_name_value`, `Date_value`, `Total_value`, `Tax_value`,
`Tips_value`, `Prod_item_value`, `Prod_quantity_value`, `Prod_price_value`.
Read the mapping from the file — don't hardcode integers.

**Sample 25 from `test.txt`**, stratified by number of product-item annotations.

## 3. OCR Receipts (retail) — Kaggle, needs auth

- **URL:** https://www.kaggle.com/datasets/trainingdatapro/ocr-receipts-text-detection

Kaggle requires an API token, so this one **cannot be fully automated**. Have the
script check for credentials and print clear manual instructions if absent:

```bash
# Requires ~/.kaggle/kaggle.json (Kaggle > Settings > Create New API Token)
pip install kaggle
kaggle datasets download -d trainingdatapro/ocr-receipts-text-detection -p data/raw --unzip
```

If `~/.kaggle/kaggle.json` is missing, print the URL and the token instructions and
exit non-zero. **Never** prompt for or store Kaggle credentials in the repo.

**Important gap:** this dataset annotates roughly item / store / date_time / total —
it has **no subtotal and no tax**. Those must be **hand-labeled from the images** for
all 15 docs. Without them you cannot score the fields your `subtotal + tax == total`
validation depends on, so you'd be blind on the strongest confidence signal exactly
where documents are hardest. Do not skip this.

---

## Converters: `scripts/convert_<source>.py`

One per source, all emitting the SAME target format: `data/ground_truth/<id>.json`
validating against `extractor.schema.Receipt`. Copy the chosen images to
`data/docs/<id>.<ext>` using matching ids. Prefix ids by source so per-source
breakdowns are trivial later: `cord_0001`, `wild_0001`, `retail_0001`.

**Throw the bounding boxes away.** Our VLM takes the raw image and returns fields;
the schema has no positional fields and the metrics score normalized values, not
spatial overlap. Boxes are useful in the converter for exactly ONE thing: grouping
description / quantity / price into the same line item by y-coordinate proximity.
Use them transiently for row grouping, then discard. No coordinates reach the schema,
the extractor, or the metrics.

Ground-truth rules (see `data/README.md`):
- Money as **strings** (`"53280"`) so no float rounding sneaks in — they parse to `Decimal`.
- Dates as `YYYY-MM-DD`.
- Absent fields → `null`. **Never invent a value the receipt doesn't show**, and never
  adjust a value to make arithmetic balance. If a receipt's printed total genuinely
  doesn't equal subtotal+tax, record what's printed — those documents are valuable
  test cases for the validation signal.
- Emit a `data/ground_truth/_provenance.json` recording, per id: source, original
  filename, and whether each field was script-converted or hand-labeled. Reviewers
  care that ground truth is trustworthy; provenance is how you show it.

**Verification is mandatory, not optional.** After conversion, build
`scripts/verify_gt.py` that renders each image alongside its converted labels for
human review. Script-converted CORD/WildReceipt labels still need eyeballing (~2
min/doc); retail needs full hand-labeling (~10 min/doc). Track verification status in
`_provenance.json`.

---

## Splits: `scripts/assign_splits.py` → `data/sets/*.txt`

| Split | n | CORD | Wild | Retail | Use |
|---|---:|---:|---:|---:|---|
| `calibration` | 22 | 9 | 9 | 4 | Fit confidence weights + threshold. **Never reported** |
| `test` | 43 | 16 | 16 | 11 | The reported scorecard |
| `smoke` | 12 | — | — | — | Fast dev loop — drawn FROM `calibration` |

Two rules that matter for rigor:

1. **`smoke` is drawn from `calibration`, never from `test`.** You'll run smoke dozens
   of times while iterating; if it overlapped `test` you'd quietly overfit the numbers
   you publish.
2. **Nothing fitted on `test`.** The confidence model's weights and the abstention
   threshold are fitted on `calibration` only.

Use a fixed random seed and commit the split files so runs stay comparable across
versions (that's what the Phase 7 regression diff depends on).

Also write `data/sets/degraded.txt`: tag the ~12 worst documents (faded, angled,
crumpled) that are already **inside `test`** — a tag for a per-subset breakdown, not a
separate held-out set. "F1 drops from 0.91 to 0.68 on degraded scans" is a strong,
honest finding.

## Optional: `scripts/degrade.py` (synthetic robustness tier)

We have no own scans, so compensate with controlled degradation of CORD images:
blur, rotation (±8°), JPEG compression, brightness/contrast shift, simulated thermal
fade. Re-run extraction against the **unchanged** ground truth — labeling cost is
zero and you get a robustness curve instead of an anecdote.

Write outputs to `data/docs/synthetic/` and a `data/sets/synthetic.txt`. **Label
these clearly as synthetic** in the README and keep them out of the headline
scorecard — augmented images must never be presented as real-world variation.

---

## Non-negotiables

- **Don't commit large raw datasets.** `data/raw/` goes in `.gitignore`. Commit ONLY
  the ~65 curated images in `data/docs/` plus their ground truth, so the eval runs
  out of the box (the test requires this).
- **Never commit Kaggle credentials** or any API token. `~/.kaggle/kaggle.json` stays
  outside the repo.
- **Respect dataset licenses.** Record each source's license and citation in
  `data/README.md`. CORD and WildReceipt are research datasets — cite the papers
  (WildReceipt: Sun et al., "Spatial Dual-Modality Graph Reasoning for Key
  Information Extraction", arXiv:2103.14470).
- **`fetch_datasets.sh` must be idempotent** — safe to re-run, skips what exists.
- Verify checksums/counts after download and fail loudly rather than proceeding with
  a partial dataset.

## Commit sequence for this work

```
data: add dataset fetch script (CORD via HF, WildReceipt direct, Kaggle manual)
data: add CORD -> Receipt schema converter with stratified sampling
data: add WildReceipt converter (value classes, boxes used only for row grouping)
data: add retail converter + hand-label subtotal/tax (dataset omits them)
data: add split assignment (calibration/test/smoke, seeded) + provenance
data: verify all 65 ground-truth files by hand; record status in _provenance.json
```

---

# Addendum: 10 invoices (second document type)

## Why add invoices at all

Receipts are strong on the scenario's *computed total* and *line items in a table*,
but two scenario requirements they cannot exercise:

- **Multi-page.** Receipts are natively single-page. Invoices are natively multi-page.
- **"A party's role."** Receipts have one merchant. Invoices have a **vendor/supplier**
  AND a **customer/buyer** — distinguishing them IS role resolution, and it is a real
  cross-section reasoning task (the two addresses look alike; only position and
  labelling disambiguate).

10 documents is deliberately small: enough to prove the system generalizes to a second
document type and to exercise multi-page + role resolution, not enough to pretend
invoices are a co-equal dataset. Report invoice metrics as a **separate breakdown**,
never pooled into the headline receipt scorecard.

## Source selection — read this before picking

> **Verified 2026-09-17 — the source below marked "Preferred" was REJECTED on
> inspection, and RVL-CDIP-N (multi-page) was chosen instead.** Evidence:
>
> - `AlroWilde/invoice-checkmark-annotations` is **synthetic**, not real scans: every
>   image in all three languages is the same HTML template (identical layout and footer
>   text), rendered at exactly 800 px wide, with Faker-style data
>   (`annoying1918@gmail.com`) and digitally overlaid pen marks. It is **single-page
>   only**, the **vendor has no name** (address only), and its own numbers are
>   inconsistent (36 × 36.50 printed as 1,314.17; a due date before the issue date).
>   It is shipped as raw zips, so `load_dataset` does not yield image rows.
> - `Voxel51/high-quality-invoice-images-for-ocr`: its card says synthetic.
> - DocILE: real, but access requires a request form and approval.
> - **Chosen: `jordyvl/rvl_cdip_n_mp`** (RVL-CDIP-N, Larson et al., NeurIPS 2022
>   Datasets & Benchmarks; CC BY-NC 4.0). Real documents gathered from the web (many
>   are public-records/FOIA releases), kept as original PDFs. Its invoice class has 57
>   documents, 16 of them multi-page. 10 chosen, all 2-5 pages, each naming a vendor
>   and a customer: see `scripts/fetch_invoices.py` for the pinned list and checksums.
>   They contain real names of private individuals: the repository must stay private.
>
> The requirements in the rest of this addendum stand; only the source changed.

**Kaggle is a poor source here.** Checked and rejected:

- `nikitpatel/invoice-ner-dataset` — **DO NOT USE.** Despite the name, it contains
  extracted *text* from PDFs, not images. Our pipeline is a VLM reading pixels, so a
  text-only dataset is unusable. This is an easy mistake to make from the title alone.
- `daneshwarisavadkar/invoice-pdf` — exists, but annotation quality unverified. Only
  consider if the options below fail.

**Decision rule:** at n=10 we hand-label regardless (~10 min/doc, ~2h total), so
optimize for **image realism** and **presence of both party roles**, NOT for
convertible annotations. Convertible labels are why CORD/WildReceipt carry the
receipt bulk; they are irrelevant at this volume.

### Preferred: `AlroWilde/invoice-checkmark-annotations` (Hugging Face)

~600 real scanned/photographed invoices, ~200 each Ukrainian / Chinese / Swedish,
Apache 2.0. Annotations include `Vendor Business Address`, `Customer/Buyer Name`,
`Unique Invoice Identifier`, `Invoice Table Row N: Line Total Amount`, and
`Applied Tax Percentage` — i.e. both party roles, line items, and tax.

```python
from datasets import load_dataset
ds = load_dataset("AlroWilde/invoice-checkmark-annotations")
```

**Known quirk, disclose it in the README:** the images carry human-drawn pen
checkmarks/circles over verified fields. This is extra real-world messiness (good for
robustness testing) but it is unusual, and a reviewer should not be left to discover
it themselves. Sample across all three languages for layout diversity.

### Backup: RVL-CDIP invoice subset

520 invoice images with OCR files and XML ground-truth boxes. Fields: **Receiver**,
**Supplier**, Invoice_info, Total — the cleanest explicit party-role labels available.
Use if the HF dataset is unavailable.

### Also acceptable: inv-cdip

350 annotated invoices from the Tobacco Collections of the Industry Documents Library,
with genuine scanning artifacts (stamps, handwritten annotations, variable print
quality) and 7 fields including purchase order, due date, and total tax.
**Caveat:** a published benchmark found annotation inconsistencies in this set (the
same field worded differently across samples). A good reminder that public labels
need verification — which is why all 10 invoices get hand-verified regardless of
source.

## Schema: extend, don't fork

Add an `Invoice` model in `extractor/schema.py` that REUSES `LineItem`. Do not
duplicate the line-item logic — the eval harness's line-item matching must work
unchanged for both document types.

```python
class Party(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    tax_id: Optional[str] = None

class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    issue_date: Optional[_Date] = None
    due_date: Optional[_Date] = None
    vendor: Party = Field(default_factory=Party)     # role resolution
    customer: Party = Field(default_factory=Party)   # role resolution
    currency: Optional[str] = None
    subtotal: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    total: Optional[Decimal] = None
    purchase_order: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
```

Note the `_Date` alias — writing `date: Optional[date]` makes the field name shadow
`datetime.date` in the resolved annotation and Pydantic then silently rejects every
date. That bug already bit us once in `Receipt`; there is a regression test for it.

**`vendor` vs `customer` is the graded part.** A swapped vendor/customer is exactly
the `swapped_field` taxonomy label, so the existing detector will catch it — make sure
`_apply_swapped` in `eval/metrics.py` compares nested party fields too.

## Validation signals for invoices

Same model-free arithmetic as receipts (`subtotal + tax == total`, line items sum to
subtotal), plus two invoice-specific checks that cost nothing:

- `due_date >= issue_date` (a violation is a real extraction error)
- `vendor.name != customer.name` (identical parties means role resolution failed)

## Ground truth + splits

- ids prefixed `inv_0001` … `inv_0010`, in `data/ground_truth/` like everything else.
- Record in `_provenance.json`: source, original filename, language, page count, and
  that all fields were **hand-labeled** (not script-converted).
- Splits: **3 to `calibration`, 7 to `test`.** Too few to fit invoice-specific
  confidence weights, so they reuse the receipt-fitted model — **say so in the README**
  and treat invoice confidence numbers as provisional rather than calibrated.
- Add `data/sets/invoices.txt` so the scorecard can report a per-type breakdown.
- Prefer genuinely multi-page invoices for at least 4 of the 10; that is the whole
  point of adding them.

## Honest framing for the README

> We chose receipts as the primary document type because their line-item tables and
> subtotal/tax/total arithmetic give verifiable cross-section reasoning and
> model-free validation. Receipts cannot exercise multi-page documents or party-role
> resolution, so we added 10 real invoices as a secondary type to cover both. Invoice
> results are reported as a separate breakdown (n=10, hand-labeled, confidence
> uncalibrated for this type) and are indicative, not statistically robust.

## Commit sequence

```
data: add invoice source (HF invoice-checkmark-annotations), reject Kaggle NER set (text-only)
schema: add Invoice + Party models reusing LineItem
eval: extend swapped_field detection to nested party fields
data: hand-label 10 invoices, 4 multi-page; record provenance
eval: add per-document-type breakdown to scorecard
```