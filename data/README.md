# data/

Everything the eval harness needs to run out of the box.

```
data/
  docs/                 the receipt files:  <id>.jpg | .png | .pdf
    synthetic/          degraded copies (scripts/degrade.py) — clearly labeled, own tier
  ground_truth/         hand-verified labels: <id>.json  (matches a Receipt)
    _provenance.json    per doc: source, original filename, script vs hand labeled, verified
    synthetic/          labels for the synthetic tier (identical to their originals)
  hand_labels/          retail.json — fields no dataset annotates, filled from the images
  sets/                 tier manifests: calibration.txt, test.txt, smoke.txt,
                        degraded.txt, synthetic.txt (one <id> per line, # comments ok)
  raw/                  downloaded datasets — gitignored, never committed
```

## Sources

| Prefix | Source | Docs | Role |
|---|---|---:|---|
| `cord_` | CORD v2 (`naver-clova-ix/cord-v2`, test split) | 25 | Structured backbone: line items, subtotal→total arithmetic. Indonesian, IDR |
| `wild_` | WildReceipt (openmmlab), test split | 25 | Template diversity, photographed in the wild. Mostly US/English |
| `retail_` | Kaggle `trainingdatapro/ocr-receipts-text-detection` | 15 | Messiest tier, photographed |
| `inv_` | RVL-CDIP-N multipage (`jordyvl/rvl_cdip_n_mp`), invoice class | 10 | Second document type: multi-page PDFs (1–5 pages), vendor/customer roles. Labels match `Invoice` |

Three independent sources means two languages and three different label schemas, which
is the closest substitute we have for owning our own scans — see the README's honest
note about that forfeited bonus.

### Licenses and citation

- **CORD v2** — CC BY 4.0. Park et al., *CORD: A Consolidated Receipt Dataset for
  Post-OCR Parsing*, Document Intelligence Workshop at NeurIPS 2019.
- **WildReceipt** — released for research with the MMOCR toolkit (Apache-2.0).
  Sun et al., *Spatial Dual-Modality Graph Reasoning for Key Information Extraction*,
  arXiv:2103.14470.
- **RVL-CDIP-N multipage** — CC BY-NC 4.0 (non-commercial). Larson et al.,
  *Evaluating Out-of-Distribution Performance on Document Image Classifiers*, NeurIPS
  2022 Datasets and Benchmarks. Fetch with `python scripts/fetch_invoices.py`: it pulls
  only the ten pinned PDFs, verifies them by SHA-256, and never overwrites labels. These
  are real public records that name private individuals, so **keep the repo private**.
- **OCR Receipts text detection** — Kaggle, TrainingData.pro. Requires an API token;
  the download is manual and credentials stay outside the repo. Check the dataset page
  for its current license terms before redistributing the images.

All three are used here for research evaluation only.

## Rebuilding this from scratch

```bash
./scripts/fetch_datasets.sh          # cord + wildreceipt automatic; retail prints manual steps
python scripts/convert_cord.py       # -> cord_0001..0025
python scripts/convert_wildreceipt.py
python scripts/convert_retail.py     # writes a hand-label stub; re-run after filling it
python scripts/verify_gt.py --html   # review every label against its image
python scripts/assign_splits.py      # seeded -> sets/*.txt
python scripts/degrade.py            # optional synthetic robustness tier
```

`fetch_datasets.sh` is idempotent — re-running skips whatever already exists.

## Bounding boxes

The source datasets ship box coordinates; the converters use them for exactly one
thing — grouping description / quantity / price into the same line item by vertical
proximity — and then throw them away. Our VLM reads the raw image and the metrics score
normalized values, so no coordinate reaches the schema, the extractor, or the metrics.

## Ground-truth format

One JSON file per document, named by the same `<id>` as its file in `docs/`. It must
validate against `extractor.schema.Receipt`. See `ground_truth_FORMAT_EXAMPLE.json` —
it lives outside `ground_truth/` on purpose, since it has no image and would otherwise
be swept into a run as a document that can never be extracted.

Rules that keep ground truth trustworthy:
- **Money as strings** (`"53280"`) so no float rounding sneaks in — they parse to Decimal.
- **Dates as `YYYY-MM-DD`.**
- **Absent fields → `null`** (or omit them). Never invent a value the receipt doesn't show.
- **Never adjust a value to make the arithmetic balance.** If a receipt's printed total
  genuinely isn't subtotal + tax, record what's printed — those documents are the most
  valuable test cases for the validation signal.
- **Every label is hand-verified.** The converters are a first pass, not the ground
  truth. `scripts/verify_gt.py` renders each image beside its labels and records the
  outcome in `_provenance.json`; a document is not trustworthy until `verified` is true.

## Provenance

`ground_truth/_provenance.json` records, per document: the source dataset, the original
filename, which fields were script-converted, which were hand-labeled, which are still
awaiting a human (`needs_hand_label`), and whether the document has been verified. The
distinction matters because a null meaning "not on the receipt" and a null meaning
"nobody looked yet" score identically but say very different things about the labels.

The retail tier has no subtotal and no tax in its source annotations. Those are
hand-labeled from the images via `hand_labels/retail.json` — without them the
`subtotal + tax == total` check, the strongest confidence signal, is blind on exactly
the hardest documents.

## Splits

| Split | n | CORD | Wild | Retail | Use |
|---|---:|---:|---:|---:|---|
| `calibration` | 22 | 9 | 9 | 4 | Fits confidence weights + threshold. **Never reported** |
| `test` | 43 | 16 | 16 | 11 | The reported scorecard |
| `invoices` | 10 | — | — | — | All invoices: 3 in `calibration`, 7 in `test`, reported under their own type |
| `smoke` | 12 | — | — | — | Fast dev loop — drawn **from** `calibration` |
| `degraded` | ~12 | — | — | — | A tag on the worst documents already **inside** `test` |
| `synthetic` | opt | — | — | — | Augmented copies. Own tier, never the headline number |

**Current state: 50 of the 65 documents exist** (CORD 25 + WildReceipt 25). The retail
tier is missing because it needs a Kaggle API token, so `assign_splits.py` rescaled the
quotas proportionally and produced calibration 18 / test 32. Re-run it after
`convert_retail.py` to get the full 22 / 43 split. The invoices add 3 / 7 on top, drawn
with their own RNG so adding them never moved a receipt between splits. `smoke` and
`degraded` stay receipts-only.

`smoke` comes from `calibration` because it runs dozens of times during iteration —
overlapping `test` would quietly overfit the published numbers. Nothing is fitted on
`test`. The seed is fixed and `sets/*.txt` is committed so runs stay comparable across
versions, which is what the Phase 7 regression diff depends on.

DAR is binary per document, so it's noisy at this size (~±6% at n≈65) — say so in the
top-level README rather than reporting it as a precise figure.
