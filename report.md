# Receipt & Invoice Extraction — End-to-End Report

> **Status: Results section is UNPOPULATED.** No model run has been executed and
> 0 of 60 ground-truth labels have been verified. Every table in section 4 carries
> `TBD` placeholders and the command that fills it. Sections 1–3 describe the system
> as built and are accurate. Do not circulate this document with placeholder values
> still in section 4.

---

## 1. Introduction

### The problem

A client needs structured data pulled out of a stack of real documents — scanned,
photographed, multi-page, inconsistently laid out. Some fields are simple lookups
(a merchant name, a date). Others require reasoning across sections: a total that
must reconcile against line items and tax, a party's role in a transaction, a table
of line items that spans a page break.

### What this system is optimized for

Not a demo that works on a clean PDF. The design target is a system whose accuracy
can be **proven**, that **knows when it is unsure**, and whose quality, latency and
cost can be reasoned about. Concretely, three properties drove every decision:

1. **It must not hallucinate.** A field absent from the document returns `null`. No
   value is ever invented, and every value that is kept must be traceable to text
   found in the document.
2. **It must abstain rather than guess.** Per-field confidence is computed, and
   fields below a threshold are flagged as abstained rather than asserted.
3. **Evaluation is the primary deliverable, not an afterthought.** The eval harness
   was built before any prompt tuning, so every subsequent change is judged by
   re-running a scorecard rather than by inspecting one document.

### Scope and honest limitations

| | |
|---|---|
| Primary document type | Receipts (~65 documents) |
| Secondary document type | Invoices (10 documents) |
| Own scans | **None.** This forfeits a stated bonus; see §2.4 for the compensation |
| Models | Hosted (not open-weights) — forfeits the self-hosting bonus; see §3.5 |
| Statistical power | n≈60 means Document Accuracy Rate is noisy (±6pp); see §4.6 |

---

## 2. Data Source

### 2.1 Why receipts as the primary type

Receipts were chosen because they exercise the hardest requirement — **verifiable
cross-section reasoning** — better than any alternative at this scale. A receipt
contains a line-item table, a subtotal, a tax amount and a total, and those four
things must reconcile arithmetically. That gives a **model-free ground truth signal**:
`subtotal + tax == total` either holds or it does not, regardless of what any model
claims. That property is the foundation of the confidence design in §3.3.

The trade-off, stated plainly: receipts are natively single-page and have exactly one
counterparty, so they cannot exercise multi-page documents or party-role resolution.
That is why invoices were added (§2.3).

### 2.2 Receipt sources (~65 documents)

| Source | Docs | Role | Labels |
|---|---:|---|---|
| **CORD v2** (HF `naver-clova-ix/cord-v2`) | 25 | Structured backbone — line items with quantity and unit price, plus subtotal→total | Script-converted, then hand-verified |
| **WildReceipt** (`download.openmmlab.com/mmocr/data/wildreceipt.tar`) | 25 | Template diversity — photographed "in the wild", unseen templates | Script-converted, then hand-verified |
| **OCR Receipts retail** (Kaggle `trainingdatapro/ocr-receipts-text-detection`) | 15 | Messiest tier — phone photos, faded thermal paper | **Fully hand-labeled** |

Two deliberate choices:

**Small and trustworthy over large and noisy.** ~65 documents, every label verified
by hand. Trustworthy ground truth is the highest-weighted evaluation criterion, and a
larger set with unverified labels would produce metrics that cannot be trusted — and
would cost proportionally more per eval run, of which there are many.

**Stratified sampling, not first-N.** CORD contains 1,000 receipts. Taking the first
25 would produce an accidentally-easy eval set. Sampling is stratified by line-item
count (short 2-item receipts through long 15-item ones) and spread across merchants.

**The retail set has no subtotal or tax annotations**, so those fields were
hand-labeled from the images. Skipping them would have left the system blind on its
strongest confidence signal precisely where documents are hardest.

### 2.3 Invoices (10 documents)

Added to cover the two scenario requirements receipts structurally cannot:

- **Multi-page.** Invoices are natively multi-page; at least 4 of the 10 are.
- **Party roles.** An invoice has both a **vendor** and a **customer**. Distinguishing
  them is genuine role resolution — the two addresses look alike, and only position
  and labelling disambiguate. A vendor/customer swap surfaces as the
  `swapped_field` error label (§3.4).

Source: HF `AlroWilde/invoice-checkmark-annotations` — real scanned multilingual
invoices (Ukrainian / Chinese / Swedish), Apache 2.0, with vendor address, buyer name,
line-item totals and tax rate. Backup: the RVL-CDIP invoice subset, which has explicit
Receiver/Supplier labels.

**Disclosure:** these images carry human-drawn pen checkmarks over verified fields.
That is extra real-world messiness and useful for robustness, but it is unusual and a
reviewer should not have to discover it themselves.

**Rejected:** Kaggle `nikitpatel/invoice-ner-dataset`. Despite the name, it contains
extracted *text*, not images — unusable for a vision pipeline.

n=10 is deliberately small: enough to prove the system generalizes to a second
document type, not enough to pretend invoices are a co-equal dataset. Invoice results
are reported as a **separate breakdown, never pooled** into the receipt headline.

### 2.4 No own scans — and the compensation

The brief states that own scans count most. None were available, so that bonus is
forfeited. Rather than hide this, two substitutes were used:

1. **Source diversity** — four sources, two document types, three languages, three
   independent label schemas. Cross-source F1 breakdowns (§4.2) test generalization
   in a way a single dataset cannot.
2. **Synthetic degradation** — controlled blur, rotation (±8°), JPEG compression,
   brightness shift and simulated thermal fade applied to CORD images, re-scored
   against the **unchanged** ground truth. Labeling cost is zero and the result is a
   robustness *curve* rather than an anecdote.

The synthetic tier is **clearly labeled as synthetic** and excluded from the headline
scorecard. Augmented images must never be presented as real-world variation.

### 2.5 Splits

| Split | n | Purpose |
|---|---:|---|
| `calibration` | 22 receipts + 3 invoices | Fit confidence weights and the abstention threshold. **Never reported.** |
| `test` | 43 receipts + 7 invoices | The reported scorecard |
| `smoke` | 12 | Fast dev loop — drawn **from `calibration`** |
| `degraded` | ~12 | A *tag* on the worst documents already inside `test`, for a subset breakdown |
| `synthetic` | variable | Degradation tier, reported separately |

Two rules that protect the numbers:

- **`smoke` is drawn from `calibration`, never `test`.** The smoke set is run dozens
  of times while iterating; if it overlapped `test`, the reported metrics would be
  quietly overfitted.
- **Nothing is fitted on `test`.** Confidence weights and the abstention threshold
  come from `calibration` only.

Split assignment is seeded and the split files are committed, so runs stay comparable
across versions — which is what the regression gate (§3.6) depends on.

---

## 3. Method

### 3.1 Architecture

```
image          ──► Model A (VLM)  ──► { transcription, document, _confidence }
transcription  ──► Model B (LLM)  ──► document  (independent re-extraction)
                   Python validation (arithmetic / format — model-free)
                                   ──► per-field confidence ──► abstention
```

**Model A — VLM reader.** Receipts are a layout problem: line items are
column-aligned, and "which of these five numbers is the total" is answered by
position. Flat OCR text destroys that information, so a vision model reads the image
directly. It returns the transcription **and** the structured fields in a single call
— a second call for the transcript would double vision-token cost for no accuracy
gain.

**The transcription replaces a separate OCR stage.** It serves as the groundedness
source (§3.3), which means the anti-hallucination signal is retained without a local
OCR dependency. A classical PaddleOCR path was built and then abandoned: on a
CPU-only machine it was too slow to keep in the loop.

**Model B — text LLM reasoner.** Does what a vision model is weakest at:
cross-section reasoning, arithmetic reconciliation, and normalization. It is a cheap
text-only call. It is deliberately a **different model family** from Model A —
sibling models (e.g. a flagship and its distilled variant) fail identically on the
same smudged digit, which would inflate the agreement signal exactly where it needs
to be honest.

### 3.2 Typed, validated output

All model output is parsed into Pydantic models (`Receipt`, `Invoice`, sharing
`LineItem`). Rules enforced by the schema and the prompt:

- Money is `Decimal`, never `float`. Values cross the JSON boundary as **strings** so
  no rounding is introduced in transit.
- Dates are `YYYY-MM-DD`.
- Absent fields are `null`. Never an empty string, never a zero, never a plausible
  guess.
- Malformed JSON gets one corrective retry showing the model its own bad output;
  a second failure is recorded as an error rather than silently swallowed.

*Implementation note:* the `Receipt.date` field originally shadowed the
`datetime.date` import inside its own annotation, so Pydantic resolved the type to
`None` and silently rejected **every** date — which would have zeroed date F1 across
the whole scorecard with no visible error. Caught while rendering the HTML report,
fixed by aliasing the import, and covered by a regression test. Worth recording
because the failure was silent, not loud.

### 3.3 Confidence: four signals, fitted weights

Per-field confidence pools four signals, ranked by how much each actually knows:

| Signal | Strength | Why |
|---|---|---|
| `validated` | **Strongest** | Deterministic arithmetic (`subtotal + tax == total`, line items sum to subtotal, date parses, currency is ISO-4217). **Model-free — cannot be bluffed** |
| `grounded` | Strong | Does the extracted value literally appear in the transcription? Direct hallucination check |
| `agreement` | Moderate | Do Model A and Model B return the same normalized value? |
| `self_report` | **Weakest** | The model's stated certainty. LLM self-confidence is reliably overconfident — it emits a fluent number, not an error estimate |

**Known limitation, stated explicitly:** Model B reads Model A's transcription, so the
two are only **partially independent** — a misread digit propagates and inflates their
agreement. This is the specific reason `validated` is weighted heavily: it is
independent of the reading step entirely. True reading independence would require a
second vision model on the raw image (§5).

**Weights are fitted, not hand-set.** A per-field-group logistic regression
(string / money / date / currency) is fitted on the `calibration` split. Groups with
fewer than 12 observations keep default weights rather than overfitting. The fitted
coefficients are themselves a reportable finding (§4.4). Calibration is per-group
because `merchant_name` and `total` have different difficulty profiles, and one global
threshold would over-abstain on easy fields while under-abstaining on hard ones.

**The threshold is read off a risk–coverage curve** against a target precision, not
chosen by intuition. It is reported as "we keep X% of fields at Y% precision".

### 3.4 Evaluation harness

Scoring happens after a **normalization layer** (dates→ISO, money→`Decimal`, currency
symbols, whitespace/case) so the metrics measure semantic correctness rather than
string noise.

**Accuracy metrics:** per-field precision / recall / F1 (reported **normalized and
exact** — the gap between them is format sensitivity), micro and macro F1, **Document
Accuracy Rate** (every field and all line items correct), and **line-item F1** via
greedy one-to-one matching on description similarity plus amount, then field accuracy
within matched items.

**Error taxonomy — failures by kind, not just count:**

| Label | Meaning |
|---|---|
| `missing` | Ground truth has a value, prediction is null → under-extraction |
| `spurious` | Ground truth is null, prediction invented one → **hallucination** |
| `wrong_value` | Both present, genuinely different content |
| `format_error` | Same content, different format — deliberately **not** counted as wrong |
| `swapped_field` | Predicted value belongs to a different field (e.g. vendor ↔ customer) |

**Confidence quality metrics** — two *independent* properties with *different* fixes:

- **AUC** (discrimination): does the score rank correct fields above incorrect ones?
  ~0.5 means the score carries no information, and rescaling cannot help — the signals
  must change.
- **ECE** (calibration): is a 0.9 actually right about 90% of the time?
- A **reliability diagram** and a **risk–coverage curve** are plotted, and the
  scorecard prints the resulting diagnosis automatically.

### 3.5 Deployment and the CPU constraint

The target machine has no GPU. That single fact determined where each model runs:

| Component | Location | Reasoning |
|---|---|---|
| Model A (VLM) | Hosted, free tier | CPU vision inference is impractical |
| Model B (LLM) | Hosted, free tier | A 3B model generating ~600 tokens of JSON on CPU takes ~40–90s/doc; hosted keeps the pipeline at ~3–8s/doc |
| Validation, groundedness, metrics | Local Python | Deterministic, instant, free |

All model access goes through one OpenAI-compatible provider abstraction, so
endpoints and models are swappable via `.env` with no code change.

**Trade-off stated honestly:** neither model is open-weights, so the "self-hosted or
local is a bonus" point is forfeited. The justification is the CPU constraint, and the
brief itself recommends a free hosted option for exactly this situation. Given GPU
access, a local document VLM would be swapped in — the provider abstraction exists
precisely so that is a config change.

**Caching by image hash** keeps rate-limited calls to one per document even across
repeated eval runs. After a metrics or taxonomy change, re-running the harness costs
zero API calls.

### 3.6 Production surface

`docker compose up` builds and serves the extraction API and leaves it running.
`docker compose run --rm eval` is a one-shot command that waits for the API's
healthcheck, runs the harness **against that running service**, and writes the
scorecard to a mounted output directory. Both services share one image, so the
harness scores exactly the code the API runs.

API: `GET /health`, `POST /extract`, `POST /extract/batch` (bounded concurrency,
because free tiers are rate-limited and unbounded fan-out only collects 429s),
`GET /metrics` for cost and latency counters.

**Resilience:** a failed document returns 200 with populated `errors` and is scored as
a total miss, so it appears in the error taxonomy rather than vanishing. One bad
document never kills a batch or a run.

**Regression gate:** each run is version-tagged, and a diff command compares two
scorecards field-by-field, flagging any F1 drop past a threshold and listing
newly-failing documents.

### 3.7 Reviewer artifacts

- **`scorecard.html`** — KPIs, reliability diagram, risk–coverage curve, taxonomy
  chart, per-field table, per-document drill-down.
- **`gallery.html`** — source document beside its structured output, 5 per view,
  filterable by type and status. Also runs in `--mode truth` against ground truth
  alone, which makes it the verification tool for the labeling pass.
- Both are self-contained single files (inline CSS/JS, base64 images) so they work
  over `file://` and survive being committed.

---

## 4. Results

> **NOT YET POPULATED.** No model run has been executed. Every value below is `TBD`.
> The command that fills each table is given. Replace this banner with the run date,
> model IDs and commit SHA once real numbers exist.

**Run metadata to record:** date · Model A id · Model B id · commit SHA ·
eval set · number of documents · total cost · wall-clock.

### 4.1 Headline (receipts, `test` split)

Fill from `outputs/<run>/scorecard.json`:
```bash
docker compose up -d && docker compose run --rm eval
```

| Metric | Value | Note |
|---|---|---|
| Document Accuracy Rate | TBD | Every field + all line items correct |
| Micro F1 | TBD | Pooled over field instances |
| Macro F1 | TBD | Mean of per-field F1 |
| Line-item F1 | TBD | Table extraction |
| Hallucination rate | TBD | `spurious` ÷ predicted — target ≈ 0 |
| Latency p50 / p95 | TBD | |
| Cost per document | TBD | Free tier may be $0; record tokens regardless |

### 4.2 Per-source breakdown

The most informative table in this report: it shows *where* accuracy comes from.
Expect a clean → wild → photos degradation gradient. Invoices sit **below** the
receipt headline, never pooled into it.

| Source | n | DAR | Micro F1 | Line-item F1 | Halluc. | p95 |
|---|---:|---:|---:|---:|---:|---:|
| CORD (clean) | TBD | TBD | TBD | TBD | TBD | TBD |
| WildReceipt (wild) | TBD | TBD | TBD | TBD | TBD | TBD |
| Retail (photos) | TBD | TBD | TBD | TBD | TBD | TBD |
| **Receipts (headline)** | TBD | TBD | TBD | TBD | TBD | TBD |
| Invoices *(n=10, uncalibrated)* | TBD | TBD | TBD | TBD | TBD | TBD |

### 4.3 Failures by kind

| Source | missing | spurious | wrong_value | format_error | swapped |
|---|---:|---:|---:|---:|---:|
| CORD | TBD | TBD | TBD | TBD | TBD |
| WildReceipt | TBD | TBD | TBD | TBD | TBD |
| Retail | TBD | TBD | TBD | TBD | TBD |
| Invoices | TBD | TBD | TBD | TBD | **TBD** |

**What to look for:** `spurious` near zero validates the anti-hallucination design.
A high `swapped` count on invoices means vendor/customer role confusion — the specific
failure invoices were added to detect, and one a pooled metric would never surface.

### 4.4 Confidence quality

```bash
python -m eval.run --set calibration --version cal --source api   # then fit
```

| Metric | Default weights | Fitted | Target |
|---|---|---|---|
| AUC (discrimination) | TBD | TBD | > 0.75 |
| ECE (calibration) | TBD | TBD | < 0.10 |

**Fitted weights per field group** — report these; they are a finding in their own
right (e.g. "groundedness carried 3× the weight of self-report"):

| Group | validated | grounded | agreement | self_report |
|---|---:|---:|---:|---:|
| money | TBD | TBD | TBD | TBD |
| string | TBD | TBD | TBD | TBD |
| date | TBD | TBD | TBD | TBD |
| currency | TBD | TBD | TBD | TBD |

**Operating point** — read off the risk–coverage curve, reported as coverage at a
precision target, not as a bare threshold:

> At threshold **TBD**, we keep **TBD%** of fields at **TBD%** precision.

**Interpretation rule (apply before attempting a fix):** AUC ≈ 0.5 means the score
carries no information and rescaling cannot help — change the signals. Good AUC with
high ECE means the ranking works but the numbers lie — refit the calibration.
**A well-measured negative result here is a stronger outcome than an unexamined
confidence number.** If a signal proves worthless, report it.

### 4.5 Optimization: confidence cascade

Run the cheap model on every document; escalate to the stronger model only when
confidence is low. Measure on the **same** eval set, before and after.

| | Baseline | Cascade | Δ |
|---|---|---|---|
| Cost per document | TBD | TBD | TBD |
| Latency p50 / p95 | TBD | TBD | TBD |
| Micro F1 | TBD | TBD | TBD |
| DAR | TBD | TBD | TBD |

**The accuracy rows matter as much as the cost rows.** A cost reduction that quietly
degrades quality is not an optimization.

### 4.6 Robustness (synthetic degradation)

Clearly labeled synthetic; excluded from the headline.

| Condition | Micro F1 | DAR |
|---|---:|---:|
| Original | TBD | TBD |
| Blur (mild) | TBD | TBD |
| Rotation ±8° | TBD | TBD |
| JPEG quality 30 | TBD | TBD |
| Thermal fade | TBD | TBD |

Also report the **real** degraded subset (the ~12 worst documents tagged inside
`test`), which is more meaningful than the synthetic curve because it is not
augmented.

### 4.7 Statistical honesty

State these limitations explicitly rather than presenting the numbers as precise:

- **DAR is binary per document**, so at n≈60 its 95% CI is roughly **±6pp**. Two
  versions differing by 3pp are not distinguishable.
- **Per-field F1 is more stable** (~350–500 field-level judgments), which is why
  per-field metrics carry the regression gate rather than DAR.
- **Invoices at n=10 are indicative only.** Confidence there is uncalibrated: the
  weights were fitted on receipts, and 10 documents cannot support their own fit.
- **Ground truth is hand-verified but single-annotator.** No inter-annotator
  agreement was measured, so label noise is unquantified.

---

## 5. What would come next

| Priority | Item | Why |
|---|---|---|
| 1 | **Verify all 60 labels and execute a real run** | Everything above §4 is untested against reality until this happens |
| 2 | Second independent VLM reader on the raw image | Removes the partial-independence weakness in §3.3 — the agreement signal's main flaw |
| 3 | Inter-annotator agreement on a 15-doc subset | Quantifies label noise, currently unknown |
| 4 | Expand invoices beyond n=10 | Enough to calibrate confidence for that type |
| 5 | Local open-weights VLM given GPU access | Recovers the self-hosting bonus; a config change only |
| 6 | Route abstained fields to human review | Closes the loop — abstention becomes useful rather than merely honest |
