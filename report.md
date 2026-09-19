# Document Extraction — End-to-End Report

---

## 1. Introduction

### The problem

A client needs structured data pulled out of a stack of real documents — scanned,
photographed, multi-page, inconsistently laid out. Some fields are simple lookups
(a merchant name, a date). Others require reasoning across sections: a total that must
reconcile against line items and tax, a party's role in a transaction, a table of line
items that spans a page break.

### What this system is optimized for

Not a demo that works on a clean PDF. The design target is a system whose accuracy can
be **proven**, that **knows when it is unsure**, and whose quality, latency and cost can
be reasoned about. Three properties drove every decision:

1. **It must not hallucinate.** A field absent from the document returns `null`. No
   value is invented, and every value kept is traceable to text found in the document.
2. **It must abstain rather than guess.** Per-field confidence is computed and fields
   below a fitted threshold are withheld at the API boundary rather than asserted.
3. **Evaluation is the primary deliverable.** The harness was built before any prompt
   tuning, so every change is judged by re-running a scorecard. It paid off twice over:
   §4.3 lists four defects it caught in its own ground truth and in itself, and §4.5
   records the one it caught in the system — a 7.65% hallucination rate traced to a
   single inferred field, fixed, and re-measured at 0.63%.

### Scope and honest limitations

| | |
|---|---|
| Primary document type | Receipts — **50 documents** |
| Secondary document type | Invoices — **10 documents**, real, 1–5 pages |
| Own scans | None — three public sources instead, plus a synthetic degradation tier (§2.5) |
| Models | Hosted: Gemini 3.6 Flash (VLM) + GPT-OSS 120B on Groq (LLM); see §3.5 |
| Ground truth | **60 of 60 human-verified**, five label errors corrected |
| Reported accuracy | Micro F1 **0.918**, hallucination **0.63%**, abstention keeping **87.8%** of fields at **95.8%** precision (§4.5) |

---

## 2. Data

60 documents, three sources, two document types, four languages
(Indonesian, English, Spanish, French/Dutch), three independent label schemas.

### 2.1 Why receipts as the primary type

Receipts exercise the hardest requirement — **verifiable cross-section reasoning** —
better than any alternative at this scale. A receipt carries a line-item table, a
subtotal, a tax amount and a total, and those must reconcile. That gives a **model-free
ground-truth signal**: `subtotal + service − discount + tax == total` either holds or it
does not, whatever any model claims. That property is the foundation of the confidence
design in §3.3.

The trade-off, stated plainly: receipts are natively single-page with one counterparty,
so they cannot exercise multi-page reading or party-role resolution. Hence invoices.

### 2.2 Receipt sources (50 documents)

| Source | Docs | Role | Labels |
|---|---:|---|---|
| **CORD v2** (`naver-clova-ix/cord-v2`, CC BY 4.0) | 25 | Structured backbone — line items with quantity and unit price, subtotal → total | Script-converted from the dataset, header fields hand-drafted from the images |
| **WildReceipt** (MMOCR, Apache-2.0) | 25 | Template diversity — photographed in the wild, unseen templates | Script-converted, re-spaced and extended by hand |

**Small and trustworthy over large and noisy.** Trustworthy ground truth is the
highest-weighted evaluation criterion, and a larger set with unverified labels produces
metrics that cannot be trusted — while costing proportionally more per eval run.

**Stratified sampling, not first-N.** CORD holds 1,000 receipts; taking the first 25
would produce an accidentally easy set. Sampling is stratified by line-item count and
spread across merchants, with the chosen ids pinned so a converter fix re-converts the
*same* documents rather than reshuffling the splits.

**CORD is anonymised at source.** The dataset blurs the merchant header on nearly every
receipt, so most CORD merchant, address and date labels are explicit `null` — which is
itself a useful annotation: it turns "did the model invent a merchant?" into a scorable
question. Four headers remain legible through the blur (PIZZA HUT, Solaria, XXI Cafe,
and a Korean logo) and are labelled per the rule *label what a human can read*.

### 2.3 Invoices (10 documents)

Added to cover the two requirements receipts structurally cannot: **multi-page reading**
and **party-role resolution** (vendor vs customer, where a swap surfaces as the
`swapped_field` error label).

**Source: RVL-CDIP-N multipage** (`jordyvl/rvl_cdip_n_mp`, invoice class; Larson et al.,
NeurIPS 2022 Datasets & Benchmarks; **CC BY-NC 4.0, non-commercial**). Ten real
invoices, 1–5 pages each, obtained through public-records requests: a police department,
two TV stations, a school district, law firms, a state agency, small manufacturers.
`scripts/fetch_invoices.py` streams the archive and extracts only the ten pinned files,
each verified by SHA-256.

**Real documents bring real mess**, all recorded in `_provenance.json`:

- `inv_0004` prints an arithmetic error — rows sum to 56.56 against a printed 36.59.
  Kept as-is: a document whose printed total is genuinely wrong is a valuable test case.
- `inv_0005` has a row struck through by hand with "duplicate invoicing" written beside
  it, while the printed total still includes it. Labelled as printed.
- `inv_0006` is three documents in one PDF: invoice 29069 (page 1), a different invoice
  28815 (page 2) and credit memo 3094 (page 3). The label covers invoice 29069 only.
  **Kept deliberately.** The pipeline sends every page in one call and expects one
  document back, so the model may merge rows or totals across the three and be marked
  wrong for it - the failure is in the one-document-per-file premise, not in the reading.
  It sits in `calibration`, which fits on receipts only, so it affects no reported
  number; splitting a stack into N documents is scoped as a separate feature in
  `docs/OUTPUT.md` §2, not a change to `/extract`.
- `inv_0002` is the sharpest role trap in the set: the customer is the billing agency,
  not the advertiser whose campaign it is.
- `inv_0010` runs 225 line items across 5 pages; the transcribed rows sum exactly to the
  printed $59,403.38.

n=10 is deliberately small: enough to prove the system generalises to a second document
type, not enough to pretend invoices are a co-equal dataset. Invoice results are
reported as a **separate breakdown, never pooled** into the receipt headline, and the
confidence model is fitted on receipts only — invoice confidence is **uncalibrated**.

**Privacy and licence.** These are real public records naming real people and
organisations, under a non-commercial licence. The repository is private; published
views carry the dataset attribution.

### 2.4 Verification — what the human pass actually found

**60 of 60 documents verified.** Every label was checked against its image, and the pass
found **five money-field errors**, all in WildReceipt, none of which the model-free audit
could resolve on its own. The audit could say *this does not reconcile*; only a human
could say what the right number was.

| Document | Defect | After correction |
|---|---|---|
| `wild_0003` | Subtotal `675.00` for a printed `Rs1,675.00`; one of two printed taxes (VAT 13.13%, Service TAX 4.94%) dropped | 1,675.00 + 82.75 + 213.36 = **1,971.11**, the printed Cash amount |
| `wild_0019` | Subtotal `97.30` for a printed `87.30`; the printed 10% service charge missing entirely | 87.30 + 8.73 + 6.72 = **102.75**, the printed TOTAL |
| `wild_0018` | A row annotated `5390` for a printed `5.90` — **a defect in the published dataset**, not in our conversion | 6 rows sum to exactly **47.30**, the printed SUBTTL |
| `wild_0015` | Three printed coupons (−0.75, −0.25, −1.00) dropped, so rows summed to 5.00 against a printed total of 3.00 | 5.00 − 2.00 = **3.00** |
| `wild_0004` | One fuel purchase split across three junk rows: `REGULAR/15863`, a bare `3479`, a bare `55.19` — gallons, price per gallon and fuel total | 15.863 gal × $3.479 = $55.19, plus $10.00 = **$65.19** |

That is a **10% error rate in the money fields of one source**, and every one of those
errors would have been charged to the model: a correct extraction scored as
`wrong_value` on exactly the fields the confidence design leans on hardest. It is the
empirical case for "small and verified" over "large and noisy", and the reason this
project audits its ground truth rather than assuming it.

**Why verify all 60 rather than sample?** Sampling is the right tool for estimating a
label error rate across a corpus too large to check. It is the wrong tool for an
evaluation set, because this set is not the product — it is the **measuring instrument**
for every number in §4. A sample of 15 would have reported "roughly 10% of WildReceipt
money fields are wrong" and left us not knowing *which* five, so the instrument stays
bent and we merely know that it is.

The bias also does not average out. A wrong label converts a **correct** extraction into
a scored error, so the five defects pushed one way only, on `subtotal`, `tax` and `total`
— the fields the confidence design leans on hardest. For scale: document accuracy has a
noise floor of about ±8pp at n=39, and a plausible improvement between two versions is
3–5pp. The label error was **larger than the effects the harness exists to detect**.

What makes full verification affordable is the deliberate choice of a small set: 60
documents is roughly 40 minutes once, amortised over every run for the life of the
project. At 10,000 documents it would be a month of work, and sampling would be the only
rational option. The policy this project would apply at scale:

| Corpus | Approach |
|---|---|
| Evaluation set, n ≤ a few hundred | Verify 100% — it is the ruler |
| Training corpus, n large | Sample to estimate noise, report metrics with that band, fix systematically |
| Production output | Sample continuously — that is monitoring, not ground truth |

Two honest qualifications. First, the model-free audit can only flag fields it can check:
it says "this does not reconcile", but it has no signal whatsoever on merchant name, date
or currency, where a sample would yield an error rate and nothing more. Second, not every
field needed the full pass — the 10 invoice labels and the mostly-null CORD header fields
would have been well served by an audit-triaged sample. The money fields were not, and
that is where all five errors were.

Judgement calls are recorded per document in `_provenance.json`, including those the
drafter could not resolve and the reviewer settled: ambiguous `$` currencies (AUD vs SGD
vs USD, inferred from tax registrations), day-first vs month-first dates, and
`cord_0019`, where the merchant name is blurred by the dataset's own anonymisation.

### 2.5 Generalisation and robustness tiers

The set is built to test generalisation two ways rather than on one corpus:

1. **Source diversity** — three sources, two document types, four languages, three
   independent label schemas, reported as per-source breakdowns.
2. **Synthetic degradation** — 25 degraded copies (blur, rotation, JPEG, fade) scored
   against the **unchanged** ground truth. Clearly labelled synthetic, excluded from the
   headline scorecard structurally rather than by convention.

The documents are public datasets, not scans made for this project.

### 2.6 Splits (seeded, committed)

| Split | n | CORD | Wild | Invoices | Purpose |
|---|---:|---:|---:|---:|---|
| `calibration` | 21 | 9 | 9 | 3 | Fit confidence weights and the threshold. **Never reported** |
| `test` | 39 | 16 | 16 | 7 | The reported scorecard |
| `smoke` | 12 | 6 | 6 | 0 | Fast dev loop — drawn **from** `calibration` |
| `degraded` | 12 | 6 | 6 | 0 | A *tag* on the worst documents already inside `test` |
| `invoices` | 10 | — | — | 10 | All invoices, for the per-type breakdown |

Two rules protect the numbers: `smoke` comes from `calibration`, never `test`; and
nothing is fitted on `test`. Each source draws from its own seeded RNG, so adding the
invoices moved no receipt between splits — verified by test.

---

## 3. Method

### 3.1 Architecture

```
image          ──► Model A (VLM)  ──► { transcription, document, _confidence }
transcription  ──► Model B (LLM)  ──► document  (independent re-extraction)
                   Python validation (arithmetic / format — model-free)
                                   ──► per-field confidence ──► abstention
```

| Role | Model | Why |
|---|---|---|
| Model A — VLM reader | `gemini-3.6-flash` | Layout is the problem: "which of these five numbers is the total" is answered by position, which flat OCR destroys |
| Model B — LLM reasoner | `openai/gpt-oss-120b` (Groq) | Cross-section reasoning and normalisation, text-only and cheap. Open-weights, and a **different family from Model A on purpose** |
| Cascade first pass | `gemini-3.5-flash-lite` | Cheap read; escalates to the full pipeline on doubt (§4.5) |

#### Why this architecture, and not the three alternatives

| Architecture | Verdict |
|---|---|
| **LLM alone** | Impossible. The input is an image; a text model cannot read it without something converting pixels to text first. This is not a trade-off, it is a missing step. |
| **OCR → LLM** | A classical OCR pass flattens the page into a string, which destroys the cue that answers "which of these five numbers is the total": position. Column alignment, the totals block below a rule, an add-on line indented under its item — all gone before the LLM sees anything. It also puts an irreversible error stage first: a digit OCR misreads cannot be recovered downstream, and the LLM reasons confidently over it. On a CPU-only machine a local OCR engine is also too slow to keep in an eval loop. |
| **VLM alone** | **Genuinely competitive, and better on two axes.** Measured on the same 39 documents (§4.8): identical micro F1 (0.918), identical hallucination rate, 18.7 pp *better* document accuracy, at half the cost and 2.5× lower latency. If the requirement were extraction accuracy alone, this is the architecture the data would choose. |
| **VLM + LLM** (chosen) | Costs 2.1× for **no accuracy gain**. What it buys is confidence: the second model's independent re-read produces the `agreement` signal, which carries the largest fitted weight for strings (+1.50) and dates (+1.10) — the fields where no arithmetic applies and grounding is weak, because a transcription contains the text either way. |

**The second model is a confidence mechanism, not an accuracy mechanism**, and the
ablation puts a number on it (`python -m eval.ablation`, cached, no API calls):

| Confidence model | AUC | ECE | Keeps, at 95% precision |
|---|--:|--:|--:|
| Four signals (VLM + LLM) | **0.908** | 0.032 | **86.4%** |
| Three signals (VLM alone, `agreement` neutral) | 0.852 | 0.046 | 75.8% |

Losing the second model costs **10.6 points of coverage at the same precision target** —
one field in ten that the system could have asserted now goes to a human instead. That is
the entire return, and whether it is worth 2.1× cost is a deployment decision rather than
a technical one:

- **Abstention matters** (fields feed a ledger, a human reviews what is withheld): the
  second model pays for itself by cutting review volume by a tenth at equal precision.
- **Abstention does not matter** (a human checks everything anyway, or nothing): use the
  VLM alone. It is cheaper, faster, and better per document.

This report recommends the two-model path because the brief's requirement is a system
that *knows when it is unsure*. Under a different requirement the honest recommendation
would be different, and the harness is what makes that statement testable rather than
rhetorical.

**Different families, deliberately.** Sibling models fail identically on the same
smudged digit, which would inflate the agreement signal exactly where it must be honest.

**The transcription replaces a separate OCR stage** and serves as the groundedness
source, so the anti-hallucination signal exists without a local OCR dependency and
without a second vision call.

**Multi-page** documents are rendered page by page and sent in **one** call, so the
model sees the whole document; `MAX_PAGES` (5) refuses anything longer rather than
silently reading part of it.

### 3.2 Typed, validated output

Model output is parsed into Pydantic models (`Receipt`, `Invoice`, sharing `LineItem`).
Money is `Decimal`, never `float`, and crosses the JSON boundary as a **string**. Dates
are `YYYY-MM-DD`. Absent fields are `null` — never an empty string, a zero or a
plausible guess. Malformed JSON gets one corrective retry showing the model its own bad
output; a truncated response fails loudly as `TruncatedResponse` rather than as a JSON
parse error; per-field salvage keeps the fields that did parse.

*Implementation note:* `Receipt.date` originally shadowed the `datetime.date` import
inside its own annotation, so Pydantic resolved the type to `NoneType` and silently
rejected **every** date — which would have zeroed date F1 with no visible error. Fixed
and covered by a regression test. Worth recording because the failure was silent.

### 3.3 Confidence: four signals, fitted weights

| Signal | Strength | What it knows |
|---|---|---|
| `validated` | **Strongest** | Deterministic arithmetic — reconciliation, rows multiply out, rows sum to subtotal, date parses, currency is ISO-4217, and for invoices due ≥ issue and vendor ≠ customer. **Model-free, cannot be bluffed** |
| `grounded` | Strong | Does the value appear in the transcription, type-aware and whole-token? |
| `agreement` | Moderate | Do Model A and Model B agree after normalisation? Row-level F1 for tables |
| `self_report` | **Weakest** | The model's stated certainty |

**Known limitation:** Model B reads Model A's transcription, so they are only partially
independent — a misread digit propagates and inflates agreement. This is precisely why
`validated` exists: it is independent of the reading step. True reading independence
needs a second vision model on the raw image (§5).

Weights are **fitted per field group** by logistic regression on the calibration split;
groups with too few observations keep default weights rather than overfit. The threshold
is read off a **risk–coverage curve** against a target precision, never hand-picked.
Real fitted values are in §4.2.

### 3.4 Evaluation harness

Scoring runs after a normalisation layer (dates → ISO, money → `Decimal`, currency
symbols, case and whitespace) so metrics measure semantic correctness, not string noise.

**Accuracy:** per-field precision / recall / F1, reported **normalised and exact** (the
gap is format sensitivity); micro and macro F1; **Document Accuracy Rate**; and
**line-item F1** via greedy one-to-one matching on description similarity plus amount.

**Error taxonomy:** `missing` (under-extraction), `spurious` (**hallucination**),
`wrong_value`, `format_error` (same content, different format — deliberately not counted
as wrong) and `swapped_field` (the value belongs to another field, e.g. vendor ↔
customer).

**Confidence quality:** AUC for discrimination, ECE for calibration, plus a reliability
diagram and risk–coverage curve. The two have different fixes, and the scorecard prints
the diagnosis: AUC ≈ 0.5 means the signals must change; good AUC with high ECE means
refit.

**Two rules the harness enforces about its own inputs** (both added after they bit —
see §4.3):

- A field with **no label** is not scored, and the scorecard reports how many were
  skipped. A `null` label means "not on the document"; an unlabelled field is a
  different thing, and conflating them scores a correct reading as a hallucination.
- Calibration **refuses to fit** on less than 80% of the split and writes nothing,
  rather than producing a threshold from whatever documents ran before a quota expired.

### 3.5 Deployment and the CPU constraint

No GPU on the target machine, which decided where each model runs: both models hosted
(a 3B model generating ~600 tokens of JSON on CPU takes 40–90 s/doc against 3–8 s
hosted), with validation, groundedness and metrics local, deterministic and free. All
model access goes through one OpenAI-compatible provider abstraction, so endpoints are
swappable via `.env` with no code change — which is what makes a local open-weights VLM
a config change given a GPU.

**Caching by input + model + prompt hash** keeps API calls to one per document across
repeated eval runs. Re-running the harness after a metrics change costs zero calls.

**Measured free-tier limits** (observation, not documentation). Google's free tier
allows **20 requests per model per day** — not the 250 its docs suggested — which stopped
a calibration run at 8 of 18 documents; billing was enabled to lift it. OpenRouter's free
tier allows **50 requests per day account-wide**, which cannot cover a single 39-document
test run, so Model B moved to **Groq**: 1000 requests/day at 8,000 tokens/minute, and
1–2 s per call against nemotron's 18–22 s. The token window, not the daily cap, is what
now paces a run — roughly 2–3 documents per minute. A model slug must be checked against
the key's own catalogue before it is pinned: `llama-3.3-70b-versatile` 404s on this
account, which cost a cycle to discover.

### 3.6 Production surface

`docker compose up` serves the API; `docker compose run --rm eval` waits for the
healthcheck and runs the harness **against that running service**, so the harness scores
exactly the code the API runs.

Endpoints: `GET /health`, `GET /metrics`, `POST /extract`, `POST /extract/batch` with
bounded concurrency (free tiers rate-limit; unbounded fan-out only collects 429s).
`POST /extract` takes a `document_type` of `receipt` or `invoice` and returns 400 for
anything else.

**The output contract is documented and tested** (`docs/OUTPUT.md`): `document`,
`confidence`, `abstained`, `withheld`, `signals`, `validation`, `validation_notes`,
`transcription`, `usage`. Abstention is applied **at the boundary** — the value is
removed from `document` and kept under `withheld`, so a consumer never reads an
unasserted value while a human reviewer can still see what was read. A test fails if the
documented payload drifts from the code.

**Resilience:** a failed document returns 200 with a populated `error` and is scored as
an empty extraction — a recall failure, never a silent drop from the denominator.

**Regression gate:** exits 0 (ship), 1 (regression) or 2 (invalid comparison). It fails
on any *new* hallucination, and on an F1 drop only when at least two documents flipped —
at this size a threshold alone cries wolf. Runs are fingerprinted (labels, documents,
prompts, models, threshold), and it **refuses to compare across a label change**,
because the delta would measure the relabelling.

### 3.7 Reviewer artifacts

| Artifact | What it answers |
|---|---|
| `outputs/<run>/scorecard.{md,html,json}` | How good is the system |
| `outputs/<run>/documents/<id>.html` | What did it read off **this** document, and why believe it |
| `outputs/dashboard.html` | All 60 documents, source beside output, filterable by type, status and confidence band |
| `outputs/gt_review.html` | Each document beside its labels, audit flags first — the verification tool |

All are self-contained single files (inline CSS, base64 images) so they work over
`file://`. The per-document views show **no ground truth** and recompute nothing: the
arithmetic verdicts come from the payload, so a page and the API can never disagree.

---

## 4. Results

### 4.1 What has and has not been run

| | |
|---|---|
| Calibration run (18 receipts, `pipeline`) | **Done** — §4.2 |
| Extraction over all 60 documents for the dashboard | **Done** — §4.4 |
| Cost measurement from real token counts | **Done** — §4.6 |
| **Accuracy scorecard on `test`** | **Done** — §4.5 |
| Cascade before/after | **Done** — §4.7, a measured negative result |
| VLM-only baseline (does Model B earn its cost?) | **Done** — §4.8 |
| Degraded-subset breakdown | **Done** — §4.5 |
| Synthetic robustness curve | **Not run** |
| API-mode run against the container | **Not run** (verified against a stub) |

### 4.2 Confidence quality — measured

Run: 18 calibration receipts (invoices excluded from the fit), 198 field observations,
100% of the split extracted.

| Metric | Value | Target |
|---|---|---|
| AUC (discrimination, in-sample) | **0.908** | > 0.75 |
| ECE (calibration, in-sample) | **0.032** | < 0.10 |

**Fitted weights per field group** — a finding in their own right:

| Group | n | agreement | grounded | validated | self_report | bias |
|---|---:|---:|---:|---:|---:|---:|
| string | 54 | **+1.50** | −0.22 | +0.42 | −0.39 | +1.33 |
| money | 90 | +1.15 | +1.02 | +0.23 | −0.08 | +0.97 |
| date | 18 | +1.10 | −0.19 | −0.19 | **−0.80** | +1.65 |
| currency | 18 | +1.03 | +0.61 | +0.92 | +0.88 | +2.20 |
| table | 18 | −0.79 | +0.34 | **+1.23** | −0.18 | −0.82 |

Three things worth stating plainly:

1. **`self_report` is negative in four of five groups.** The model's own stated certainty
   is *anti-informative*: when it says it is sure, it is likelier to be wrong. That is the
   argument for a multi-signal design rather than trusting a model's self-assessment, and
   it is a measurement, not a prior.
2. **`validated` dominates the table group (+1.23)** while contributing modestly
   elsewhere — as designed. The arithmetic check is the signal that cannot be bluffed, and
   a table is where it has the most to say.
3. **`agreement` is the strongest signal for strings (+1.50) and dates (+1.10)** — fields
   where no arithmetic applies and grounding is weak, because a transcription contains the
   text either way. This is the whole return on the second model (§4.8).

**Operating point**, read off the risk–coverage curve:

> At threshold **0.75** the system keeps **86.4%** of fields at **95.3%** precision.

`CONFIDENCE_THRESHOLD=0.75` in `.env`. Held out on `test`, the same threshold kept 87.8%
of fields at 95.8% precision (§4.5) — the operating point transferred.

*In-sample, on 18 documents and 198 field observations.* These numbers set the threshold;
the reported result is §4.5.

### 4.3 What the harness caught in its own ground truth

The evaluation found four defects that would each have produced confident, wrong
numbers. They are reported because the catching is the point.

| Defect | Symptom | Fix |
|---|---|---|
| **Unlabelled fields scored as absent** | `merchant_name` had no label on 48 of 60 documents; a correct reading counted as `spurious` — a **hallucination** — and taught the confidence model that a visible, grounded string is probably wrong | Fields with no annotation are excluded from scoring and reported as a gap |
| **WildReceipt strips spaces** (`TUSCANYRISTORANTE`) | Correct readings scored as `wrong_value` | Whitespace-insensitive string comparison, documented as a deliberate lenience |
| **Partial calibration accepted silently** | A rate-limited run fitted weights on 8 of 18 documents, wrote a threshold and exited 0 — the report even recorded `n_documents: 18`, the count it asked for rather than the count it saw | Refuses to fit below 80% coverage; records real coverage and every failure |
| **Scorecard written in the platform default encoding** | Em dashes became undecodable bytes in a file declaring UTF-8 | Explicit UTF-8 on every write |

The evidence for the first two, measured on the calibration split before the fix:
string fields that *were* grounded scored **15% correct**, against **93%** for
ungrounded ones. Grounding was working perfectly and the scorer was calling it wrong.
Fixing it moved the `grounded` weight for strings from **−0.95** (anti-informative) to
**+0.81**, and let `date` and `currency` be fitted at all.

It also made the headline AUC *fall* from 0.917 to 0.845 — the earlier number was
largely the model learning to predict broken labels, a leaked feature. About 44 of the
59 "errors" it had been trained to detect were not errors.

### 4.4 System behaviour over all 60 documents

From the dashboard run (`eval/dashboard.py`, every labelled document). **These are
self-checks, not accuracy** — they say what the system did and whether its own
arithmetic held, not whether it matched ground truth.

| | n | share |
|---|---:|---:|
| Documents extracted | 60 | — |
| All model-free checks passed | 27 | 45% |
| At least one field abstained | 23 | 38% |
| At least one check failed | 10 | 17% |
| Mean confidence ≥ 0.80 | 54 | 90% |
| Mean confidence 0.50–0.79 | 5 | 8% |
| **Read by one model only** | **16** | 27% |

The last row is a caveat on the rest: OpenRouter's free tier capped mid-run, so 16
documents were read by the vision model alone. With Model B absent the `agreement`
signal does not fail — it defaults to a neutral 0.50 — so those documents carry a
normal-looking confidence built on three signals instead of four. They are badged on the
dashboard and say so in words. Re-running them needs only Model B calls; the vision side
is cached.

### 4.5 Accuracy — measured

Run `groq-final`, `test` split (39 documents: 32 receipts + 7 invoices), Gemini 3.6
Flash + GPT-OSS 120B, threshold 0.75, every label human-verified, every document read by
both models.

| Metric | Receipts (headline, n=32) | Invoices (n=7, separate) |
|---|--:|--:|
| Document accuracy | **21.9%** | 14.3% |
| Micro F1 | **0.918** | 0.800 |
| Macro F1 | 0.917 | — |
| Line-item F1 | **0.951** | — |
| Hallucination rate | **0.63%** | 4.8% |
| Latency p50 / p95 | 16.1 s / 39.9 s | — |
| Cost per document | $0.0058 (list-price equivalent) | — |

**Abstention, the central claim of the design, holds on held-out data.** Keeping 87.8% of
fields raises precision from **0.906 to 0.958** — the risk–coverage curve was fitted for
95% and the held-out result landed at 95.8%.

**The degraded subset barely degrades.** The twelve worst scans inside `test`, tagged by
sharpness and exposure rather than by source, score **micro F1 0.897** against 0.918 for
the full set — a two-point drop — with line-item F1 *higher* at **0.977**. Abstention
behaves the same way there (87.1% of fields kept, precision 0.894 → 0.948), which is what
a confidence signal should do when the input gets harder: keep slightly less, stay
accurate on what it keeps.

| Subset | n | Micro F1 | Line-item F1 | Hallucination | Abstention (kept @ precision) |
|---|--:|--:|--:|--:|--:|
| Receipts, all of `test` | 32 | **0.918** | 0.951 | 0.63% | 87.8% @ 0.958 |
| Degraded subset | 12 | 0.897 | **0.977** | 1.72% | 87.1% @ 0.948 |
| Invoices | 7 | 0.800 | — | 4.84% | — |

**Per field** (normalised / exact F1): `total` 1.00 / 0.81, `date` 1.00 / 1.00,
`currency` 1.00 / 1.00, `service_charge` 1.00 / 0.86, `subtotal` 0.95 / 0.64,
`tax` 0.94 / 0.94, `payment_method` 0.83 / 0.50, `merchant_name` 0.80 / 0.70,
`merchant_address` 0.46 / 0.46. The gap between normalised and exact on `subtotal`
(0.95 vs 0.64) and `payment_method` (0.83 vs 0.50) is **format, not content** — which is
exactly why the harness reports both and why `format_error` is a separate label.

**Failures by kind** (receipts): 305 correct, 6 `wrong_value`, 5 `format_error`,
2 `missing`, 1 `spurious`, 1 `swapped_field`. The single swap is on an invoice party —
the failure invoices were added to detect.

**Document accuracy of 21.9% is strict by construction**: every field and every line item
must be right. Micro F1 is the number that describes field-level behaviour. At n=32 one
document moves document accuracy by 3.1 pp, so treat it as coarse.

#### What the harness caught, and the fix it verified

The first real run reported a **7.65% hallucination rate** against a design that promises
approximately zero. The drill-down located it immediately: **12 of 13 spurious values were
an inferred `currency`** — "IDR" written onto CORD receipts that print no symbol at all.
Precision on that field was 0.56 against recall 1.00: the model never missed a printed
currency, it only invented absent ones.

The prompt already said "never guess or invent values", which was too weak, because a
currency *is* deducible from the language and the amount format — the model was not
inventing in its own terms. The rule now names the inference and forbids it:

> currency: ONLY when a symbol or code is printed on the document (Rp, $, EUR, GBP,
> AED...). Do NOT infer it from the language, the country, the tax name or the amount
> format. An Indonesian receipt that prints no symbol has currency null. Reporting a
> currency nobody printed is a hallucination, and a null here is a correct answer.

| | before the fix | after |
|---|--:|--:|
| Hallucination rate | 7.65% | **0.63%** |
| Spurious values | 13 | **1** |
| Currency F1 (P / R) | 0.71 (0.56 / 1.00) | **1.00 (1.00 / 1.00)** |
| Micro F1 | 0.866 | **0.918** |
| Document accuracy | 15.6% | **21.9%** |

Recall on `currency` stayed at 1.00, so the rule removed the inferences without making the
model refuse currencies that *are* printed — the specific way such a fix can overshoot.

The same run also failed `inv_0010` (225 line items over 5 pages) with `TruncatedResponse`
at 24,576 output tokens: loudly, as designed, and scored as an empty extraction rather
than a silent truncation. The budget is now 16,384 + 8,192 per extra page, and the
document extracts.

### 4.6 Cost and latency — measured

Median per-call figures from real token counts, priced at list rates from `.env`:

| Call | Prompt | Completion | Cost |
|---|---:|---:|---:|
| Model A (VLM) | 2,758 | 853 | **$0.00296** |
| Model B (LLM) | 1,573 | 1,576 | **$0.00022** |
| **One receipt, both models** | | | **$0.0032** |

The vision call costs ~13× the text call: image tokens dominate. Projected cost of the
full remaining programme — calibration, the test run, a cascade run and a VLM-only
baseline — is **$0.40**, or about **$0.80** allowing a 2× margin for re-runs.

Cost is reported as a **list-price equivalent** rather than actual spend, so free-tier
and paid runs produce the same $/doc figure and the cascade comparison stays meaningful.

### 4.7 Optimization: confidence cascade — measured, and it does not pay

Cheap first pass (`gemini-3.5-flash-lite`), escalating to the full pipeline on
`arithmetic_failed`, `ungrounded:<field>`, `low_confidence:<field>`, `consistency_failed`
or an empty extraction. Same 39 documents, same labels.

| | Pipeline (`groq-v3`) | Cascade | Δ |
|---|--:|--:|--:|
| Cost per document | $0.00579 | $0.00679 | **+17.3%** |
| Tokens per document | 5,884 | 10,857 | **+84.5%** |
| Latency p50 | 16.1 s | 20.4 s | +27.1% |
| Latency p95 | 39.9 s | 49.4 s | +23.7% |
| Micro F1 | 0.918 | 0.918 | 0 |
| Document accuracy | 21.9% | 21.9% | 0 |
| Hallucination rate | 0.63% | 0.63% | 0 |
| **Escalation rate** | — | **100%** | — |

**Every document escalated**, so the cascade paid for a cheap pass that was then thrown
away, and the result is strictly worse: more cost, more tokens, more latency, identical
accuracy. Reported as a **negative result**, because a measured failure is the outcome the
harness exists to produce and burying it would make the other numbers less trustworthy.

Escalation reasons across the 39 documents: `low_confidence` 38, `ungrounded` 9,
`arithmetic_failed` 3, `invalid_output` 1 (a document can raise several). The gate fires
on `low_confidence` for almost everything because `CASCADE_ESCALATE_BELOW` defaults to the
abstention threshold, 0.75 — a bar the cheap model rarely clears on at least one field of
a real receipt. Two changes would make the idea testable rather than dead:

1. **Separate the escalation threshold from the abstention threshold.** They answer
   different questions: "should a human see this?" against "is the cheap read good
   enough?". Fitting the second on the calibration split is the honest version.
2. **Escalate on the model-free checks alone** (`arithmetic_failed`, `invalid_output`,
   `ungrounded`), which fired on 13 documents rather than 38. That would accept roughly
   two-thirds of documents from the cheap model, which is where a saving could exist.

Neither was measured, so neither is claimed. What is claimed is the negative: **as
configured, the cascade costs 17% more for no accuracy gain.**

### 4.8 Does the second model earn its cost?

The VLM-only baseline exists so that "two models beat one" is a measurement rather than
an assertion. Run on the same 39 documents, same labels, same token budget:

| | VLM only | Pipeline (VLM + LLM) | Δ |
|---|--:|--:|--:|
| Micro F1 | 0.918 | 0.918 | **0** |
| Macro F1 | 0.917 | 0.917 | −0.1 pp |
| Line-item F1 | 0.948 | 0.951 | +0.2 pp |
| Hallucination rate | 0.63% | 0.63% | 0 |
| Document accuracy | **40.6%** | 21.9% | **−18.7 pp** |
| Cost per document | ~$0.0030 | $0.0063 | **2.1×** |
| Latency p50 | ~8 s | 19.9 s | **2.5×** |

**Field-level accuracy is identical, and the pipeline is worse at document level.** This
is not a subtle result and it should not be buried.

Two facts explain it, and they matter more than the headline:

1. **Model B never changes the answer.** By construction `result.document` is Model A's
   extraction; Model B's re-read populates `document_b`, which exists only to produce the
   `agreement` signal. So the second model cannot add accuracy — it was never wired to.
   Its entire contribution is to confidence, where `agreement` carries the largest fitted
   weight in three of five field groups (§4.2).
2. **The document-accuracy gap is a prompt effect, not a model effect.** The pipeline
   asks Model A for a transcription *and* fields *and* a self-reported certainty in one
   call; the baseline asks for fields alone. Same model, different instructions, and the
   errors redistribute: identical micro F1 with 18.7 pp lower document accuracy means the
   pipeline makes single-field mistakes across *more* documents while the baseline
   concentrates its errors in fewer. At n=32, 18.7 pp is about six documents — well past
   the 3.1 pp one-document noise floor.

**What this justifies and what it does not.** It justifies the second model *as a
confidence signal*, and the ablation in §3.1 measures that: neutralising `agreement`
drops AUC from 0.908 to 0.852 and coverage at the 95% precision target from 86.4% to
75.8% — 10.6 points of fields that would otherwise need a human. It does **not** justify
the second model as an accuracy mechanism, and the report does not claim it does.

**The experiment this points at**, not run: split the VLM call so that fields are
extracted by a fields-only prompt (the baseline's, which scores better per document) and
the transcription is produced separately, keeping Model B for agreement. That trades one
extra vision call against a possible 18 pp of document accuracy. It is the obvious next
measurement and it is cheap; it is left undone rather than guessed at.

### 4.9 Statistical honesty

- **DAR is binary per document**, so at n=39 in `test` its 95% CI is roughly ±8pp. Two
  versions differing by 3pp are not distinguishable.
- **Per-field F1 is more stable** (several hundred field-level judgments), which is why
  the regression gate rests on per-field metrics rather than DAR.
- **Invoices at n=10 are indicative only**, and their confidence is uncalibrated: the
  weights were fitted on receipts, and 10 documents cannot support their own fit.
- **Ground truth is single-annotator** and, at present, mostly unverified. No
  inter-annotator agreement has been measured, so label noise is unquantified.
- **The calibration numbers in §4.2 are in-sample** on 18 documents. They set the
  operating point; they do not demonstrate it.

---

## 5. What comes next

Ordered by evidence, not by appeal. The first two follow directly from measurements in
§4 and are cheap; the rest are real work.

| # | Item | Why, and what it would cost |
|---|---|---|
| 1 | **Split the VLM prompt: fields-only, transcription separately** | §4.8's sharpest result: asking one call for fields *and* a transcription *and* a self-report costs **18.7 pp of document accuracy** against a fields-only prompt, at identical micro F1. A second vision call is ~$0.003/doc. If the gap holds, this is the largest single improvement available, and it is one run to find out. |
| 2 | **Separate the cascade's escalation threshold from the abstention threshold** | §4.7: the cascade escalated **100%** of documents because `CASCADE_ESCALATE_BELOW` defaults to the abstention threshold, which answers a different question. Escalating on the model-free checks alone fired on 13 documents rather than 38, so a saving plausibly exists — currently unmeasured. |
| 3 | **Route abstained fields to human review** | Abstention is measured (87.8% kept at 95.8% precision) but nothing consumes it. A queue that shows the withheld value, its four signals and the source image turns honesty into throughput — the review server in `scripts/` is most of the UI already. |
| 4 | **A second, independent VLM on the raw image** | §3.3's stated weakness: Model B reads Model A's transcription, so a misread digit propagates and inflates `agreement`, which the ablation shows is the signal carrying strings and dates. True reading independence would cost a second vision call and make the strongest signal trustworthy. |
| 5 | **More invoices** | At n=7 in `test`, invoice numbers are indicative and their confidence is uncalibrated, since the model is fitted on receipts. Roughly 40 invoices would support their own fit and a reportable rate. |
| 6 | **Inter-annotator agreement on a 15-document subset** | Ground truth is single-annotator. §2.4 found a 10% money-field error rate in one source *before* verification; the residual rate after it is unknown, and a second annotator on a subset is the only way to bound it. |
| 7 | **The two runs marked not-run in §4.1** | The synthetic robustness curve (25 degraded copies) and an API-mode run that scores the container rather than the library. Neither is hard; both were left rather than guessed. |
| 8 | **Stack splitting (`POST /extract/stack`)** | `inv_0006` is three documents in one PDF. The design is already written down in `docs/OUTPUT.md` §2 as a separate endpoint, specifically so it cannot break `/extract`. |
| 9 | **A local open-weights VLM, given a GPU** | Model B (`openai/gpt-oss-120b`) is already open-weights; only Model A is tied to a hosted API. With a GPU both halves could run locally, and the provider abstraction makes it a `.env` change. |
| 10 | **The regression gate in CI** | It exits 0/1/2 and refuses to compare across a label change, so it is ready to block a PR against a committed baseline. Nothing in the harness needs to change; it needs a workflow file. |
