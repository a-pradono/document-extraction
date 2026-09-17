# Receipt extraction and evaluation

Extracts typed, validated fields from messy real-world receipts with a VLM reader and a
text-LLM cross-check. Deterministic validation sits between the two, and every field gets
a confidence score, so the system abstains instead of guessing. The evaluation harness
is the main deliverable: it shows **how** the system fails, not just how often, and it
blocks a regression before it ships.

> **Status of the numbers.** Everything below is built and tested (374 unit tests plus
> a 7-check offline end-to-end run over real HTTP). **No result in this README comes from
> a real model run yet**, and the 50 ground-truth labels are not yet hand-verified.
> Results sections say *pending* and give the exact command that fills them. That's
> deliberate: a stub-model number presented as a result would contradict the point of
> the harness.

---

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add a Google AI Studio key and an OpenRouter key

pytest eval/tests -q                                 # 374 tests, no keys needed
python -m agent.offline_e2e                          # whole system over HTTP, no keys needed

python -m extractor.pipeline data/docs/cord_0001.jpg # one receipt -> fields + confidence
python -m eval.run --calibrate                       # fit confidence on the calibration split
python -m eval.run --set test --version v1           # the scorecard -> outputs/
python -m eval.regression outputs/<v1> outputs/<v2>  # exit 1 if v2 regressed

uvicorn api.app:app --port 8000                      # or: docker compose up
python -m api.batch path/to/folder                   # many receipts -> results + cost/latency
```

---

## 1. Extraction

### Schema

[`extractor/schema.py`](extractor/schema.py): `merchant_name`, `merchant_address`, `date`,
`currency`, `subtotal`, `service_charge`, `discount`, `tax`, `total`, `payment_method`,
and `line_items[]` (`description`, `quantity`, `unit_price`, `line_total`).

- **Money is `Decimal`, never float.** Ground truth stores money as strings.
- **Absent means `null`.** The schema has no defaults that could pass for a value.
- **`line_items` is a real table**, so row alignment, dropped rows and merged rows are
  things the system and the metrics both have to handle.
- **`service_charge` and `discount` exist because the data demanded them.** Auditing the
  real labels showed `subtotal + tax == total` is simply false on 12% of CORD receipts
  (service charge) and on tipped WildReceipt ones (238,000 + **17,850** + 25,585 =
  281,435). Without those fields, the strongest confidence signal failed on *correct*
  extractions. Reconciliation is `subtotal + service_charge − discount + tax == total`,
  and it accepts a total rounded to a cash step (CORD prints 334,000 for 334,011). That
  rule is exact, not a looser tolerance, so a misread digit still fails.

### Architecture, and why this one

```
pages ──► Model A: VLM (Gemini 3.6 Flash) ──► transcription + fields + self-reported certainty
              │                                          (ONE call, all pages together)
              └─ transcription ──► Model B: text LLM, different family (NVIDIA Nemotron 3 Super)
                                             ──► independent re-extraction
                                   ▼
            deterministic validation (arithmetic, row math) + type-aware grounding
                                   ▼
          per-field confidence (logistic, fitted per field group) ──► abstain below threshold
```

**Why a VLM reads the image, rather than OCR followed by an LLM.** Receipts are a
layout problem. Line items sit in columns, and which of five numbers is the total is
decided by position. Flat OCR text throws that away. The VLM also returns a
transcription **in the same call**, which then serves as the grounding source without a
second vision call or a slow local OCR engine. (PaddleOCR was tried and dropped: too slow
on CPU.)

**Why a second model, from a different family.** Model B re-extracts from A's
transcription and does what VLMs are weakest at: cross-section reasoning and arithmetic.
It must come from a different family. Siblings misread the same smudged digit the same
way, which inflates agreement exactly where it needs to be honest.

**Why not an agent loop.** A receipt is a bounded, single-pass document with nothing to
look up and no tools to call. An agent loop would add latency, cost and non-determinism
without adding information. The adaptive part a loop would provide, spending more effort
on hard receipts, is provided more cheaply by the **confidence cascade** (section 3),
which escalates only on concrete evidence of doubt.

**Known limitation, stated plainly:** B reads A's transcription, so the two are only
*partially* independent, and a misread digit propagates into both. That's why
deterministic validation is a separate signal that no model can bluff, and why
confidence quality is *measured* with AUC and ECE instead of assumed. Real reading
independence would need a second VLM on the raw image, noted under "with more time".

### Confidence and abstention

Four signals per field, ranked by how much they actually know
([`extractor/confidence.py`](extractor/confidence.py)):

1. **validated.** Model-free arithmetic: `subtotal + service_charge − discount + tax ==
   total` (cash rounding allowed), the rows sum to the subtotal, and each row satisfies
   `qty × unit_price == line_total`. The strongest signal, because it can't be bluffed.
2. **grounded.** Can the value be read from the transcription?
   ([`extractor/grounding.py`](extractor/grounding.py)) Type-aware: the parsed `53280` is
   compared against printed `53.280` under both separator conventions, dates across
   printed orders, and `IDR` against a printed `Rp`. Matching is on whole tokens, so an
   invented `5` does not ground inside `53.280`.
3. **agreement.** Do A and B return the same normalized value? For the table this is
   row-level line-item F1, so one dropped row lowers agreement without zeroing it.
4. **self_report.** The model's own certainty. The weakest signal: LLM self-confidence
   is reliably overconfident.

Weights are **fitted** by logistic regression on the calibration split, **per field
group** (string / money / date / currency / table), because `merchant_name` and `total`
fail in completely different ways. The threshold is read off the risk–coverage curve
against a target precision, not picked by hand. `--calibrate` refuses to recommend a
threshold when the target is unreachable.

**Abstention is enforced at the boundary.** The API returns an abstained field as
`null` (an abstained table as `[]`, never a partial table). What the model read goes in
`withheld` for a human reviewer, so a consumer reading `receipt` never sees a value the
system isn't confident in.

### The messy parts

| Problem | Handling |
|---|---|
| Multi-page | Every PDF page is rendered and sent to the VLM **in one call**, in order, with a note that items continue across pages and totals sit on the last page. Documents over `MAX_PAGES` are **refused, never truncated**, because truncation is how a missing or invented total gets in. |
| Phone photos | EXIF rotation is applied; otherwise the model gets a sideways receipt. |
| Tables | Row arithmetic, table-level grounding, row-level agreement, and confidence and abstention on the whole table. |
| Absent fields | `null`, and scored as correct when ground truth is also null. |
| Long receipts | Output budget scales with page count, and a `max_tokens` stop raises `TruncatedResponse` instead of failing as a baffling JSON parse error. Gemini 2.5 is a thinking model, and its reasoning tokens come out of the same budget. |
| Hallucination | Type-aware grounding feeds confidence, abstained values are withheld, and the regression gate fails on **any** new hallucination by default. |
| Cross-section fields | Items, charges, tax and total sit in different parts of a receipt; they're tied together by row arithmetic and the full reconciliation above, checked without a model. |
| Bad model output | Unparseable JSON gets **one text-only repair call** (no images re-sent). A cut-off response is never repaired. A schema-invalid value (`"12 Maret"`) is validated **per field**: the other fields survive, and the invalid one is forced to confidence 0 and abstained, never shipped or silently dropped. A table with an invalid row is abstained whole. |

---

## 2. Evaluation

### Ground truth

About 65 documents from three independent sources: two languages, three label schemas
([`DATASETS.md`](DATASETS.md), [`data/README.md`](data/README.md)).

| Source | Docs | Why |
|---|--:|---|
| CORD v2 (Indonesian) | 25 | Structured line items; subtotal → total arithmetic |
| WildReceipt | 25 | Template diversity, photographed in the wild |
| OCR Receipts, retail (Kaggle) | 15 | Messiest tier. **Pending**: needs a Kaggle token, and subtotal/tax must be hand-labeled |

- **Stratified sampling** by line-item count and merchant. Taking the first N would
  produce an accidentally easy set that flatters every metric.
- **Provenance per document** in `data/ground_truth/_provenance.json`: source, original
  file, which fields were script-converted vs hand-labeled, which still need a human,
  and whether the document has been verified. That separates a null meaning "not on the
  receipt" from a null meaning "nobody looked yet".
- **Verification status: 0 of 50 verified.** `python scripts/verify_gt.py --html`
  renders each image beside its labels. **No scorecard from this set should be reported
  until verification is at 100%.**
- **Splits** ([`scripts/assign_splits.py`](scripts/assign_splits.py), fixed seed,
  committed): calibration 18 / test 32 now, 22 / 43 once retail lands. `smoke` is drawn
  **from calibration**, so heavy iteration can't overfit the reported numbers. Nothing is
  fitted on `test`. `degraded` tags the worst scans **inside** test, stratified by source
  so the subset measures image quality and not "is this CORD". `synthetic` holds blurred,
  rotated and faded copies, kept out of the headline scorecard structurally, not by
  convention.

**Auditing the labels before trusting them.** [`scripts/audit_gt.py`](scripts/audit_gt.py)
checks every label without a model: reconciliation (using the *same* rule and tolerance
as the pipeline), row arithmetic, rows summing to the subtotal, implausible dates,
duplicated rows, and space-stripped text. `verify_gt.py --html` shows the flags beside
each image and puts the most suspicious documents first. On the real 50 labels it drove
four converter fixes:

| Audit finding | Cause | Fix | Flags |
|---|---|---|---|
| CORD totals ≠ subtotal + tax | service charge / discount dropped by the converter | schema fields + converter mapping | `total`: 15 → 3 |
| WildReceipt tipped totals ≠ subtotal + tax | `Tips_value` class unmapped | mapped to `service_charge` | (included above) |
| Rows with a price and no description | tilted photos: each price sits ~11 px below its item, so nearest-neighbour pairing shifted whole tables by a row | **order-preserving alignment** (rows can't cross) | row flags 32 → 26 |
| Cash-rounded totals rejected | fixed 0.02 tolerance | exact cash-rounding rule | – |

Overall, documents flagged fell **31 → 21** and flags **71 → 54**. The fixed pairing also
exposed a real **error in the WildReceipt annotation itself**: wild_0018's last price is
annotated `5390` for a printed 5.90, which is exactly what hand verification is there to
correct. Earlier conversion issues: WildReceipt annotates each total up to four times
(they were concatenated into `77437743774377.43`), and its text strips spaces
(`CaltexCentralMyrtleford`), so those fields are flagged for re-spacing by hand.

Converter fixes re-convert the **same pinned documents** (`--resample` to redraw).
Otherwise a changed row count would reshuffle which receipts are in the set and
invalidate every split and every verification. Verification status carries over only
when a document's labels are byte-identical to what was verified.

### Second document type: invoices

Ten real invoices test two things receipts can't: **multi-page documents** and
**vendor/customer role resolution**. There are two parties, each with a name, address and
tax ID, and returning them swapped is the failure that matters.

- **Source: real, not synthetic.** The suggested synthetic set was checked and rejected:
  one template, Faker data, totals that don't add up, and every document a single page.
  The ten come from **RVL-CDIP-N multipage** (`jordyvl/rvl_cdip_n_mp`, CC BY-NC 4.0).
  They are public-records invoices from a police department, TV stations, a school
  district, law firms, a state agency and small manufacturers, and 1 to 5 pages each.
  [`scripts/fetch_invoices.py`](scripts/fetch_invoices.py) streams the archive and pulls
  only the ten pinned files, verified by SHA-256 ([`DATASETS.md`](DATASETS.md) records
  the search).
- **Schema:** `Invoice` has `vendor: Party` and `customer: Party` and reuses `LineItem`.
  Nested fields are scored, calibrated and withheld under dotted names
  (`customer.name`). `swapped_field` detects vendor↔customer swaps.
- **Invoice-only validation:** due date ≥ issue date, and the vendor name must differ
  from the customer name. The arithmetic checks are the same as for receipts. Neither
  check can prove the roles are the right way round. They only catch documents that
  are obviously inconsistent, and those go to escalation.
- **Reported separately, never pooled.** Invoices have their own section under
  `by_document_type` in the scorecard. The headline numbers are receipts only, and the
  confidence model is fitted on receipts only. **Confidence is uncalibrated for
  invoices.** With 3 invoices in calibration and 7 in test, invoice numbers are
  anecdotes, not rates.
- **Labels are drafted, pending human verification.** Every judgment call is recorded
  in `_provenance.json` (`label_notes`). These are the ones to check first:
  - `inv_0006` is a bundle of three documents, and only invoice 29069 is labeled.
    Replacing it is the likely fix.
  - `inv_0005` has a struck-through row. It's kept because the printed total includes it.
  - `inv_0003`'s invoice number is redacted, so it's labeled null.
  - `inv_0004` has a printed arithmetic error, which is kept deliberately as a test case.
- **Privacy:** these are real public records that name private individuals. This is
  why the repository is private.

### Metrics that show how it fails

[`eval/metrics.py`](eval/metrics.py), [`eval/taxonomy.py`](eval/taxonomy.py)

- **Normalize before scoring** (money → Decimal, dates → ISO, currency symbols → codes),
  so semantic correctness is scored, not string noise. Exact F1 is reported beside
  normalized F1; the gap between them is format sensitivity.
- **Per-field precision, recall and F1.** A field with no ground truth and no
  prediction anywhere is **undefined, not 0**, and is excluded from macro F1. Otherwise
  a field the system got entirely right reads as a total failure.
- **Every failure gets a kind:** `missing` · `spurious` (hallucinated) · `wrong_value` ·
  `format_error` · `swapped_field`.
- **Hallucination rate:** spurious values over predicted values.
- **Line items:** rows are aligned first, then each cell is scored; plus a perfect-table
  rate.
- **By subset:** the same headline metrics and top failure kinds per source (clean CORD,
  wild photos, retail) and for the degraded tag. A blended number averages away exactly
  the "messy documents" story.
- **Document accuracy rate.** Binary per document, so it's noisy: at n=32 one document
  moves it by 3.1 points, and every comparison says so.
- **Confidence quality:** AUC (does the score rank errors?), ECE (do the numbers mean
  what they say?), a reliability diagram, the risk–coverage curve, and precision on
  kept fields versus all fields. AUC near 0.5 means change the *signals*; good AUC with
  high ECE means *refit*.

### Comparing versions and catching a regression

```bash
python -m eval.compare    outputs/<v1> outputs/<v2>   # headline trade-off table
python -m eval.regression outputs/<v1> outputs/<v2>   # the ship gate
```

[`eval/regression.py`](eval/regression.py) exits **0** (ship), **1** (regression) or
**2** (invalid comparison), so CI can block a merge. Both tools **refuse to compare runs
over different documents**, since the delta would measure the sets.

**Runs are fingerprinted** ([`eval/fingerprint.py`](eval/fingerprint.py)): hashes of the
ground-truth labels, document bytes and prompts, plus the models, threshold, fitted
confidence model and git commit. If the **labels** changed between two runs (which
verification will do), the gate refuses with exit 2, because the delta would measure the
relabeling, not the system. Anything else that changed is listed at the top of the report
under "What changed between the runs", so every delta has a stated cause.

| Gate | Rule | Why this shape |
|---|---|---|
| Field F1 | fails if F1 drops past 5 pts **and** ≥ 2 documents flipped correct → incorrect on that field | At this size one document moves F1 by points; a threshold alone cries wolf, and a count alone misses rare fields |
| Hallucinations | **any** new spurious value fails | The one failure the system promises never to make |
| Document accuracy | past 5 pts **and** more than one document's worth | Same noise reasoning |
| Cost / p95 latency | +25% / +50% | An optimization that quietly doubles the tail should not ship either |
| Confidence AUC | a drop is a **warning** | Real, but too noisy at a few hundred observations to block on |

The report lists **newly failing documents with the exact transition**: "d1 total:
correct → wrong_value (gt 101, was 101, now 1)" is actionable, while "DAR −4.7 pts"
isn't. It also shows the failure-kind delta and newly fixed documents. A test builds
the case the gate exists for: a candidate **35% cheaper that regressed**, which fails.

---

## 3. Production

- **Batch:** `POST /extract/batch` and `python -m api.batch <folder>` run with bounded
  concurrency (free tiers allow about 10 requests/minute; wider fan-out just becomes 429
  retries). **One bad document never sinks the batch**: it comes back as an error entry.
  The folder runner appends each result to `results.jsonl` as it finishes, so an
  interrupted batch keeps everything already completed.
- **Cost:** tokens × `.env` prices, per stage, reported as a **list-price equivalent**
  (free-tier spend is $0). Cached calls keep their cost, because $/doc is a property of
  the system, not of whether this run hit the cache.
- **Latency:** model latency and wall time are reported separately, as p50/p95.
- **Cache** keyed on image bytes **+ model + prompt**. Keying on the image alone would
  have served the cascade model Flash's cached answers and measured nothing.
- **Config preflight:** a missing key fails once up front (503 from the API), not once
  per document.
- **`GET /metrics`:** failure rate, abstention rate, escalation rate, $/doc, cache hit
  rate, and rolling latency.
- **Docker:** `docker compose up` serves the API; `docker compose run --rm eval` scores
  the running container into `./outputs` (`--source api` restores withheld values before
  scoring and measures at the server's threshold). *The image has not been built on the
  development machine (no Docker there); its start command, healthcheck command and eval
  command were verified directly.*

### Optimization: confidence cascade

[`extractor/cascade.py`](extractor/cascade.py): Flash-Lite reads first and is accepted
unless the model-free checks (arithmetic, grounding), a low-confidence present field,
an empty read, or a failed call give a concrete reason to escalate to the full
two-model path. **An escalated receipt is billed for both paths**, so if escalation is
frequent, the cascade costs more than the baseline and the table has to show it.

| | Two-model baseline | Cascade | Δ |
|---|--:|--:|--:|
| $ / doc (list price) | *pending* | *pending* | |
| Latency p50 / p95 | *pending* | *pending* | |
| Document accuracy / micro F1 | *pending* | *pending* | |
| Escalation rate | – | *pending* | |

```bash
python -m eval.run --set test --source pipeline --version base
python -m eval.run --set test --source cascade  --version cascade
python -m eval.compare outputs/<base> outputs/<cascade> --out outputs/cascade_table.md
```

The scorecard also reports **field accuracy on accepted vs escalated documents**. If
accepted documents score well below escalated ones, the gate is letting bad reads
through and the saving is being paid for in accuracy. **First thing to check in the real
run:** VLMs often *infer* a currency the receipt never prints. Grounding rightly flags
that, but escalation can't fix it (Flash infers too). If the escalation reasons are
dominated by `ungrounded:currency`, the right move is abstaining on that field, not
escalating.

---

## 4. Agentic development

Built with **Claude Code**. The specs were written *for* the agent: [`CLAUDE.md`](CLAUDE.md)
holds the mission, the non-negotiables ("never hallucinate", "eval before
optimization"), the phase plan with done-when gates, and a Current state section the
agent keeps up to date. [`DATASETS.md`](DATASETS.md) is a dataset-acquisition spec with
its own non-negotiables. The agent implemented the phases against those; the human set
direction, made the calls only a human should make (which account, when to commit, what
counts as verified ground truth), and redirected when needed.

The most useful habit turned out to be **running the system rather than reading it**.
With no API keys available mid-build, the agent stood up a stub OpenAI-compatible server
and drove the real code through it. That surfaced bugs that every unit test had passed:

- `grounded` was a raw substring match: correct Indonesian totals scored as
  hallucinated, and an invented `5` scored as grounded, so the signal was
  *anti-*informative on the CORD tier;
- `localhost` on Windows added about 4 s to every model call through an IPv6 fallback,
  silently inflating latency;
- `Receipt.date` could only ever be `null` (the field name shadowed the `date` type),
  which had also broken the committed example ground truth;
- a missing API key was reported against every document instead of once up front;
- the cascade escalated 100% of documents. The cause was a fixture returning values
  absent from its own transcription, which confirmed the gate was right.

### The custom extension: `offline-e2e`

That stub became the shipped extension: a **Claude Code skill**
([`.claude/skills/offline-e2e/SKILL.md`](.claude/skills/offline-e2e/SKILL.md)) wrapping
[`agent/offline_e2e.py`](agent/offline_e2e.py) and
[`agent/stub_provider.py`](agent/stub_provider.py).

```bash
python -m agent.offline_e2e     # ~15 s, no keys, exit 0 = all checks pass
```

It starts the stub and the real API on free ports and checks seven behaviours through
real sockets: grounding of printed values, cache hits with zero new calls, 3 PDF pages in
one VLM call, truncation raising, the cascade accepting a clean read and escalating
(and double-billing) a contradictory one, API withholding and batch isolation, and
`eval --source api` plus the regression gate. The skill tells the agent to run it after
touching `extractor/`, `api/` or `eval/`, and before claiming a change works, and never
to report stub numbers as results.

**It was checked to confirm it can fail:** putting back the old substring grounding
fails `pipeline` and `cascade`, and dropping the cascade's billing of the cheap call
fails `cascade`. A check that can't fail proves nothing, and the first draft had one:
the API withholding check passed vacuously because nothing abstained. It now forces a
threshold that makes fields abstain.

---

## Trade-offs and limits

- **Not open-weights.** Both models are hosted, so the self-hosting bonus is forfeited.
  CPU-only inference was impractical: a 3B model producing about 600 tokens of JSON takes
  roughly 40–90 s per document, against 3–8 s hosted. Qwen2.5-VL would be swapped in
  given a GPU.
- **No own scans.** Compensated with three independent sources and a clearly labeled
  synthetic degradation tier, never presented as real-world variation.
- **n≈65 is small on purpose** (trustworthy beats large), which makes document accuracy
  noisy. Every comparison states its one-document noise floor.
- **Model A and B are partially correlated** (see Architecture).
- **Free-tier rate limits** cap throughput at about 10 requests/minute; the image-hash
  cache makes re-scoring after a metrics change cost zero calls.

## With more time

- A second **VLM** on the raw image for real reading independence.
- Abstain on inferred values (currency, date) instead of escalating them, if the real
  cascade run shows they dominate escalations.
- Self-consistency (N samples) as a fifth confidence signal on escalated documents only.
- Per-source and degraded-subset breakdowns as first-class scorecard sections.
- Run the regression gate in CI on every PR against a committed baseline run.

## Repository map

```
extractor/  schema, provider, extract (pages), pipeline, grounding, confidence, cascade, cost
api/        app (FastAPI), service, schemas, batch, telemetry
eval/       run, metrics, taxonomy, scorecard, compare, regression, tests/ (374)
agent/      stub_provider, offline_e2e          .claude/skills/offline-e2e/SKILL.md
scripts/    fetch_datasets.sh, fetch_invoices, convert_*, verify_gt, audit_gt, assign_splits, degrade
data/       docs/, ground_truth/ (+ _provenance.json), sets/, README.md
Dockerfile  docker-compose.yml  .env.example  CLAUDE.md  DATASETS.md
```
