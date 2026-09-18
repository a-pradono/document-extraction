# CLAUDE.md — Receipt Extraction & Evaluation System

Receipt extraction and evaluation system using AI.

`README.md` is for humans setting the project up. This file is the things that
will cost you an hour if you don't know them.

## Mission

Build a receipt document-extraction system that pulls **typed, validated, structured
fields** from *messy real-world* receipts using a **two-model design (VLM reader +
LLM reasoner/validator)**, and an **evaluation harness that proves accuracy**. The
eval harness is the primary deliverable, not the extractor. We are graded on being
able to *prove* the system is accurate, knows when it is unsure, and is reasonable on
quality, latency, and cost.

## Non-negotiable principles (do not violate these)

1. **Never hallucinate.** Absent fields return `null`. Never invent a value that is
   not visibly on the document. Every kept value must be groundable in the source text.
2. **Abstain instead of guessing.** Low-confidence fields are flagged/abstained, not
   filled with a best guess.
3. **Eval before optimization.** Build the schema and the eval harness *before* tuning
   prompts or adding the second model. Every change after that is judged by re-running
   the scorecard, never by eyeballing one document.
4. **Small, trustworthy ground truth beats volume.** ~60 hand-verified docs total.
   Do NOT scale to hundreds — it wastes tokens and the labels become untrustworthy.
5. **Provider stays configurable.** All model calls go through `extractor/provider.py`
   (any OpenAI-compatible endpoint). Never hardcode a model or URL elsewhere.
6. **Token discipline.** This runs on free/open models. Cache by image hash, downscale
   images, use constrained JSON output, and only enable the expensive path (second
   model / self-consistency) on full runs — not the smoke loop.
7. **No secrets in git.** Only `.env.example` is committed. Real `.env` is gitignored.

## Architecture (and *why* — put this reasoning in the README)

**Option 1: VLM reader + text-LLM reasoner, with deterministic validation.**

```
image          -> Model A (VLM)  -> {transcription, fields, _confidence}
transcription  -> Model B (LLM)  -> fields (independent re-extraction)
               -> Python validation (arithmetic/format, model-free)
               -> aggregate per-field confidence -> abstention
```

- **Model A = VLM reader.** Receipts are a layout problem: column-aligned line items,
  and "which of these five numbers is the total" is answered by position. Flat OCR
  text loses that, which is why a VLM reads the image rather than an OCR engine.
  It returns the transcription AND the fields in ONE call — a second call for the
  transcript would double vision-token cost for no accuracy gain.
- **The transcription replaces a separate OCR stage.** It is the groundedness source,
  so we keep the anti-hallucination signal without a slow local OCR dependency.
  (PaddleOCR was dropped: too slow on CPU. `extractor/ocr.py` is dead reference code;
  groundedness lives in `extractor/grounding.py`.)
- **Model B = text LLM reasoner.** Does what a VLM is weakest at: cross-section
  reasoning and arithmetic reconciliation. Cheap, text-only. MUST be a different
  model family from A — siblings (e.g. Flash + Flash-Lite) fail the same way on the
  same smudged digit, inflating the agreement signal exactly where it must be honest.
- **KNOWN LIMITATION, state it in the README:** Model B reads Model A's
  transcription, so the two are only PARTIALLY independent — a misread digit
  propagates and inflates their agreement. Mitigation: weight deterministic
  validation heavily (it is model-free and breaks the correlation), and report
  whether confidence still discriminates via AUC/ECE rather than asserting it does.
  True reading independence needs a second VLM on the raw image — document as an
  upgrade path under "what I'd do with more time".

**Confidence aggregates four signals** (`extractor/confidence.py`), ranked by how much
they actually know: (1) `validated` — deterministic arithmetic, model-free, cannot be
bluffed, STRONGEST; (2) `grounded` — value literally present in the transcription;
(3) `agreement` — A and B return the same normalized value; (4) `self_report` — the
model's stated certainty, WEAKEST, because LLM self-confidence is reliably
overconfident. Weights are **fitted** by logistic regression on a held-out calibration
split, per field GROUP (string/money/date/currency), not hand-set and not global.

**Nothing runs on the local CPU** except downscaling, validation and metrics. Both
models are hosted. CPU-only vision/LLM inference is impractical (a 3B model generating
~600 tokens of JSON on CPU is ~40-90s/doc); hosted keeps it ~3-8s/doc, network-bound
and parallelizable. Trade-off to state honestly: neither model is open-weights, so the
test's "self-hosted is a bonus" point is forfeited — justify via the CPU constraint
(the test's own notes recommend Google AI Studio for this case) and note that
LightOnOCR-2-1B or Qwen2.5-VL would be swapped in locally given GPU access.

## Tech stack

- Python 3.11+, Pydantic v2 (typed schema + validation), FastAPI + uvicorn (API),
  `openai` client (generic OpenAI-compatible), Pillow + pypdfium2 (no poppler), httpx, pytest.
- **Model A (VLM):** `gemini-3.6-flash` via Google AI Studio's OpenAI-compatible
  endpoint. Free tier 10 RPM / 250 RPD, no credit card.
- **Model B (LLM):** a different-family text model, e.g.
  `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter (qwen-2.5-7b free was removed). Free slugs rotate — build a
  fallback chain in `provider.py` (`openrouter/free` auto-router survives churn).
- **Cascade base (Phase 6):** `gemini-3.5-flash-lite`, 15 RPM / 1000 RPD — the
  highest free quota, so it absorbs the bulk of the volume.
- No local model inference. Caching by image hash keeps rate-limited calls to one
  per document even across repeated eval runs; re-running eval after a metrics or
  taxonomy change costs ZERO API calls.

## Repo layout

```
/extractor   schema.py, provider.py, extract.py, confidence.py, pipeline.py
/api         FastAPI service (POST /extract, batch runner)
/eval        metrics.py, taxonomy.py, scorecard.py, regression.py, run.py
/data        sample docs + hand-verified ground_truth/*.json
/outputs     committed scorecard + per-doc results from a REAL run
/agent       one custom extension actually used during dev
docker-compose.yml, Dockerfile, .env.example, README.md, CLAUDE.md
```

## Current state

Built and tested (435/435 pytest passing; `python -m agent.offline_e2e` 7/7):
- Phase 0-2: `extractor/schema.py`, `extractor/provider.py`, `extractor/extract.py`
  (single-doc walking skeleton, VLM-only).
- Phase 3: `eval/metrics.py`, `eval/taxonomy.py`, `eval/scorecard.py`, `eval/run.py`,
  `eval/tests/test_metrics.py`. Scorecard now also reports **AUC** (discrimination),
  **ECE** (calibration), a **reliability diagram**, and per-field AUC/ECE.
- Phase 4: `extractor/pipeline.py` (Model A + Model B + signal assembly),
  `extractor/confidence.py` (four signals, fitted per-group logistic calibration,
  risk-coverage curve, `pick_threshold`), `eval/tests/test_confidence.py`.
- `extractor/ocr.py` is the abandoned PaddleOCR path, kept for reference only.
- **Grounding fixed (was anti-informative).** The old `ocr.is_grounded()` was a raw
  substring check on `str(parsed_value)`, so every correct CORD amount (`53280` vs
  printed `53.280`), every non-ISO date and every symbol-printed currency scored as
  ungrounded, while an invented `5` grounded inside `53.280`. Replaced by type-aware,
  whole-token matching in `extractor/grounding.py` (`eval/tests/test_grounding.py`).
  Any confidence numbers produced before this fix are invalid.
- Phase 4 wiring: `eval/run.py` now runs the pipeline by default, passes per-field
  `confidence` into `compare_document`, captures per-doc errors (a failed document is
  scored as empty — a recall failure, never a silent drop from the denominator), and
  has `--calibrate`, which fits the confidence model on the calibration split only,
  reads the threshold off the risk-coverage curve, and writes `outputs/calibration.json`.
  `--source vlm-only` keeps the Phase 3 baseline runnable so "abstention raised
  precision" stays a measured claim. The scorecard reports AUC/ECE and the
  before/after abstention delta.
- Dataset acquisition (see `docs/DATASETS.md`): `scripts/fetch_datasets.sh` +
  `convert_cord.py` / `convert_wildreceipt.py` / `convert_retail.py` / `verify_gt.py` /
  `assign_splits.py` / `degrade.py`, with `eval/tests/test_dataset_converters.py`
  covering normalization, date refusal, row grouping and stratified sampling.

- **Ground truth VERIFIED 60/60** (2026-09-18). The pass found five WildReceipt money
  errors (wild_0003, wild_0019, wild_0018, wild_0015, wild_0004), one of them a defect in
  the published dataset. `scripts/review_server.py` is the fast verification tool: one
  document per screen, only the fields needing judgement plus the disputed arithmetic,
  y/n verdicts written straight to _provenance.json.
- **Review dashboard** (`eval/dashboard.py` + `dashboard.css`): source document beside
  the extraction, type/status filters, pagination, keyboard nav, self-contained HTML with
  images inlined. Built from the same ExtractionOut payload as the API; shows no ground
  truth. `outputs/dashboard.html` is gitignored - it embeds real invoice images.
- **Output contract (`docs/OUTPUT.md`).** `ExtractionOut` now also carries `signals`,
  `validation` (named model-free checks, null = not applicable), `validation_notes` (the
  discrepancy behind each failure) and `transcription`. `extractor/confidence.document_checks`
  produces the named checks from the SAME rules as `validate_arithmetic`, so the two can
  never disagree. `eval/render_doc.py` renders one payload as a self-contained HTML card
  (fields with ok/abstained/absent, line items with the arithmetic footer, validation
  chips, cost/latency, collapsed transcription); it recomputes nothing and shows no
  ground truth. `eval.run` writes `outputs/<run>/documents/<doc_id>.html` per document and
  the scorecard's per-document table links to them. Tests: `eval/tests/test_render_doc.py`.
  NOT built: OUTPUT.md section 2 (one PDF holding a STACK of separate receipts, and
  `/extract` returning a list) — that changes the wire contract for every consumer and is
  a pending decision, not an oversight.
- **Invoices (second document type, docs/DATASETS.md addendum).** 10 real multi-page invoices
  from RVL-CDIP-N (`scripts/fetch_invoices.py`, pinned + SHA-256); the synthetic HF set
  was rejected. `Invoice`/`Party` in `extractor/schema.py`; type comes from the `inv_`
  prefix (`document_type_of`). Pipeline, cascade, API (`document_type` form field) and
  batch (`--type`) are type-aware; dotted nested fields everywhere. Scorecard reports
  invoices under `by_document_type`, never pooled into the receipt headline; calibration
  fits on receipts only (invoice confidence is uncalibrated). Labels are **drafted by the
  assistant, pending human verification** — see `label_notes`; open decisions: replace
  inv_0006 (bundle), inv_0005 struck-through row, inv_0003 redacted number, inv_0008
  "c/o" customer. Tests: `eval/tests/test_invoices.py`.
- Dataset: 60 of 75 documents built and split (CORD 25 + WildReceipt 25 + invoices 10).
  Retail (15) needs a Kaggle token. **Verification is at 60/60** — the ground truth is
  now checked, so scorecards from it can be reported. Human verification caught what the
  audit could not: wild_0003 (subtotal 675.00 for a printed 1,675.00; VAT and Service TAX
  collapsed into one field) and wild_0007 (missing date and currency). inv_0005 keeps its
  struck-through row, since the printed rows sum to the printed total and the
  strike-through is a recipient's annotation. Decisions are in `_provenance.json`
  `verified_note`.
- Schema bug fixed: `Receipt.date` was annotated `Optional[date]`, which the field name
  shadowed, so the annotation resolved to `NoneType` and every date silently became
  null — including in the committed ground truth. Regression test in
  `eval/tests/test_schema.py`.

- **Phase 5 built** (gate verified against a stub OpenAI-compatible endpoint, NOT yet
  against real models): `api/app.py` (`POST /extract`, `GET /health`), `api/batch.py`
  (folder in -> `results.jsonl` + `summary.json/md` with $/doc, tokens, p50/p95),
  `api/service.py` (one entry point for both; per-doc error isolation; config preflight
  so a missing key fails once up front, not once per document), `api/schemas.py`
  (abstained values are WITHHELD from `receipt` at the API boundary and reported under
  `withheld`), `extractor/cost.py` (list-price-equivalent cost from `.env` prices;
  cached calls keep their cost; model latency vs wall latency reported separately).
  The eval scorecard now carries the same cost/latency block.
- Phase 5 bug fixes: the VLM cache was keyed on the image only (a different model or
  edited prompt got stale answers — would have made the Phase 6 before/after measure
  nothing); `prepare_image` named temp files by stem (concurrent `receipt.jpg` uploads
  overwrote each other); percentile used banker's rounding.

- **Phase 6 built, NOT yet measured.** `extractor/cascade.py`: Flash-Lite first,
  escalating to the full two-model pipeline on `arithmetic_failed`, `ungrounded:<f>`,
  `low_confidence:<f>` (present fields only), `empty_extraction`, or a failed cheap
  call. An escalated doc is billed for BOTH paths (the cheap call is sunk). Available as
  `eval.run --source cascade` (scorecard gains a Cascade block: escalation rate, reasons,
  and field accuracy on accepted vs escalated docs, which shows whether the gate is
  leaking) and as `EXTRACTION_MODE=cascade` for the API/batch. `eval/compare.py` produces
  the before/after table ($/doc, tokens, p50/p95, accuracy, AUC/ECE); it refuses runs over
  different document sets and states the one-document noise floor. The machinery was
  verified end to end against a stub; the stub's numbers are not a result.
- **Risk to check first in the real Phase 6 run: `ungrounded:currency`/`date`.** VLMs
  often INFER a currency (IDR from an Indonesian receipt with no "Rp" printed). The
  grounding check correctly flags that, but escalating can't fix it, since Flash will
  infer too. If the reasons table is dominated by these, the cascade escalates most of
  CORD and costs MORE than the baseline. The principled fix is to abstain on (null) an
  ungrounded inferred value rather than escalate it. Decide from the measured reasons
  table, not in advance.

- **Phase 8 files written, Docker NOT verified** (Docker is not installed on the dev
  machine, so the image has never been built). `Dockerfile` (python:3.11-slim,
  poppler-utils, non-root with overridable `APP_UID`, python-based HEALTHCHECK),
  `docker-compose.yml` (`api` on :8000 with `./.cache` bind-mounted so a locally
  fitted confidence model and warm cache are reused; `eval` behind the `tools` profile,
  waits for a healthy api, writes to `./outputs`), `.dockerignore` (excludes `.env`,
  `data/raw`). The API gained `POST /extract/batch` (per-file errors in a 200, upload
  order, duplicate names disambiguated, file cap) and `GET /metrics` (`api/telemetry.py`).
  `eval.run --source api` is real: it preflights `/health`, measures at the SERVER's
  threshold, restores withheld values before scoring, and records server models in meta.
  Verified without Docker by running the image's CMD, HEALTHCHECK command and the eval
  service's command over real HTTP against a stub. First `docker compose up` is untested.

- **Brief gaps closed (2026-09-17):**
  - Phase 7 `eval/regression.py`: exit 0/1/2. Field gate needs an F1 drop AND >= 2
    docs flipped. Any new hallucination fails. Lists newly failing docs with their
    transitions. Cost and p95 are gated; AUC drop only warns.
  - Phase 2b multi-page: `prepare_pages` renders every PDF page via pypdfium2 (no
    poppler) into one VLM call and refuses docs over `MAX_PAGES`. EXIF rotation applied.
    `TruncatedResponse` on a `max_tokens` stop; output budgets raised (Gemini thinking
    tokens share the budget).
  - Table confidence: `line_items` signals (row arithmetic, row grounding, row-level
    agreement) in a `table` group. `metrics.confidence_observations` is the single
    definition used by calibration, abstention and fitting. An abstained table is
    withheld whole.
  - Phase 9 agentic extension: `agent/stub_provider.py` + `agent/offline_e2e.py`,
    shipped as the `offline-e2e` Claude Code skill, mutation-tested.
  - Phase 10: README written; every result marked pending.
  - Fixed along the way: fields with no GT and no prediction scored F1 0 and dragged
    macro F1 (now undefined, excluded); the `--source vlm-only` baseline read env names
    nothing sets.

- **Rubric pass (2026-09-17):**
  - Resilience: `extractor/output.py` makes one text-only JSON repair call (never on
    truncation) and salvages per field. An invalid field gets confidence 0 and is
    abstained; a table with a bad row is abstained whole. A model returning its fields at
    the top level is no longer read as an empty receipt. Fixed: the cascade dropped the
    cheap call's cost when the cheap output was invalid.
  - Comparability: `eval/fingerprint.py` records label/doc/prompt hashes plus models,
    threshold, confidence model and commit. The regression gate exits 2 if labels or
    docs changed (`--allow-label-drift`) and lists config changes.
  - Ground-truth trust: `scripts/audit_gt.py` (model-free label checks, shared
    reconciliation with the pipeline) drives `verify_gt.py --html` ordering. On the REAL
    labels it found: CORD service charge/discount dropped (schema gained `service_charge`
    and `discount`; the reconciliation rule changed accordingly); WildReceipt `Tips_value`
    unmapped; tilted-photo row pairing off by one (now an order-preserving DP,
    `align_column`; a deskew attempt was abandoned because boxes are axis-aligned); cash-
    rounded totals (exact rounding rule, not a looser tolerance). Flags 71 -> 54, docs
    flagged 31 -> 21, `total` flags 15 -> 3. Also surfaced a WildReceipt annotation error
    (wild_0018 `5390` for 5.90), to fix during verification.
  - Converters pin the sampled docs (`--resample` to redraw); verification carries over
    only for byte-identical labels.
  - Scorecard `breakdown`: headline metrics per source and for degraded / not degraded.

**Next up:** hand-verify the 50 labels (start from `verify_gt.py --html`, flagged
first), add the retail tier, create `.env`, then
`--calibrate`, `--set test --source pipeline`, `--set test --source cascade`, and
`eval.compare` for the first REAL scorecard and before/after table. Then Phase 7
(regression diff, which can build on `eval/compare.py`).

**Diagnosing "high confidence but poor accuracy":** separate the two failures before
fixing. AUC ~0.5 means the score carries no information — change the SIGNALS, since
rescaling cannot help. Good AUC with high ECE means the ranking works but the numbers
lie — refit the calibration. Report the before/after (e.g. "ECE 0.23 -> 0.05") in the
README; a measured negative result on a signal is a stronger submission than an
unexamined confidence number.

## Build plan — do phases IN ORDER. Each has a "done when" gate.

- **Phase 1 — Schema.** Flat typed `Receipt` with `line_items`. Absent → null.
  *Done when:* hand-written JSON validates/fails cleanly. ✅ (skeleton)
- **Phase 2 — Walking skeleton.** One image → validated JSON via single VLM call,
  downscaled image, JSON mode, one corrective retry.
  *Done when:* `python -m extractor.extract data/<file>` prints valid JSON. ✅ (skeleton)
- **Phase 2b — Robustness.** Multi-page (stitch/loop pages), PDF handling, absent
  fields. *Done when:* a multi-page doc extracts without crashing.
- **Phase 3 — Ground truth + eval harness (THE DIFFERENTIATOR — do before tuning).**
  See "Eval spec" below. *Done when:* one command runs the extractor over the eval set
  and writes a scorecard + per-doc results to `/outputs`.
- **Phase 4 — Second model + confidence/abstention.** Add Path B, cross-check, combine
  confidence signals, tune abstention on the calibration subset.
  *Done when:* scorecard shows calibration and that abstaining raises precision on kept
  fields vs the Phase 3 baseline.
- **Phase 5 — Production.** FastAPI service, batch/folder with bounded concurrency,
  per-stage tokens+latency+$ logged into the scorecard, per-doc error capture (one bad
  doc never kills the batch). *Done when:* folder in → results + cost/latency summary.
- **Phase 6 — One optimization, measured.** Prefer a **confidence cascade** (cheap
  single path first, escalate to two-model only on low confidence). Alt: image
  downscaling. Measure `$ /doc`, latency p50/p95, AND accuracy delta before/after on the
  *same* eval set. *Done when:* a before/after table exists.
- **Phase 7 — Regression tooling.** Version-tag runs; diff two scorecards field-by-field,
  flag any F1 drop past threshold + list newly-failing docs.
  *Done when:* `eval/regression.py v1 v2` prints a readable diff.
- **Phase 8 — Docker.** `docker compose up` serves the API; `docker compose run --rm eval`
  runs the harness against the API and writes the scorecard to a mounted `/outputs`. Ship
  sample docs + ground truth so it runs out of the box. Commit real output artifacts.
  *Done when:* fresh clone → up → run → scorecard appears.
- **Phase 9 — Agentic extension.** Ship one small tool actually used (e.g. a ground-truth
  scaffolding command, or an MCP tool that runs eval and returns the scorecard). Document
  how it helped.
- **Phase 10 — README.** Run commands; dataset/schema/approach + why; architecture (why
  two models); trade-offs; eval results; optimization impact; agentic note; what you'd do
  with more time. Be honest about limits (DAR is noisy at n≈60).

## Datasets (Phase 3 prerequisite)

See **`docs/DATASETS.md`** for download URLs, converters, and split assignment. Summary:
~65 hand-verified docs from three sources — CORD via HF `naver-clova-ix/cord-v2` (25),
WildReceipt via `https://download.openmmlab.com/mmocr/data/wildreceipt.tar` (25), and
Kaggle `trainingdatapro/ocr-receipts-text-detection` (15, needs an API token, so
manual). Splits: calibration 22 / test 43, with smoke (12) drawn FROM calibration so
iterating never overfits the reported numbers. Nothing is fitted on `test`.

Discard bounding boxes — they exist for layout-model training; our VLM reads the raw
image and the metrics score normalized values. Boxes are used transiently in the
converters for line-item row grouping only.

We have no own scans, so that bonus is forfeited — say so plainly in the README and
note the compensation: three independent sources (two languages, three label schemas)
plus a clearly-labeled synthetic degradation tier.

## Eval spec (Phase 3 — get this right)

- **Ground truth:** ~60 docs total, tiered — ~10–15 smoke, ~50 full, optional ~15 stress.
  Stratified across clean CORD + own messy scans. Script CORD→schema conversion, then
  **hand-verify every label.** Store as `data/ground_truth/*.json`.
- **Normalize before scoring:** dates→ISO, money→Decimal, strip currency/whitespace, case.
  Measure semantic correctness, not string noise.
- **Metrics:** per-field precision/recall/F1 (report exact AND normalized); Document
  Accuracy Rate (all fields correct); line-item matching (align predicted↔GT items, then
  score fields within matches).
- **Error taxonomy (required):** classify each failure as `missing` / `spurious`
  (hallucinated) / `wrong_value` / `swapped_field` / `format_error`. Failures by KIND,
  not just count.
- **Hallucination rate:** fraction of extracted values not groundable in source.
- **Confidence calibration:** bin by confidence, accuracy per bin; coverage/accuracy
  tradeoff (accuracy on the non-abstained fraction).
- **Scorecard:** one command → aggregate JSON + readable HTML/markdown, plus per-doc
  results. Include tokens/latency/$ once Phase 5 lands.

## Commands (define these; keep them stable)

```bash
python -m extractor.extract <path>            # single doc → JSON (VLM-only skeleton)
python -m extractor.pipeline <path>           # single doc → JSON + confidence + abstention
./scripts/fetch_datasets.sh                   # datasets → data/raw (idempotent)
python -m eval.run --calibrate                # fit confidence on calibration → threshold
python -m eval.run --set smoke|test|degraded  # run harness → scorecard in /outputs
python -m eval.run --set test --source vlm-only   # Phase 3 baseline, for the comparison
python -m eval.run --set test --source cascade    # Phase 6 optimization
python -m eval.compare <baseline_run> <optimized_run> --out table.md   # before/after
uvicorn api.app:app --port 8000               # serve POST /extract, GET /health
python -m api.batch <folder> --concurrency 3  # folder -> results.jsonl + cost/latency summary
python -m eval.regression <runA> <runB>       # ship gate: exit 0 ok / 1 regression / 2 invalid
python -m agent.offline_e2e                   # whole system over HTTP vs a stub, no keys (skill: offline-e2e)
docker compose up                             # serve extraction API on :8000
docker compose run --rm eval                  # run harness against API → ./outputs
SET=smoke VERSION=v2 docker compose run --rm eval   # other set / version tag
python -m eval.run --source api --set smoke   # score an already-running API from the host
```

## Conventions

- Type everything. Pydantic for all model I/O. `Decimal` for money, never float.
- Pure functions in `eval/metrics.py` (easy to unit-test); write pytest tests for
  normalization and the taxonomy classifier — these are correctness-critical.
- Keep the extraction logic and eval harness OUR OWN and readable. Reviewers may ask
  for a live walkthrough and a small live change — favor clarity over cleverness.
- Log tokens/latency on every provider call. Never silently swallow model errors.

## Git discipline (this is graded — history should show the thought process)

- Commit at each phase's "done when" gate, not once at the end of a session. If a
  phase has sub-steps (e.g. Phase 3: taxonomy → metrics → scorecard → runner), commit
  each as it lands rather than batching the whole phase into one commit.
- Message format: `<phase>: <what, in a few words>` — e.g. `phase3: add error
  taxonomy (missing/spurious/wrong_value/format_error/swapped)`,
  `phase4: cross-check VLM vs OCR+LLM paths for confidence`,
  `phase6: add confidence-cascade optimization, measured -40% cost/doc`.
  A reviewer should be able to read `git log --oneline` and reconstruct the build
  order without opening the diffs.
- Commit a failing test or a known-broken intermediate state if that's genuinely
  where a working session ended — don't rewrite history to look cleaner than the
  process was. Squashing to hide false starts defeats the point of this requirement.
- Never bundle unrelated changes (e.g. a metrics fix + a README rewrite) in one
  commit just because they landed in the same sitting.
- Before a submission-ready commit: rerun `pytest eval/tests -q` and `eval.run
  --set smoke` so the tip of the branch is never in a known-broken state.

## Do NOT

- Do not build extraction fully before the eval harness.
- Do not add fields the schema doesn't define; do not fill absent fields with guesses.
- Do not scale the eval set past ~60 docs or trust dataset labels without verifying.
- Do not hardcode providers/models/keys outside `provider.py` + `.env`.
- Do not commit real secrets or large raw datasets — only the curated sample + GT.
- Do not squash meaningful development history into one commit before submitting.

## Definition of done (maps to the grading criteria)

Trustworthy ground truth ✔ · failures understood by kind ✔ · comparable across versions
(regression diff) ✔ · accurate on messy docs, tables + cross-section fields, no
hallucination ✔ · validated output + meaningful confidence/abstention ✔ · cost/latency
visible + one measured optimization ✔ · a real agentic extension ✔ · clean code + clear
README ✔.