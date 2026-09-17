---
name: offline-e2e
description: Run the receipt system end to end over real HTTP against a deterministic stub model server, with no API keys. Use after changing anything in extractor/, api/, eval/run.py or eval/regression.py, and before claiming such a change works. Also use to diagnose a pipeline, cache, cascade, multi-page, API or scorecard problem when real models are unavailable. Not a substitute for a real accuracy run.
---

# Offline end-to-end check

Unit tests in `eval/tests` mock at function boundaries. This check mocks at the NETWORK
boundary instead: `agent/stub_provider.py` is an OpenAI-compatible server, and everything
above the socket is production code (provider, cache, grounding, confidence, cascade,
FastAPI, the eval client, the regression gate).

## Run it

```bash
python -m agent.offline_e2e          # exit 0 = all checks passed
E2E_VERBOSE=1 python -m agent.offline_e2e   # tracebacks on failure
```

~15 seconds. It starts both servers on free ports in background threads and writes only
to a temp directory. It never touches `outputs/`, `.cache/` or `.env`.

## When to use it

- After editing `extractor/`, `api/`, `eval/run.py` or `eval/regression.py`: run
  `pytest eval/tests -q` AND this. Several bugs in this repo passed every unit test and
  were caught only here (see "What it has caught").
- When there are no API keys and a change still needs to be seen working.
- Before telling the user something works. Report the check lines, not a summary.

## Reading a failure

Each check line names the expectation that broke:

| check | a FAIL usually means |
|---|---|
| pipeline | grounding regressed (printed values scored as hallucinated), a stage went unpriced, or the table lost its confidence |
| cache | a rerun made model calls (the cache key changed, or stopped including model and prompt), or cached runs lost their cost |
| multi-page | pages sent as separate calls, or only page 1 sent |
| truncation | a `max_tokens` stop is being parsed as JSON instead of raising `TruncatedResponse` |
| cascade | the accept/escalate gate is wrong, or an escalated doc is not billed for both paths |
| api | an abstained value leaked into `receipt`, a bad file sank a batch, or `/metrics` miscounts |
| eval + gate | `--source api` or the regression gate broke, or an identical run is flagged as a regression |

Fix the cause, rerun, and paste the new check lines. Don't edit the check to make it pass
unless the expectation itself is wrong, and if it is, say so explicitly.

## What it does NOT prove

The stub returns the same receipt for every image, so accuracy, calibration and cascade
savings from a stub run are meaningless. Never report them as results. Real numbers come
only from `python -m eval.run` against real models.

## Extending it

When a bug is found that this check would have caught, add a check to
`agent/offline_e2e.py` asserting the behaviour, and a model name to
`agent/stub_provider.py` if new stub behaviour is needed. Then prove the check can fail by
reintroducing the bug once, as was done for grounding and cascade billing.

## What it has caught

Found by running the system against the stub while it was built, before this was
packaged as a skill:

- `grounded` was a raw substring match: every correct Indonesian total (`53280` vs
  printed `53.280`) scored as hallucinated, and an invented `5` scored as grounded.
- On Windows, `localhost` added ~4 s to every model call (IPv6 fallback), silently
  inflating every latency number. Use `127.0.0.1`.
- A missing API key was recorded as the same `KeyError` against every document instead
  of failing once up front.
- `--calibrate` recommended a threshold immediately after reporting that the target
  precision was unreachable.
- The cascade escalated 100% of documents. It turned out the fixture's model was returning
  values absent from its own transcription, which confirmed the gate was right to refuse them.

Mutation-tested after packaging: reintroducing the substring grounding fails `pipeline`
and `cascade`; dropping the cascade's sunk-cost billing fails `cascade`.
