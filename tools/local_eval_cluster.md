# Local evaluation cluster

`local_eval_cluster.sh` captures the exact local 27B + PP-OCRv6 environment
used for the frozen 60-case factuality evaluation. It manages four independent
model endpoints on GPUs 3–6 and four business APIs, one request per API, so separate
resumes can run in parallel without increasing one deployment's KV-cache or
application-memory budget.

## Normal workflow

```bash
# Read-only environment diagnosis. Safe while another run is active.
bash tools/local_eval_cluster.sh preflight
bash tools/local_eval_cluster.sh status

# Start missing model containers, then start APIs from the current source.
bash tools/local_eval_cluster.sh up

# Cheap gate before a full run.
bash tools/local_eval_cluster.sh eval-case HV2-S1-012 3

# Run a fixed diagnostic subset under one named content profile.
LOCAL_EVAL_PIPELINE_PROFILE=f507_compatible \
  bash tools/local_eval_cluster.sh restart
LOCAL_EVAL_PIPELINE_PROFILE=f507_compatible \
  bash tools/local_eval_cluster.sh eval-subset f507-compatible-12 \
  HV2-S1-003,HV2-S1-010,HV2-S1-012,HV2-S2-003

# Experimental Composer-preserving candidate. It is not a production default:
# the grouped scan cases showed that OCR can corrupt record ownership before
# the fact ledger is built, so OCR requests skip record-level recovery.
LOCAL_EVAL_PIPELINE_PROFILE=quality_v2 \
  bash tools/local_eval_cluster.sh restart

# Five 12-case shards, followed by deterministic merge.
bash tools/local_eval_cluster.sh eval-plan
bash tools/local_eval_cluster.sh eval-full fcv1-r7

# Reload application code after editing Python files. Models stay resident.
bash tools/local_eval_cluster.sh restart

# Stop only the API processes owned by this helper. Models remain loaded.
bash tools/local_eval_cluster.sh api-down
```

Stopping the model containers is intentionally separate:

```bash
bash tools/local_eval_cluster.sh models-down
```

## Built-in guards

- `api-up` refuses any occupied API port that it did not create. It never
  silently adopts or kills an old process.
- A digest of `main.py` and all top-level `core/*.py` files is recorded when
  APIs start. `eval-case` and `eval-full` reject `stale-code` processes after
  source changes.
- Effective process environments are checked for `LLM_CONTEXT_WINDOW=16384`,
  the requested `PIPELINE_PROFILE`, and the requested `FACT_COMPILER_MODE`.
- All model, tokenizer, OCR and frozen dataset paths are validated before
  startup. The 60 cases and 60 annotations must both be present.
- Full evaluation uses `--timeout 480 --max-attempts 1`, creates disjoint
  modulo shards, and merges only a complete comparable result.
- API processes run in owned process groups, so `api-down` also terminates
  their OCR children. It never targets an unrecorded listener.

Runtime PID files and logs live under `.codex/research-loop/runtime/` and
evaluation artifacts under `.codex/research-loop/artifacts/`; both are already
excluded from Git in this workspace. Paths and ports can be overridden with
the `LOCAL_EVAL_*` environment variables shown by `--help`.
