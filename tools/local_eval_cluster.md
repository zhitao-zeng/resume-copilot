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

# Reproduce an immutable historical control without copying its local dataset.
# The business API imports source only from that worktree; the evaluator and
# hash-locked cases remain owned by this repository.
LOCAL_EVAL_SOURCE_ROOT=/path/to/read-only/worktree \
  LOCAL_EVAL_RESUME_PIPELINE_VERSION=v2 \
  bash tools/local_eval_cluster.sh restart

# Cheap gate before a full run.  NOTE: case IDs rotate as holdout cases are
# retired (HV2-S1-012 etc. left the split in R27) — pick a live one first:
#   head -1 validation_sets/public_resume_holdout/holdout_v2/cases.jsonl
bash tools/local_eval_cluster.sh eval-case "$(.venv/bin/python -c "import json;print(json.loads(open('validation_sets/public_resume_holdout/holdout_v2/cases.jsonl').readline())['id'])")" 3

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

# Opt-in V3 compiler. The version is part of process ownership and result
# identity, so a V2 API cannot be silently reused for this run.
LOCAL_EVAL_RESUME_PIPELINE_VERSION=v3 \
  LOCAL_EVAL_PIPELINE_PROFILE=f507_compatible \
  bash tools/local_eval_cluster.sh restart
LOCAL_EVAL_RESUME_PIPELINE_VERSION=v3 \
  LOCAL_EVAL_PIPELINE_PROFILE=f507_compatible \
  bash tools/local_eval_cluster.sh eval-subset v3-representative \
  HV2-S1-003,HV2-S1-010,HV2-S1-012,HV2-S2-003

# Run the same V3 contract against the DeepSeek Local Flash ceiling/schema probe.
# The private env file is sourced into the managed process only; its values are
# never stored in cluster.env or evaluator artifacts.  Use a separate runtime
# and port so the local Qwen APIs remain available for a controlled A/B.
LOCAL_EVAL_MODEL_PROVIDER=deepseek-local \
  LOCAL_EVAL_SECRET_ENV_FILE=/home/zengzhitao/.config/claude-router/env \
  LOCAL_EVAL_INSTANCE_COUNT=1 \
  LOCAL_EVAL_API_PORT_BASE=18189 \
  LOCAL_EVAL_RUNTIME_DIR=.codex/research-loop/runtime/v3-deepseek-ceiling \
  LOCAL_EVAL_RESUME_PIPELINE_VERSION=v3 \
  LOCAL_EVAL_PIPELINE_PROFILE=f507_compatible \
  bash tools/local_eval_cluster.sh up
LOCAL_EVAL_MODEL_PROVIDER=deepseek-local \
  LOCAL_EVAL_SECRET_ENV_FILE=/home/zengzhitao/.config/claude-router/env \
  LOCAL_EVAL_INSTANCE_COUNT=1 \
  LOCAL_EVAL_API_PORT_BASE=18189 \
  LOCAL_EVAL_RUNTIME_DIR=.codex/research-loop/runtime/v3-deepseek-ceiling \
  LOCAL_EVAL_RESUME_PIPELINE_VERSION=v3 \
  LOCAL_EVAL_PIPELINE_PROFILE=f507_compatible \
  bash tools/local_eval_cluster.sh eval-case HV2-S2-003 0

# Synthetic/local holdout only: retain the exact schema inputs and outputs for
# cross-model schema/error analysis.  Production remains opt-out by default.
LOCAL_EVAL_V3_TRAINING_TRACE_ENABLED=1 \
LOCAL_EVAL_V3_TRAINING_TRACE_DIR=.codex/research-loop/artifacts/v3-training-traces \
  bash tools/local_eval_cluster.sh restart

# V3 scan-layout candidate.  Use the isolated CPU Paddle environment locally;
# production Docker has a separate explicit GPU configuration.
LOCAL_EVAL_LAYOUT_ORDER_ENGINE=ppstructure_hybrid \
  LOCAL_EVAL_PPSTRUCTURE_PYTHON=/path/to/ppstructure-venv/bin/python \
  LOCAL_EVAL_PPSTRUCTURE_MODEL_DIR=/path/to/official_models \
  LOCAL_EVAL_PPSTRUCTURE_DEVICE=cpu \
  bash tools/local_eval_cluster.sh restart

# The reproducible GPU path uses the interpreter-compatible Docker adapter.
# Each API is pinned to the corresponding LOCAL_EVAL_GPU_IDS entry; with the
# default 3,4,5,6 assignment it does not touch GPU 0--2.
LOCAL_EVAL_LAYOUT_ORDER_ENGINE=ppstructure \
  LOCAL_EVAL_PPSTRUCTURE_PYTHON="$PWD/tools/ppstructure_docker_python.sh" \
  LOCAL_EVAL_PPSTRUCTURE_MODEL_DIR="$PWD/models_slim/ppstructure-v3/official_models" \
  LOCAL_EVAL_PPSTRUCTURE_DEVICE=gpu:0 \
  LOCAL_EVAL_PPSTRUCTURE_DOCKER_IMAGE=resume-copilot:ppstructure-gpu-sharedcuda-local \
  LOCAL_EVAL_GPU_IDS=3,4,5,6 \
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
- A digest of `main.py` and every Python file below `core/` is recorded when
  APIs start, including `core/v3/*.py`;
  `eval-case` and `eval-full` reject `stale-code` processes after source changes.
- `LOCAL_EVAL_SOURCE_ROOT` is part of process ownership. It allows an exact
  historical API worktree to be evaluated with the current hash-locked
  evaluator/data without importing current product code into that process.
- Effective process environments are checked for `LLM_CONTEXT_WINDOW=16384`,
  the requested `PIPELINE_PROFILE`, and the requested `FACT_COMPILER_MODE`.
  `RESUME_PIPELINE_VERSION`, model provider, endpoint and model name are checked
  as well, so a Qwen API cannot be silently reused as a DeepSeek result.
- `qwen27b` remains the default and owns only the named local vLLM containers.
  `deepseek-local` is evaluation-only; model lifecycle commands never
  start or stop the shared remote service.
- All model, tokenizer, OCR and frozen dataset paths are validated before
  startup. The 60 cases and 60 annotations must both be present.
- Full evaluation uses `--timeout 480 --max-attempts 1`, creates disjoint
  modulo shards, and merges only a complete comparable result.
- V3 realization starts only when the request has at least 240 seconds left
  by default. `LOCAL_EVAL_V3_REALIZER_MIN_REMAINING_SECONDS` makes this gate
  explicit in controlled timing experiments; otherwise exact deterministic
  realization is used without risking the 480-second request deadline.
- Independent semantic batches use two bounded in-flight calls by default,
  matching the local vLLM `max-num-seqs=2` limit. A controlled timing run can
  override this with `LOCAL_EVAL_V3_SEMANTIC_CONCURRENCY=1|2`; the value is
  process-identity checked so sequential and parallel results cannot mix.
- Semantic batch shape is also part of the recorded process identity. Use
  `LOCAL_EVAL_V3_SEMANTIC_BATCH_FACTS` and
  `LOCAL_EVAL_V3_SEMANTIC_BATCH_CHARS` for cross-model contract A/B runs;
  the calibrated defaults are `28` facts and `9000` serialized characters. Smaller
  batches are an evaluation of schema adherence, not a change to fact policy.
- For each material V3 compiler candidate, retain a same-input, same-schema,
  same-batch Qwen27B/DeepSeek Local Flash pair. Also retain a second DeepSeek
  run at its provider-optimized batch shape when that differs. Report the two
  DeepSeek views separately: the first isolates model behavior, while the
  second probes a practical latency/quality ceiling. DeepSeek is not assumed
  to be a teacher or a guaranteed upper bound: schema failures and unsupported
  facts remain first-class results. Credentials stay in `SECRET_ENV_FILE`;
  neither artifacts nor commands may serialize them.
- `LOCAL_EVAL_LAYOUT_ORDER_ENGINE=ppstructure` and `ppstructure_hybrid` require an explicit isolated
  interpreter and a complete four-model offline bundle.  The interpreter,
  model root, device and worker timeout are recorded and process-identity
  checked.  Local runs default to CPU so they cannot silently consume GPU 0--2;
  production Docker supplies its own explicit GPU device.
- `ppstructure` uses StructureV3 layout and recognized text.  The safer
  `ppstructure_hybrid` candidate uses only its region/order geometry and keeps
  PP-OCRv6 as the sole text source; any layout failure returns to BBOX ordering.
- For the local GPU candidate, `tools/ppstructure_docker_python.sh` preserves
  the worker-interpreter contract while running the four offline models in an
  immutable Paddle image.  The launcher passes one explicit allowed GPU per API
  and validates the image before startup; the wrapper rejects any GPU outside
  `LOCAL_EVAL_PPSTRUCTURE_DOCKER_ALLOWED_GPUS`.
- API processes run in owned process groups, so `api-down` also terminates
  their OCR children. It never targets an unrecorded listener.

Runtime PID files and logs live under `.codex/research-loop/runtime/` and
evaluation artifacts under `.codex/research-loop/artifacts/`; both are already
excluded from Git in this workspace. Paths and ports can be overridden with
the `LOCAL_EVAL_*` environment variables shown by `--help`.

The launcher defaults to the frozen 60-case holdout and refuses a different
line count. For another frozen split, set `LOCAL_EVAL_CASES`,
`LOCAL_EVAL_ANNOTATIONS` and `LOCAL_EVAL_EXPECTED_CASE_COUNT` together. The
expected count must be a positive integer and must match both files, so a
partial copy cannot silently pass as a complete evaluation set.
