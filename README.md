# GPU Benchmark — Execution Harnesses (gpubench)

Benchmark execution tooling for the ATC GPU benchmark engagement, implementing outcomes
from the **GPU Benchmark Methodology v1.3** (O1 — KV cache memory scaling, O9 — model
cold-start; further outcomes in development). Runs identically on the on-prem nodes
(8× MI300X, 8× H200 DGX) and cloud (AWS p5e) — platform is auto-detected and every
result row is emitted in the engagement's Appendix B manifest format.

> **Status (29 Jul 2026):** MI300X is the active node. The H200 DGX is blocked on a
> rack-power investigation — do not schedule sustained load on it until cleared.

## Contents

| File | What it is |
|---|---|
| `run_o1_benchmark_v2.py` | O1 harness (`wwt-o1-harness/1.3.4-kvdtype`). Closed-loop dual-ceiling search: interactive (SLO) ceiling at P99 TTFT ≤ 1.5s + memory ceiling, per (model × context × KV-mode) cell. Runs **inside** the pinned vLLM container. |
| `run_o9_benchmark.py` | O9 orchestrator (`wwt-o9-harness/1.0`). Runs on the **host** (stdlib only, needs sudo for cold cells): drops page cache, meters NVMe bytes read, launches the probe per (model × cache-state × repeat). |
| `o9_probe.py` | O9 in-container probe. Times import → engine build → first inference → first-100 latency curve. Launched by the orchestrator; not run directly. |
| `test_o9.py` | Offline test suite for the O9 pair (fake docker/diskstats — runs anywhere, no GPUs). |
| `requirements.txt` | Python deps for a bare-metal (venv) install of the O1 harness. **Not needed for container execution** — the pinned vLLM image carries everything. |
| `O1_Runbook_AWS_p5e.docx` | Runbook for executing O1 natively on AWS (EC2 p5e.48xlarge), per the customer-comparison scope. |

Related documents (engagement artefact repo / SharePoint, not in this repo):
GPU Benchmark Methodology v1.3 + change logs · O1 Execution Runbook v1.1 (on-prem) ·
O1 mini-sweep & combined analysis workbooks · standup exec summary.

## Quick start — MI300X node

```bash
# O1: validate, then run (inside the pinned ROCm vLLM container)
sudo docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  --ipc=host --shm-size=32g \
  -v /opt/huggingface-cache:/root/.cache/huggingface \
  -v ~/dl_script/gpubench:/work -w /work \
  -e HF_HUB_OFFLINE=1 --entrypoint python3 \
  vllm/vllm-openai-rocm:nightly run_o1_benchmark_v2.py --validate-only

# O1 sweep — detached (never run attached to your SSH TTY)
sudo docker rm -f o1run 2>/dev/null
sudo docker run -d --name o1run <same flags> \
  vllm/vllm-openai-rocm:nightly run_o1_benchmark_v2.py \
  --models qwen --profile standard --context-lengths 2048,32768,131072 \
  --operator <you> --out /work/o1_results.csv
sudo docker logs -f --tail 50 o1run

# O9 cold-start — on the HOST, in tmux (sudo needed for cache drops)
sudo python3 run_o9_benchmark.py --image vllm/vllm-openai-rocm:nightly \
  --models qwen,qwen-fp8,deepseek --repeats 2 \
  --workdir ~/dl_script/gpubench --operator <you> --out o9_results.csv
```

H200/DGX: identical commands with `--gpus all` replacing the AMD device flags
(O9: add `--nvidia`), image `wwt/vllm-bench:o1`. AWS: see the p5e runbook docx.

## Conventions that keep results admissible

1. **Pin images by digest, never by tag.** `nightly`/`latest` have already drifted
   once mid-engagement. `docker inspect --format='{{index .RepoDigests 0}}' <image>`
   → record as `container_image_sha` in manifests.
2. **Check `harness_version` on every CSV row** before trusting a result — it is the
   tamper-evident guard against running a stale script (this has happened; the column
   caught it). Current: O1 `1.3.4-kvdtype`, O9 `1.0`.
3. **Detached execution only.** `docker run -d --name ...` (no `--rm`, no `-it`) or
   tmux. SSH/VPN drops have masqueraded as "node crashes" twice.
4. **Resume is automatic.** Both harnesses append to CSV and skip completed cells on
   re-launch — after any failure, re-run the identical command.
5. **No live downloads during measurement** (`HF_HUB_OFFLINE=1` is forced). Mirror
   weights beforehand into `/opt/huggingface-cache`.
6. **Model-specific engine requirements are encoded in the harness**, not in runbooks
   or memory: DeepSeek-V4 needs fp8 KV cache + `VLLM_ROCM_USE_AITER=1` (ROCm) +
   engine-default KV block size. Rows record these in `notes`.

## Reading O1 results

- `concurrent_sessions` — interactive (SLO) ceiling: max sessions with P99 TTFT ≤ 1.5s.
  **0 at long context is a valid, expected result** (single prefill exceeds the bound).
- `concurrent_sessions_memory` + `memory_ceiling_reason` — KV-capacity ceiling.
  ⚠ Under methodology v1.3 the decode-health trigger dominates and this column is
  **not usable as KV capacity** (replacement to OOM/preemption-only is pending as
  v1.3.1); the exception is `catastrophic queueing`, which is a genuine pool signal.
- `kv_bytes_per_session_config` — derived from the model's config.json. Flagged
  approximate for DeepSeek (MLA fields not parsed; GQA formula used).
- `notes` — bounds, stagger, profile, engine overrides, prompt-trim and borderline
  flags. Read it before quoting any row.

## Testing

`test_o9.py` runs anywhere (no GPUs, fake docker). The O1 suite
(`test_o1_harness.py`, 25 checks, mocked torch/vllm/engine) lives in the artefact
repo — **add it here**; both suites gate any harness change:

```bash
python3 test_o9.py && python3 test_o1_harness.py
```

## Known issues / open items

- DGX H200: two hard power-offs under load (Jul 24, Jul 27) — BMC SEL review + rack
  power audit open with ATC ops. NVSM services also failing on that node.
- Model SHA-256s unpinned (`--verify-sha` inert until Appendix A values are set).
- Cross-vendor engine versions differ (0.22.1 CUDA vs 0.23.1rc1-dev ROCm) — recorded
  per row; parity decision pending before final cross-vendor claims.
- Decode-health memory predicate replacement (methodology v1.3.1) pending standup.

---
*World Wide Technology — ATC GPU Benchmark engagement. Engagement-internal.*
