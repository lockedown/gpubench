# GPU Benchmark — Execution Harnesses (gpubench)

Benchmark execution tooling for the ATC GPU benchmark engagement, implementing outcomes
from the **GPU Benchmark Methodology v1.3** (O1 — KV cache memory scaling, O2-lite —
co-hosting isolation, O3-lite — quantization quality gate, O9 — model cold-start). Runs identically on the on-prem nodes
(8× MI300X, 8× H200 DGX) and cloud (AWS p5e) — platform is auto-detected and every
result row is emitted in the engagement's Appendix B manifest format.

> **Status (06 Aug 2026):** Both nodes active. The DGX H200 returned to service after
> the rack-power fix and completed the full O1 sweep (15 cells, ~5h sustained load) with
> no power event.

## Contents

| File | What it is |
|---|---|
| `run_o1_benchmark_v2.py` | O1 harness (`wwt-o1-harness/1.3.4-kvdtype`). Closed-loop dual-ceiling search: interactive (SLO) ceiling at P99 TTFT ≤ 1.5s + memory ceiling, per (model × context × KV-mode) cell. Runs **inside** the pinned vLLM container. |
| `run_o9_benchmark.py` | O9 orchestrator (`wwt-o9-harness/1.3-quiesce`). Runs on the **host** (stdlib only, needs sudo for cold cells): waits for GPUs to quiesce, drops page cache, meters NVMe bytes read, launches the probe per (model × cache-state × repeat). Resolves `o9_probe.py` from its own directory (bind-mounted into the container) — workdir is outputs-only. |
| `o9_probe.py` | O9 in-container probe. Times import → engine build → first inference → first-100 latency curve. Launched by the orchestrator; not run directly. |
| `test_o9.py` | Offline test suite for the O9 pair (fake docker/diskstats — runs anywhere, no GPUs). |
| `run_o2_cohost.py` | O2-lite orchestrator (`wwt-o2-harness/1.3-crossvendor`). Runs on the **host** (stdlib): two models co-resident on shared GPUs (memory partition, TP4 each), served as endpoints; siloed ceilings → 50/50 → noisy-neighbour rotation → isolation scores. Same-GPU co-residency proved unconfigurable on the ROCm dev build (V1 memory pre-check) — the CUDA engine (0.22.1) may permit it; co-host phases are worth re-attempting on the DGX. |
| `run_o3_quality.py` | O3-lite quality gate (`wwt-o3lite-harness/1.2-crossvendor`). Runs on the **host**: serves each model (TP8) and runs lm-eval (GSM8K 8-shot, MMLU 5-shot) against the endpoint via a local tokenizer. Needs the `o3eval` image (Dockerfile in the script header — base it on `python:3.12-slim` with `transformers<5`, **not** the vLLM image, whose transformers breaks lm-eval 0.4.9 at import; the image is CPU-only and platform-neutral). |
| `test_o2_o3.py` | Offline test suite for O2-lite + O3-lite (fake SSE vLLM server + fake docker — no GPUs). |
| `check_redhat_dsv4_integrity.py` | Model-artifact integrity gate (offline, no GPU): verifies MoE router tensors (`tid2eid`) against the original checkpoint, cross-checks key sets, writes evidence to `redhat_dsv4_integrity.json`. Exit 0 pass / 2 fail / 3 indeterminate. Ran 05 Aug: RedHat DSv4-BF16 **FAILED**. |
| `requirements.txt` | Python deps for a bare-metal (venv) install of the O1 harness. **Not needed for container execution** — the pinned vLLM image carries everything. |
| `O1_Runbook_AWS_p5e.docx` | Runbook for executing O1 natively on AWS (EC2 p5e.48xlarge), per the customer-comparison scope. |

Related documents (engagement artefact repo / SharePoint, not in this repo):
GPU Benchmark Methodology v1.3 + change logs · O1 Execution Runbook v1.1 (on-prem) ·
O1 mini-sweep & combined analysis workbooks · standup exec summary.

## Quick start — MI300X node

```bash
# Pin the image once, use $IMG everywhere (convention 1 — nightly has drifted twice)
IMG=vllm/vllm-openai-rocm@sha256:3a83d5c1e0ee537cd123095f3ac2b78ffa2236701722569f464af8870b8c0393

# O1: validate, then run (inside the pinned ROCm vLLM container)
sudo docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  --ipc=host --shm-size=32g \
  -v /opt/huggingface-cache:/root/.cache/huggingface \
  -v ~/dl_script/gpubench:/work -w /work \
  -e HF_HUB_OFFLINE=1 --entrypoint python3 \
  $IMG run_o1_benchmark_v2.py --validate-only

# O1 sweep — detached (never run attached to your SSH TTY)
sudo docker rm -f o1run 2>/dev/null
sudo docker run -d --name o1run <same flags> \
  $IMG run_o1_benchmark_v2.py \
  --models qwen --profile standard --context-lengths 2048,32768,131072 \
  --operator <you> --out /work/o1_results.csv
sudo docker logs -f --tail 50 o1run

# O9 cold-start — on the HOST, in tmux (sudo needed for cache drops)
sudo python3 run_o9_benchmark.py --image $IMG \
  --models qwen,qwen-fp8,deepseek --repeats 2 \
  --workdir ~/dl_script/gpubench --operator <you> --out o9_results.csv

# O3-lite quality gate — host; needs the o3eval image (script header)
python3 run_o3_quality.py --serve-image $IMG \
  --eval-image wwt/vllm-bench:o3eval --models qwen,qwen-fp8 \
  --operator <you> --out o3_quality.csv

# O2-lite co-hosting — host, tmux
sudo python3 run_o2_cohost.py --image $IMG \
  --pair qwen-fp8,qwen --gpus 0,1,2,3 --operator <you> --out o2_results.csv
```

H200/DGX: identical commands with `--gpus all` replacing the AMD device flags
(O2/O3/O9: add `--nvidia` — ROCm-specific model env is dropped automatically),
image `wwt/vllm-bench:o1`. O9's probe is resolved from the script's own directory,
not the workdir. AWS: see the p5e runbook docx.

## Conventions that keep results admissible

1. **Pin images by digest, never by tag.** `nightly`/`latest` have already drifted
   once mid-engagement. `docker inspect --format='{{index .RepoDigests 0}}' <image>`
   → record as `container_image_sha` in manifests.
2. **Check `harness_version` on every CSV row** before trusting a result — it is the
   tamper-evident guard against running a stale script (this has happened; the column
   caught it). Current: O1 `1.3.4-kvdtype`, O2 `1.3-crossvendor`, O3-lite
   `1.2-crossvendor`, O9 `1.3-quiesce`.
3. **One harness per node at a time.** O1/O2/O3/O9 all assume an exclusive,
   quiesced node — overlapping them (or launching before the previous cell's
   container releases GPU memory) kills engine startup with an opaque
   "Engine core initialization failed". O9 now waits for GPUs to drain
   between cells; for the others, check `nvidia-smi`/`rocm-smi` shows ~0 MB
   used and `docker ps` shows no serve containers before launching.
4. **Detached execution only.** `docker run -d --name ...` (no `--rm`, no `-it`) or
   tmux. SSH/VPN drops have masqueraded as "node crashes" twice.
5. **Resume is automatic.** Both harnesses append to CSV and skip completed cells on
   re-launch — after any failure, re-run the identical command.
6. **No live downloads during measurement** (`HF_HUB_OFFLINE=1` is forced). Mirror
   weights beforehand into `/opt/huggingface-cache`.
7. **Model-specific engine requirements are encoded in the harness**, not in runbooks
   or memory: DeepSeek-V4 needs fp8 KV cache + `VLLM_ROCM_USE_AITER=1` (ROCm) +
   engine-default KV block size. Rows record these in `notes`. Under `--nvidia`,
   ROCm-specific env (`VLLM_ROCM_*`) is dropped automatically (O2/O3/O9; O1
   auto-detects platform).

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
python3 test_o9.py && python3 test_o2_o3.py && python3 test_o1_harness.py
```

## Known issues / open items

- DGX H200 power: RESOLVED 06 Aug — rack fix applied; node completed the full O1
  sweep (~5h sustained load) with no power event. NVSM service state still worth a
  check before long campaigns. (History: hard power-offs Jul 24 + Jul 27.)
- Aug H200 O1: both FP8 2048 cells failed the CoV repeatability gate (ceilings
  consistent at 132 regardless) — treat FP8 as 131-132 borderline.
- DeepSeek below TP8 is a ROCm/AITER kernel-shape constraint — untested on CUDA;
  re-attempt with the O2 pair on the DGX.
- Model SHA-256s unpinned (`--verify-sha` inert until Appendix A values are set).
- **RedHatAI/DeepSeek-V4-Flash-BF16 FAILED the integrity gate (05 Aug 2026)** — router
  tensors (`tid2eid`) all-zeros vs the original checkpoint (`check_redhat_dsv4_integrity.py`,
  evidence in `redhat_dsv4_integrity.json`). Do NOT pin, serve, or benchmark this artifact.
  June MI300X DeepSeek-BF16 rows are quarantined in the combined analysis workbook. A
  replacement BF16 baseline (integrity-checked) is needed for the roster.
- Cross-vendor engine versions differ (0.22.1 CUDA vs 0.23.1rc1-dev ROCm) — recorded
  per row; parity decision pending before final cross-vendor claims.
- Decode-health memory predicate replacement (methodology v1.3.1) pending standup.

---
*World Wide Technology — ATC GPU Benchmark engagement. Engagement-internal.*
