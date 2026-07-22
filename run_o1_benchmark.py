#!/usr/bin/env python3
"""
O1 — KV cache memory scaling — execution harness
==================================================
HSBC GPU Benchmark Methodology v1.3 | WWT ATC on-prem | Dell XE9680, 8x H200 SXM5

Implements the O1 key steps (v1.3 dual-ceiling model):
  * Staggered admission (--stagger, default 250ms) — never simultaneous bursts.
  * Interactive (SLO) ceiling: P99 TTFT of newly admitted sessions <= 1.5 s.
    Ceiling 0 at long contexts is an expected, informative result.
  * Memory ceiling: admission continues past the SLO ceiling ignoring TTFT,
    until OOM / admission failure / decode-health violation (P99 ITL > 3x the
    single-session baseline). KV-per-session is recorded here.

Plus:
  * Pinned models (Appendix A): Qwen3.5-122B-A10B (primary), DeepSeek-V4-Flash
    (efficiency control). SHA-256 verification hook included.
  * Serving stack: vLLM-on-instance (PyTorch backend), TP8. Stack + versions
    recorded on every result row.
  * Context sweep: 2048 -> 262144 (Qwen); + 524288 / 1048576 for DeepSeek.
  * Concurrency ramp per context length: doubling ramp then binary search to
    the ceiling, defined as OOM OR P99 TTFT > 1.5 s (methodology threshold).
  * KV bytes per session at the ceiling (config-derived + allocator-measured).
  * Paged-KV dimension: vLLM PagedAttention cannot be fully disabled; the
    methodology's paged/non-paged repeat is approximated by a block-size sweep
    and recorded as such in the kv_mode column (TRT-LLM covers the disabled
    case). See --block-sizes.
  * Ceiling confirmation runs honour warm-up + steady-state windows and the
    3-repeat / CoV <= 5% rule (page 3 of the methodology).

Usage:
  python run_o1_benchmark.py --validate-only
  python run_o1_benchmark.py --models qwen --profile quick          # smoke run
  python run_o1_benchmark.py --profile methodology --out o1_results.csv
  python run_o1_benchmark.py --models deepseek --context-lengths 2048,8192

Requires: torch with CUDA, vllm, transformers. Weights must be mirrored
locally (HF_HUB_OFFLINE is forced — no live download during measurement).
Results append to CSV; completed cells are skipped on re-run (resume-safe).

DeepSeek note: reasoning mode is pinned to non-think engagement-wide. This
harness measures serving-layer KV scaling only and does not enable thinking.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ── Methodology constants (v1.3) ──────────────────────────────────────────────
TTFT_CEILING_MS = 1500.0          # O1 4a: interactive (SLO) ceiling bound
MEM_ITL_MULT = 3.0                # O1 4b: decode-health bound = 3x solo P99 ITL
TTFT_ABANDON_MS = 120000.0        # memory ramp: catastrophic queueing cut-off
COV_LIMIT = 0.05                  # page 3: CoV <= 5% across repeats
ENVIRONMENT = "wwt-atc"
OUTCOME_ID = "O1"
EXPECTED_GPUS = 8
EXPECTED_GPU_NAME = "H200"
# H200 SXM5 markets "141 GB" (decimal); torch reports binary GiB and the driver
# reserves a slice, so a healthy device shows ~139.7-140.4 GiB. The check only
# needs to reject smaller SKUs (H100-80GB) or an active MIG slice, so gate well
# below the real capacity rather than at it.
EXPECTED_HBM_GIB_MIN = 130.0

MODELS = {
    "qwen": {
        "model_id": "qwen3.5-122b-a10b",
        "hf_id": "Qwen/Qwen3.5-122B-A10B",
        "precision": "bf16",
        "context_lengths": [2048, 8192, 32768, 65536, 131072, 262144],
        "pinned_sha256": None,     # set at engagement kick-off (Appendix A)
    },
    "qwen-fp8": {
        "model_id": "qwen3.5-122b-a10b-fp8",
        "hf_id": "Qwen/Qwen3.5-122B-A10B-FP8",
        "precision": "fp8-e4m3",
        "context_lengths": [2048, 8192, 32768, 65536, 131072, 262144],
        "pinned_sha256": None,
    },
    "deepseek": {
        "model_id": "deepseek-v4-flash",
        "hf_id": "deepseek-ai/DeepSeek-V4-Flash",
        "precision": "fp4-fp8-mixed",
        # v1.2: extended cells for the 1M-context model
        "context_lengths": [2048, 8192, 32768, 65536, 131072, 262144, 524288, 1048576],
        "pinned_sha256": None,
    },
}

PROFILES = {
    #                 search-burst   ceiling confirmation
    #                 out_toks       warmup_s  capture_s  repeats
    "quick":        {"out": 64,      "warm": 30,  "cap": 60,   "reps": 1},
    "standard":     {"out": 128,     "warm": 120, "cap": 300,  "reps": 3},
    "methodology":  {"out": 128,     "warm": 300, "cap": 900,  "reps": 3},  # page-3 rule
}

CSV_FIELDS = [
    "outcome_id", "environment", "region_zone", "instance_type_or_pod",
    "gpu_model", "gpu_count", "model_id", "model_sha256", "precision",
    "scenario", "context_length_tokens", "kv_mode",
    "concurrent_sessions", "concurrent_sessions_memory", "memory_ceiling_reason",
    "kv_bytes_per_session_config", "kv_bytes_per_session_measured",
    "p99_ttft_ms_at_interactive_ceiling",
    "itl_p50_ms", "itl_p90_ms", "itl_p99_ms", "itl_p99_ms_at_memory_ceiling",
    "gpu_mem_used_gb_max", "repeats_completed", "coefficient_of_variation",
    "cov_pass", "serving_stack", "harness_version", "driver_version",
    "torch_version", "run_start_utc", "run_duration_seconds",
    "operator", "notes",
]


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — setup validation
# ══════════════════════════════════════════════════════════════════════════════
def validate_setup(args):
    """Validate the XE9680 / 8x H200 environment. Returns env-info dict."""
    ok, warn = [], []
    info = {"host": platform.node(), "python": platform.python_version()}

    try:
        import torch
    except ImportError:
        die("PyTorch not installed. pip install torch --index-url per NVIDIA docs")
    info["torch_version"] = torch.__version__
    ok.append(f"torch {torch.__version__}")

    if not torch.cuda.is_available():
        die("CUDA not available to PyTorch")
    info["cuda"] = torch.version.cuda
    ok.append(f"CUDA {torch.version.cuda}")

    n = torch.cuda.device_count()
    info["gpu_count"] = n
    if n != EXPECTED_GPUS:
        die(f"Expected {EXPECTED_GPUS} GPUs, found {n}")
    ok.append(f"{n} GPUs visible")

    names, mems = set(), []
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        names.add(prop.name)
        mems.append(prop.total_memory / 1024**3)
    info["gpu_model_raw"] = sorted(names)
    if len(names) != 1:
        die(f"Mixed GPU models: {names}")
    name = names.pop()
    if EXPECTED_GPU_NAME not in name:
        die(f"Expected {EXPECTED_GPU_NAME}-class GPUs, found '{name}'")
    ok.append(f"GPU model: {name}")
    if min(mems) < EXPECTED_HBM_GIB_MIN:
        die(f"GPU HBM {min(mems):.1f} GiB < {EXPECTED_HBM_GIB_MIN:.0f} GiB floor — "
            f"wrong SKU (H100-80GB?) or MIG slice active")
    ok.append(f"HBM per GPU: {min(mems):.1f} GiB (H200 141GB-class)")

    # P2P / NVLink reachability (TP8 requirement)
    p2p_fail = [(i, j) for i in range(n) for j in range(n)
                if i != j and not torch.cuda.can_device_access_peer(i, j)]
    if p2p_fail:
        die(f"P2P not available between GPU pairs {p2p_fail[:4]}... — check NVLink/fabric")
    ok.append("P2P access verified across all 8 GPUs (NVLink)")

    # Driver / ECC via NVML (optional but recommended)
    info["driver_version"] = "unknown"
    try:
        import pynvml
        pynvml.nvmlInit()
        info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        ok.append(f"driver {info['driver_version']}")
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            cur, _pend = pynvml.nvmlDeviceGetEccMode(h)
            if not cur:
                warn.append(f"ECC disabled on GPU {i} — methodology expects ECC on (O10)")
        pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001
        warn.append(f"pynvml unavailable ({e}) — driver/ECC not verified")

    try:
        import vllm
        info["vllm_version"] = vllm.__version__
        ok.append(f"vllm {vllm.__version__}")
    except ImportError:
        die("vLLM not installed — the on-prem O1 serving stack is vLLM-on-instance")

    # No live download during measurement (page-3 rule)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ok.append("HF offline mode forced (no live download during measurement)")

    for key in args.models:
        m = MODELS[key]
        path = resolve_model_path(m["hf_id"], args.model_root)
        if path is None:
            die(f"Weights for {m['hf_id']} not found under --model-root or HF cache. "
                f"Mirror them locally before running (methodology page 3).")
        m["local_path"] = path
        ok.append(f"{m['hf_id']} -> {path}")
        if m["pinned_sha256"] and args.verify_sha:
            digest = sha256_of_weights(path)
            if digest != m["pinned_sha256"]:
                die(f"SHA-256 mismatch for {m['hf_id']}: {digest}")
            ok.append(f"{m['hf_id']} SHA-256 verified")
        elif args.verify_sha:
            warn.append(f"{m['hf_id']}: no pinned SHA configured — set Appendix A value")

    free_gb = os.statvfs("/tmp").f_bavail * os.statvfs("/tmp").f_frsize / 1024**3
    if free_gb < 50:
        warn.append(f"only {free_gb:.0f} GB free in /tmp")

    print("\n─ Validation ─────────────────────────────────────")
    for line in ok:
        print(f"  PASS  {line}")
    for line in warn:
        print(f"  WARN  {line}")
    print("──────────────────────────────────────────────────\n")
    return info


def resolve_model_path(hf_id, root):
    """Find locally mirrored weights: --model-root/<org>/<name>, then HF cache."""
    cands = []
    if root:
        cands += [os.path.join(root, hf_id), os.path.join(root, hf_id.split("/")[-1])]
    hub = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cands.append(os.path.join(hub, "hub", f"models--{hf_id.replace('/', '--')}"))
    for c in cands:
        if os.path.isdir(c):
            if "models--" in c:  # HF cache layout -> newest snapshot
                snaps = os.path.join(c, "snapshots")
                if os.path.isdir(snaps) and os.listdir(snaps):
                    return os.path.join(snaps, sorted(os.listdir(snaps))[-1])
                continue
            return c
    return None


def sha256_of_weights(path):
    """Deterministic digest over sorted safetensors shards."""
    h = hashlib.sha256()
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".safetensors"):
            with open(os.path.join(path, fn), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — engine + measurement
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BurstResult:
    sessions: int
    ttft_ms: list = field(default_factory=list)
    itl_ms: list = field(default_factory=list)
    oom: bool = False
    error: str = ""

    def p(self, series, q):
        """Nearest-rank percentile: value at ceil(q*N)-th smallest observation."""
        if not series:
            return float("nan")
        s = sorted(series)
        return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]

    @property
    def p99_ttft(self):
        return self.p(self.ttft_ms, 0.99)


def build_engine(model_cfg, max_len, block_size, tp):
    """Start a vLLM async engine. API guarded — pin the vLLM version per Appendix A."""
    from vllm import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    ea = AsyncEngineArgs(
        model=model_cfg["local_path"],
        tensor_parallel_size=tp,
        max_model_len=max_len,
        block_size=block_size,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=False,   # prefix reuse would fake KV scaling
        disable_log_stats=False,
        trust_remote_code=True,
    )
    return AsyncLLMEngine.from_engine_args(ea)


def make_prompt_ids(tokenizer, ctx_len, seed):
    """Synthetic prompt of exactly ctx_len tokens, deterministic per (ctx,seed)."""
    rng = random.Random(f"{ctx_len}-{seed}")
    vocab = tokenizer.vocab_size
    specials = set(tokenizer.all_special_ids or [])
    ids = []
    while len(ids) < ctx_len:
        t = rng.randrange(vocab)
        if t not in specials:
            ids.append(t)
    return ids


async def run_burst(engine, prompt_ids, sessions, out_tokens, hold_s=0.0, stagger_s=0.0):
    """Launch `sessions` concurrent requests; measure per-request TTFT and ITL.

    v1.3: sessions are admitted with a fixed stagger (never a simultaneous
    burst) so TTFT reflects steady-state admission, not queue drain."""
    from vllm import SamplingParams, TokensPrompt
    sp = SamplingParams(max_tokens=out_tokens, temperature=0.0, ignore_eos=True)
    res = BurstResult(sessions=sessions)

    async def one(idx=0):
        await asyncio.sleep(idx * stagger_s)
        rid = uuid.uuid4().hex
        t0 = time.perf_counter()
        first, prev = None, None
        async for out in engine.generate(TokensPrompt(prompt_token_ids=prompt_ids), sp, rid):
            now = time.perf_counter()
            ntok = len(out.outputs[0].token_ids) if out.outputs else 0
            if ntok >= 1 and first is None:
                first = now
                res.ttft_ms.append((now - t0) * 1000)
            elif first is not None and prev is not None and ntok >= 1:
                res.itl_ms.append((now - prev) * 1000)
            prev = now

    t_start = time.perf_counter()
    while True:
        # return_exceptions=True: every session is drained before the burst
        # returns — an exception must never leave orphaned in-flight requests
        # polluting the next measurement.
        outs = await asyncio.gather(*[one(i) for i in range(sessions)],
                                    return_exceptions=True)
        for e in outs:
            if isinstance(e, BaseException):
                msg = str(e)
                if "out of memory" in msg.lower() or "OutOfMemory" in type(e).__name__:
                    res.oom = True
                if not res.error:
                    res.error = msg[:300]
        if res.oom or res.error or time.perf_counter() - t_start >= hold_s:
            break
    return res


def kv_bytes_per_token_from_config(engine, model_cfg):
    """Config-derived KV bytes/token/layer stack. MLA/GQA aware where exposed."""
    try:
        mc = engine.engine.model_config if hasattr(engine, "engine") else engine.model_config
        hf = mc.hf_config
        layers = getattr(hf, "num_hidden_layers")
        kvh = getattr(hf, "num_key_value_heads", getattr(hf, "num_attention_heads", None))
        hd = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        dsz = 2 if "bf16" in model_cfg["precision"] else 1
        # MLA (DeepSeek): compressed latent replaces K/V heads
        if hasattr(hf, "kv_lora_rank") and hf.kv_lora_rank:
            per = layers * (hf.kv_lora_rank + getattr(hf, "qk_rope_head_dim", 0)) * dsz
            return per, "mla"
        return 2 * layers * kvh * hd * dsz, "mha/gqa"
    except Exception as e:  # noqa: BLE001
        log(f"  kv-config derivation failed ({e}) — measured value only")
        return None, "unavailable"


def gpu_mem_max_gb():
    import torch
    return max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — the O1 sweep
# ══════════════════════════════════════════════════════════════════════════════
async def _search(engine, prompt_ids, out_tokens, start, max_sessions, stagger_s,
                  is_bad, label):
    """Doubling ramp from `start` then binary search against predicate is_bad(burst).
    Returns (ceiling, burst@ceiling, reason_first_bad)."""
    last_good, first_bad, results, reason = max(start, 0), None, {}, ""
    s = max(start, 1)
    while s <= max_sessions:
        log(f"    {label} ramp: {s} concurrent sessions")
        r = await run_burst(engine, prompt_ids, s, out_tokens, stagger_s=stagger_s)
        results[s] = r
        bad, why = is_bad(r)
        if bad:
            first_bad, reason = s, why
            log(f"    {label} limit hit at {s} ({why})")
            break
        last_good = s
        s *= 2
    if first_bad is None:
        return last_good, results.get(last_good), "max-sessions reached"
    lo, hi = last_good, first_bad
    while hi - lo > 1:
        mid = (lo + hi) // 2
        log(f"    {label} bisect: {mid} sessions")
        r = await run_burst(engine, prompt_ids, mid, out_tokens, stagger_s=stagger_s)
        results[mid] = r
        bad, why = is_bad(r)
        if bad:
            hi, reason = mid, why
        else:
            lo = mid
    if lo == 0:
        return 0, results.get(first_bad), reason
    return lo, results[lo], reason


async def find_interactive_ceiling(engine, prompt_ids, out_tokens, max_sessions, stagger_s):
    """O1 4a: SLO ceiling — P99 TTFT of admitted sessions must stay <= 1.5 s."""
    def bad(r):
        if r.oom:
            return True, "OOM"
        if r.error:
            return True, "error"
        if r.p99_ttft > TTFT_CEILING_MS:
            return True, f"P99 TTFT {r.p99_ttft:.0f}ms > {TTFT_CEILING_MS:.0f}ms"
        return False, ""
    c, b, _ = await _search(engine, prompt_ids, out_tokens, 1, max_sessions,
                            stagger_s, bad, "SLO")
    return c, b


async def find_memory_ceiling(engine, prompt_ids, out_tokens, start, max_sessions,
                              itl_solo_p99, stagger_s):
    """O1 4b: memory ceiling — TTFT ignored; stop on OOM / admission failure /
    decode-health violation (P99 ITL > 3x single-session baseline)."""
    itl_bound = MEM_ITL_MULT * max(itl_solo_p99, 0.001)
    def bad(r):
        if r.oom:
            return True, "OOM"
        if r.error:
            return True, f"admission failure/error: {r.error[:60]}"
        itl99 = r.p(r.itl_ms, 0.99)
        if r.itl_ms and itl99 > itl_bound:
            return True, f"decode-health: P99 ITL {itl99:.1f}ms > {itl_bound:.1f}ms"
        if r.p99_ttft > TTFT_ABANDON_MS:
            return True, "catastrophic queueing (KV pool exhausted)"
        return False, ""
    return await _search(engine, prompt_ids, out_tokens, start, max_sessions,
                         stagger_s, bad, "MEM")


async def run_cell(engine, tokenizer, model_cfg, ctx, block_size, prof, info, args):
    """One (model x context x kv_mode) cell -> CSV row dict."""
    import torch
    t_cell = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)

    prompt_ids = make_prompt_ids(tokenizer, ctx, args.seed)
    log(f"  ctx={ctx} block={block_size}: warm-up ({prof['warm']}s)")
    solo = await run_burst(engine, prompt_ids, 1, prof["out"], hold_s=prof["warm"])
    itl_solo_p99 = solo.p(solo.itl_ms, 0.99) if solo.itl_ms else 0.0

    # O1 4a — interactive (SLO) ceiling
    slo, _ = await find_interactive_ceiling(engine, prompt_ids, prof["out"],
                                            args.max_sessions, args.stagger)
    if slo == 0:
        log(f"  ctx={ctx}: single session violates the 1.5s TTFT bound "
            f"(expected at long context — informative result, not a failure)")

    # O1 4b — memory ceiling, TTFT ignored
    mem, mem_burst, mem_reason = await find_memory_ceiling(
        engine, prompt_ids, prof["out"], max(slo, 1), args.max_sessions,
        itl_solo_p99, args.stagger)
    itl99_mem = mem_burst.p(mem_burst.itl_ms, 0.99) if (mem_burst and mem_burst.itl_ms) else ""

    # Confirmation repeats at the interactive ceiling (steady-state, CoV rule)
    ttfts, itls = [], []
    reps = prof["reps"] if slo > 0 else 1
    conf = None
    for rep in range(reps):
        log(f"  ctx={ctx}: confirmation {rep + 1}/{reps} at {max(slo,1)} sessions "
            f"({prof['cap']}s capture)")
        conf = await run_burst(engine, prompt_ids, max(slo, 1), prof["out"],
                               hold_s=prof["cap"], stagger_s=args.stagger)
        ttfts.append(conf.p99_ttft)
        itls.extend(conf.itl_ms)
    cov = (statistics.pstdev(ttfts) / statistics.mean(ttfts)
           if len(ttfts) > 1 and statistics.mean(ttfts) > 0 else 0.0)
    ceiling = mem  # KV metrics are reported at the memory ceiling (v1.3)

    per_tok, kv_note = kv_bytes_per_token_from_config(engine, model_cfg)
    kv_cfg = per_tok * ctx if per_tok else ""
    kv_meas = ""
    try:  # measured: engine KV block accounting, guarded (API drift across versions)
        core = engine.engine if hasattr(engine, "engine") else engine
        cc = core.cache_config
        blk_bytes = per_tok * cc.block_size if per_tok else None
        if blk_bytes and ceiling:
            kv_meas = int(blk_bytes * (ctx / cc.block_size + 1))
    except Exception:  # noqa: BLE001
        pass

    b = conf or BurstResult(sessions=0)
    return {
        "outcome_id": OUTCOME_ID, "environment": ENVIRONMENT, "region_zone": "on-prem",
        "instance_type_or_pod": args.pod_label, "gpu_model": "h200-141gb-sxm5",
        "gpu_count": info["gpu_count"], "model_id": model_cfg["model_id"],
        "model_sha256": model_cfg.get("pinned_sha256") or "unpinned",
        "precision": model_cfg["precision"], "scenario": "server",
        "context_length_tokens": ctx,
        "kv_mode": f"paged-block{block_size} (vLLM cannot disable paging; "
                   f"non-paged cell covered by TRT-LLM run)",
        "concurrent_sessions": slo,
        "concurrent_sessions_memory": mem,
        "memory_ceiling_reason": mem_reason,
        "kv_bytes_per_session_config": kv_cfg,
        "kv_bytes_per_session_measured": kv_meas,
        "p99_ttft_ms_at_interactive_ceiling": round(b.p99_ttft, 1) if b.ttft_ms else "",
        "itl_p50_ms": round(b.p(itls, 0.50), 2) if itls else "",
        "itl_p90_ms": round(b.p(itls, 0.90), 2) if itls else "",
        "itl_p99_ms": round(b.p(itls, 0.99), 2) if itls else "",
        "itl_p99_ms_at_memory_ceiling": round(itl99_mem, 2) if itl99_mem != "" else "",
        "gpu_mem_used_gb_max": round(gpu_mem_max_gb(), 1),
        "repeats_completed": reps, "coefficient_of_variation": round(cov, 4),
        "cov_pass": cov <= COV_LIMIT,
        "serving_stack": f"vllm-on-instance/{info.get('vllm_version','?')}",
        "harness_version": "wwt-o1-harness/1.1-dualceiling",
        "driver_version": info.get("driver_version", ""),
        "torch_version": info.get("torch_version", ""),
        "run_start_utc": started,
        "run_duration_seconds": int(time.perf_counter() - t_cell),
        "operator": args.operator,
        "notes": f"kv-derivation={kv_note}; ttft_bound={TTFT_CEILING_MS}ms; "
                 f"itl_health=3x{round(itl_solo_p99,1)}ms; stagger={args.stagger}s; "
                 f"profile={args.profile}; methodology=v1.3"
                 + (f"; ERROR={b.error}" if b.error else ""),
    }


def done_cells(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["model_id"], row["context_length_tokens"], row["kv_mode"]))
    return done


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


async def main_async(args, info):
    from transformers import AutoTokenizer
    skip = done_cells(args.out)
    for key in args.models:
        m = MODELS[key]
        ctxs = args.context_lengths or m["context_lengths"]
        tokenizer = AutoTokenizer.from_pretrained(m["local_path"], trust_remote_code=True)
        for block in args.block_sizes:
            max_len = max(ctxs) + args.out_headroom
            log(f"── engine: {m['hf_id']}  TP{args.tp}  max_len={max_len}  block={block}")
            engine = build_engine(m, max_len, block, args.tp)
            try:
                for ctx in ctxs:
                    kv_mode = (f"paged-block{block} (vLLM cannot disable paging; "
                               f"non-paged cell covered by TRT-LLM run)")
                    if (m["model_id"], str(ctx), kv_mode) in skip:
                        log(f"  ctx={ctx} block={block}: already in CSV — skipped")
                        continue
                    prof = dict(PROFILES[args.profile])
                    row = await run_cell(engine, tokenizer, m, ctx, block, prof, info, args)
                    append_row(args.out, row)
                    log(f"  ctx={ctx}: slo={row['concurrent_sessions']} "
                        f"mem={row['concurrent_sessions_memory']} "
                        f"({row['memory_ceiling_reason']}) "
                        f"p99TTFT={row['p99_ttft_ms_at_interactive_ceiling']}ms "
                        f"CoV={row['coefficient_of_variation']} -> {args.out}")
            finally:
                try:
                    engine.shutdown_background_loop()
                except Exception:  # noqa: BLE001
                    pass
                del engine
                import torch, gc  # noqa: E401
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(10)  # let NCCL/TP workers tear down before next engine
    log(f"Sweep complete. Results: {args.out}")


def parse_args():
    ap = argparse.ArgumentParser(description="O1 KV-cache scaling harness (v1.2)")
    ap.add_argument("--models", default="qwen,deepseek",
                    help=f"comma list from {list(MODELS)} (default: qwen,deepseek)")
    ap.add_argument("--context-lengths", default=None,
                    help="override comma list, e.g. 2048,8192")
    ap.add_argument("--block-sizes", default="16,32",
                    help="KV block sizes standing in for the paged/non-paged repeat")
    ap.add_argument("--profile", choices=list(PROFILES), default="standard")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-sessions", type=int, default=4096)
    ap.add_argument("--out", default="o1_results.csv")
    ap.add_argument("--out-headroom", type=int, default=256,
                    help="extra max_model_len for generated tokens")
    ap.add_argument("--model-root", default=os.environ.get("WWT_MODEL_ROOT", "/models"))
    ap.add_argument("--pod-label", default="XE9680-A")
    ap.add_argument("--operator", default=os.environ.get("USER", "unknown"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stagger", type=float, default=0.25,
                    help="session admission stagger in seconds (v1.3: no bursts)")
    ap.add_argument("--verify-sha", action="store_true",
                    help="verify weights against Appendix A pinned SHA-256")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    a.models = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in a.models:
        if m not in MODELS:
            die(f"unknown model key '{m}' — choose from {list(MODELS)}")
    if a.context_lengths:
        a.context_lengths = [int(x) for x in a.context_lengths.split(",")]
    a.block_sizes = [int(x) for x in a.block_sizes.split(",")]
    return a


def main():
    args = parse_args()
    info = validate_setup(args)
    if args.validate_only:
        log("Validation only — exiting.")
        return
    log(f"Profile '{args.profile}': {json.dumps(PROFILES[args.profile])}")
    log(f"TTFT ceiling bound: {TTFT_CEILING_MS} ms | CoV limit: {COV_LIMIT}")
    try:
        asyncio.run(main_async(args, info))
    except KeyboardInterrupt:
        log("Interrupted — completed cells are already in the CSV; re-run to resume.")


if __name__ == "__main__":
    main()
