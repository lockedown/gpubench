#!/usr/bin/env python3
"""O9 in-container probe — times engine bring-up phases and the first-100
inference latency curve. Launched by run_o9_benchmark.py; do not run directly
unless debugging. Writes a JSON result to --out.

Phases measured (methodology O9 step 4):
  engine_build_s   weights transfer + allocator warm-up + engine init
  first_infer_ms   first end-to-end inference (captures kernel autotune/JIT)
  infer_ms[100]    latency curve of the first 100 sequential inferences
"""
import argparse, asyncio, json, random, time  # noqa: E401

# Per-model engine requirements — keep in sync with run_o1_benchmark_v2.py
ENGINE_REQS = {
    "qwen":     {},
    "qwen-fp8": {},
    "deepseek": {"engine_kwargs": {"kv_cache_dtype": "fp8"},
                 "engine_env": {"VLLM_ROCM_USE_AITER": "1"},
                 "default_block": True},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-key", required=True, choices=list(ENGINE_REQS))
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--out-tokens", type=int, default=32)
    ap.add_argument("--inferences", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reqs = ENGINE_REQS[args.model_key]

    import os
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    for k, v in (reqs.get("engine_env") or {}).items():
        os.environ.setdefault(k, v)

    t0 = time.monotonic()
    from vllm import AsyncEngineArgs, SamplingParams, TokensPrompt
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from transformers import AutoTokenizer
    t_import = time.monotonic()

    ea_kwargs = dict(model=args.model_path, tensor_parallel_size=args.tp,
                     max_model_len=4096, gpu_memory_utilization=0.92,
                     enable_prefix_caching=False, trust_remote_code=True,
                     **(reqs.get("engine_kwargs") or {}))
    if not reqs.get("default_block"):
        ea_kwargs["block_size"] = 16
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**ea_kwargs))
    t_ready = time.monotonic()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rng = random.Random(999)
    specials = set(tok.all_special_ids or [])
    ids = []
    while len(ids) < args.prompt_tokens:
        t = rng.randrange(tok.vocab_size)
        if t not in specials:
            ids.append(t)
    sp = SamplingParams(max_tokens=args.out_tokens, temperature=0.0, ignore_eos=True)

    async def run():
        lat, ttft_first = [], None
        nonlocal_first = {}
        for i in range(args.inferences):
            t1 = time.monotonic()
            first = None
            async for out in engine.generate(
                    TokensPrompt(prompt_token_ids=ids), sp, f"o9-{i}"):
                if first is None and out.outputs and out.outputs[0].token_ids:
                    first = time.monotonic()
            lat.append((time.monotonic() - t1) * 1000)
            if i == 0 and first:
                nonlocal_first["ttft"] = (first - t1) * 1000
        return lat, nonlocal_first.get("ttft")

    lat, ttft_first = asyncio.get_event_loop().run_until_complete(run()) \
        if False else asyncio.run(run())

    import vllm, torch  # noqa: E401
    result = {
        "import_s": round(t_import - t0, 2),
        "engine_build_s": round(t_ready - t_import, 2),
        "total_ready_s": round(t_ready - t0, 2),
        "first_infer_ms": round(lat[0], 1),
        "ttft_first_ms": round(ttft_first, 1) if ttft_first else None,
        "infer_ms": [round(x, 2) for x in lat],
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
    }
    with open(args.out, "w") as f:
        json.dump(result, f)
    print("O9_PROBE_OK", json.dumps({k: v for k, v in result.items()
                                     if k != "infer_ms"}))

if __name__ == "__main__":
    main()
