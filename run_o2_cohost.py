#!/usr/bin/env python3
"""
O2-lite — Multi-model co-hosting via shared-GPU memory partition
=================================================================
GPU Benchmark Methodology v1.3, O2 subset | HOST-side orchestrator+load client
(stdlib only — threads + http.client; no aiohttp/torch needed on the host).

Topology: two models co-resident on the SAME GPU subset (default 0-3, TP4,
gpu-memory-utilization split), served as OpenAI endpoints. Phases:

  1. siloed A   — model A alone on the subset -> closed-loop interactive ceiling
  2. siloed B   — model B alone               -> ceiling
  3. cohost 50/50 — both up, each driven at 50% of its siloed ceiling
  4. noisy A    — A ramped to 100% of ITS ceiling, B held at 50% (victim)
  5. noisy B    — rotated
Isolation score (v1.3 page-3): victim P99 under pressure / victim P99 at 50/50.

  sudo python3 run_o2_cohost.py --image vllm/vllm-openai-rocm:nightly \
      --pair qwen-fp8,deepseek --operator <you> --out o2_results.csv

Scope caveat (recorded per row): this is co-tenancy via memory partition on
shared GPUs — the closest legal topology on MI300X without CPX partitioning
(O4). Ensemble-on-one-GPU is impossible with 122B/284B-class models.
"""
import argparse, csv, http.client, json, os, statistics, subprocess, sys  # noqa: E401
import threading, time  # noqa: E401
from datetime import datetime, timezone

HARNESS_VERSION = "wwt-o2-harness/1.0"
TTFT_BOUND_MS = 1500.0
RAMP_BUDGET_S = 20.0
WINDOW_S = 45.0                      # steady-state capture per probe/phase

MODELS = {
    "qwen":     {"model_id": "qwen3.5-122b-a10b", "precision": "bf16",
                 "hub": "models--Qwen--Qwen3.5-122B-A10B", "serve_args": [], "env": {}},
    "qwen-fp8": {"model_id": "qwen3.5-122b-a10b-fp8", "precision": "fp8-e4m3",
                 "hub": "models--Qwen--Qwen3.5-122B-A10B-FP8", "serve_args": [], "env": {}},
    "deepseek": {"model_id": "deepseek-v4-flash", "precision": "fp4-fp8-mixed",
                 "hub": "models--deepseek-ai--DeepSeek-V4-Flash",
                 "serve_args": ["--kv-cache-dtype", "fp8"],
                 "env": {"VLLM_ROCM_USE_AITER": "1"}},
}
# ~1,000-token synthetic banking prompt (relative isolation, not absolute perf)
PROMPT = ("The quarterly risk assessment covering liquidity market and operational "
          "exposure across the retail and institutional portfolios requires detailed "
          "reconciliation of the following positions and their hedges. ") * 55

CSV_FIELDS = ["outcome_id", "environment", "region_zone", "instance_type_or_pod",
              "topology", "model_id", "precision", "role", "workers",
              "siloed_ceiling", "p99_ttft_ms", "itl_p50_ms", "itl_p99_ms",
              "baseline_p99_ttft_ms", "isolation_score_ttft",
              "serving_stack", "harness_version", "run_start_utc", "operator", "notes"]

def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)

# ── HTTP closed-loop load client (threads; one in-flight request per worker) ──
class Burst:
    def __init__(self):
        self.ttft, self.itl, self.errors = [], [], 0
        self.lock = threading.Lock()

def one_request(port, model_name, out_tokens, burst, record):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        body = json.dumps({"model": model_name, "prompt": PROMPT,
                           "max_tokens": out_tokens, "temperature": 0,
                           "stream": True, "ignore_eos": True})
        t0 = time.monotonic()
        c.request("POST", "/v1/completions", body,
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        first, prev = None, None
        while True:
            line = r.fp.readline()
            if not line:
                break
            if line.startswith(b"data:") and b"[DONE]" not in line:
                now = time.monotonic()
                if first is None:
                    first = now
                    if record:
                        with burst.lock:
                            burst.ttft.append((now - t0) * 1000)
                elif prev is not None and record:
                    with burst.lock:
                        burst.itl.append((now - prev) * 1000)
                prev = now
        c.close()
        return first is not None
    except Exception:  # noqa: BLE001
        with burst.lock:
            burst.errors += 1
        return False

def run_phase(port, model_name, workers, out_tokens, window_s):
    """Closed-loop: `workers` threads each keep one request in flight."""
    burst = Burst()
    stagger = min(0.25, RAMP_BUDGET_S / max(workers, 1))
    ramp_done = time.monotonic() + stagger * workers
    end = ramp_done + window_s
    stop = threading.Event()

    def worker(idx):
        time.sleep(idx * stagger)
        while not stop.is_set() and time.monotonic() < end:
            one_request(port, model_name, out_tokens, burst,
                        record=time.monotonic() >= ramp_done)
    ts = [threading.Thread(target=worker, args=(i,), daemon=True)
          for i in range(workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return burst

def pctl(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    import math
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]

def find_ceiling(port, model_name, out_tokens, max_workers):
    last_good, first_bad, results = 0, None, {}
    n = 1
    while n <= max_workers:
        log(f"    ceiling ramp: {n} workers")
        b = run_phase(port, model_name, n, out_tokens, WINDOW_S)
        results[n] = b
        if b.errors or pctl(b.ttft, 0.99) > TTFT_BOUND_MS or not b.ttft:
            first_bad = n
            break
        last_good = n
        n *= 2
    if first_bad is None:
        return last_good, results[last_good]
    lo, hi = last_good, first_bad
    while hi - lo > 1:
        mid = (lo + hi) // 2
        log(f"    ceiling bisect: {mid} workers")
        b = run_phase(port, model_name, mid, out_tokens, WINDOW_S)
        results[mid] = b
        if b.errors or pctl(b.ttft, 0.99) > TTFT_BOUND_MS or not b.ttft:
            hi = mid
        else:
            lo = mid
    return lo, results.get(lo) or results.get(first_bad)

# ── container management ───────────────────────────────────────────────────────
def snapshot_dir(cache_dir, key):
    base = os.path.join(cache_dir, "hub", MODELS[key]["hub"], "snapshots")
    return f"/root/.cache/huggingface/hub/{MODELS[key]['hub']}/snapshots/" + \
        sorted(os.listdir(base))[-1]

def start_serve(args, key, name, port):
    m = MODELS[key]
    dev = (["--gpus", "all", "-e", f"CUDA_VISIBLE_DEVICES={args.gpus}"]
           if args.nvidia else
           ["--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video", "--group-add", "render",
            "--security-opt", "seccomp=unconfined", "--cap-add=SYS_PTRACE",
            "-e", f"HIP_VISIBLE_DEVICES={args.gpus}",
            "-e", f"ROCR_VISIBLE_DEVICES={args.gpus}"])
    env = sum((["-e", f"{k}={v}"] for k, v in m["env"].items()), [])
    cmd = ["docker", "run", "-d", "--name", name, "--network", "host",
           "--ipc=host", "--shm-size=32g", *dev, *env,
           "-v", f"{args.cache_dir}:/root/.cache/huggingface",
           "-e", "HF_HUB_OFFLINE=1", args.image,
           snapshot_dir(args.cache_dir, key),
           "--served-model-name", key,
           "--tensor-parallel-size", str(args.tp),
           "--gpu-memory-utilization", str(args.mem_frac),
           "--max-model-len", "4096", "--port", str(port)]
    subprocess.run(["docker", "rm", "-f", name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd, check=True)
    log(f"  {name} ({key}) starting on :{port} — waiting for ready")

def wait_ready(port, timeout_s):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                log(f"  :{port} ready ({time.monotonic()-t0:.0f}s)")
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)
    return False

def stop_all(names):
    for n in names:
        subprocess.run(["docker", "rm", "-f", n],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── main flow ──────────────────────────────────────────────────────────────────
def emit(args, rows, topology, key, role, workers, ceiling, burst,
         baseline_p99=None):
    p99 = round(pctl(burst.ttft, 0.99), 1) if burst.ttft else ""
    iso = (round(p99 / baseline_p99, 2)
           if baseline_p99 and p99 != "" and baseline_p99 > 0 else "")
    rows.append({
        "outcome_id": "O2", "environment": args.environment,
        "region_zone": args.region_zone, "instance_type_or_pod": args.pod_label,
        "topology": topology, "model_id": MODELS[key]["model_id"],
        "precision": MODELS[key]["precision"], "role": role, "workers": workers,
        "siloed_ceiling": ceiling, "p99_ttft_ms": p99,
        "itl_p50_ms": round(pctl(burst.itl, 0.5), 2) if burst.itl else "",
        "itl_p99_ms": round(pctl(burst.itl, 0.99), 2) if burst.itl else "",
        "baseline_p99_ttft_ms": baseline_p99 or "",
        "isolation_score_ttft": iso,
        "serving_stack": f"vllm-serve/{args.image}",
        "harness_version": HARNESS_VERSION,
        "run_start_utc": datetime.now(timezone.utc).isoformat(),
        "operator": args.operator,
        "notes": f"shared GPUs {args.gpus} tp{args.tp} mem_frac={args.mem_frac}; "
                 f"ttft_bound={TTFT_BOUND_MS}ms; window={WINDOW_S}s; "
                 f"co-tenancy via memory partition (CPX topology deferred to O4); "
                 f"errors={burst.errors}",
    })

def main():
    ap = argparse.ArgumentParser(description="O2-lite co-hosting (host)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--pair", default="qwen-fp8,deepseek",
                    help="two model keys, comma-separated")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--mem-frac", type=float, default=0.42)
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--max-workers", type=int, default=512)
    ap.add_argument("--ready-timeout", type=int, default=2400)
    ap.add_argument("--cache-dir", default="/opt/huggingface-cache")
    ap.add_argument("--out", default="o2_results.csv")
    ap.add_argument("--pod-label", default="MI300X-A")
    ap.add_argument("--environment", default="wwt-atc")
    ap.add_argument("--region-zone", default="on-prem")
    ap.add_argument("--operator", default=os.environ.get("USER", "unknown"))
    ap.add_argument("--nvidia", action="store_true")
    args = ap.parse_args()
    a_key, b_key = [x.strip() for x in args.pair.split(",")]
    ports = {a_key: 8801, b_key: 8802}
    names = {a_key: "o2a", b_key: "o2b"}
    rows, ceilings = [], {}

    try:
        # Phase 1+2 — siloed ceilings (same mem_frac as co-hosting, for fairness)
        for key in (a_key, b_key):
            log(f"── siloed: {key}")
            start_serve(args, key, names[key], ports[key])
            if not wait_ready(ports[key], args.ready_timeout):
                sys.exit(f"{key} never became ready")
            c, b = find_ceiling(ports[key], key, args.out_tokens, args.max_workers)
            ceilings[key] = max(c, 1)
            emit(args, rows, "siloed", key, "alone", c, c, b)
            log(f"  {key} siloed ceiling = {c}")
            stop_all([names[key]])

        # Phase 3 — both up, 50/50
        log("── co-host: starting both")
        for key in (a_key, b_key):
            start_serve(args, key, names[key], ports[key])
        for key in (a_key, b_key):
            if not wait_ready(ports[key], args.ready_timeout):
                sys.exit(f"{key} never became ready (co-host)")
        half = {k: max(1, round(0.5 * ceilings[k])) for k in (a_key, b_key)}
        base_p99 = {}
        log(f"── co-host 50/50: {a_key}={half[a_key]}, {b_key}={half[b_key]} workers")
        res = {}
        th = {k: threading.Thread(
            target=lambda k=k: res.__setitem__(k, run_phase(
                ports[k], k, half[k], args.out_tokens, WINDOW_S)))
            for k in (a_key, b_key)}
        [t.start() for t in th.values()]
        [t.join() for t in th.values()]
        for k in (a_key, b_key):
            base_p99[k] = round(pctl(res[k].ttft, 0.99), 1) if res[k].ttft else None
            emit(args, rows, "cohost-50/50", k, "co-tenant", half[k],
                 ceilings[k], res[k])

        # Phase 4+5 — noisy neighbour, rotated
        for noisy in (a_key, b_key):
            victim = b_key if noisy == a_key else a_key
            log(f"── noisy={noisy} @100% ({ceilings[noisy]}), victim={victim} @50%")
            res = {}
            th = {
                noisy: threading.Thread(target=lambda: res.__setitem__(
                    noisy, run_phase(ports[noisy], noisy, ceilings[noisy],
                                     args.out_tokens, WINDOW_S))),
                victim: threading.Thread(target=lambda: res.__setitem__(
                    victim, run_phase(ports[victim], victim, half[victim],
                                      args.out_tokens, WINDOW_S))),
            }
            [t.start() for t in th.values()]
            [t.join() for t in th.values()]
            emit(args, rows, f"cohost-noisy-{noisy}", noisy, "aggressor",
                 ceilings[noisy], ceilings[noisy], res[noisy])
            emit(args, rows, f"cohost-noisy-{noisy}", victim, "victim",
                 half[victim], ceilings[victim], res[victim],
                 baseline_p99=base_p99[victim])
    finally:
        stop_all(list(names.values()))

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    log(f"O2 complete — {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
