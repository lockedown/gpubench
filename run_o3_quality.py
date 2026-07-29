#!/usr/bin/env python3
"""
O3-lite — Quantization QUALITY gate (partial O3)
=================================================
GPU Benchmark Methodology v1.3, O3 steps 5-6 subset | HOST orchestrator.
Serves each model (full node, TP8) and runs lm-evaluation-harness against the
endpoint: GSM8K 8-shot (generative) + MMLU 5-shot (loglikelihood). Pairs with
the O1 capacity data to complete the per-platform precision verdict
(e.g. "FP8 costs 17% capacity on MI300X — does it also cost accuracy?").

One-time derived image (keeps the pinning discipline — never pip into a
running benchmark container):

    cat > Dockerfile.o3eval <<'EOF'
    FROM <pinned-vllm-image>
    RUN pip install "lm-eval[api]==0.4.9"
    EOF
    sudo docker build -f Dockerfile.o3eval -t wwt/vllm-bench:o3eval .

Run:  python3 run_o3_quality.py --serve-image <pinned> \\
          --eval-image wwt/vllm-bench:o3eval --models qwen,qwen-fp8 \\
          --operator <you> --out o3_quality.csv

NOTE: eval DATASETS download at runtime (HF_HUB_OFFLINE deliberately unset in
the eval container only — datasets, not weights; recorded in notes). Partial,
non-MLPerf cells: MTEB/DocVQA and the Appendix A tolerance gate need the full
O3 campaign.
"""
import argparse, csv, glob, json, os, subprocess, sys, time  # noqa: E401
from datetime import datetime, timezone

HARNESS_VERSION = "wwt-o3lite-harness/1.0"
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
TASKS = [  # (lm-eval task, n-shot, per-subtask limit, primary metric key)
    ("gsm8k", 8, 250, "exact_match,strict-match"),
    ("mmlu", 5, 12, "acc,none"),
]
CSV_FIELDS = ["outcome_id", "environment", "region_zone", "instance_type_or_pod",
              "model_id", "precision", "task", "n_shot", "limit_per_subtask",
              "metric", "value", "stderr", "serving_stack", "eval_harness",
              "harness_version", "run_start_utc", "run_duration_s",
              "operator", "notes"]

def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)

def snapshot_dir(cache_dir, key):
    base = os.path.join(cache_dir, "hub", MODELS[key]["hub"], "snapshots")
    return f"/root/.cache/huggingface/hub/{MODELS[key]['hub']}/snapshots/" + \
        sorted(os.listdir(base))[-1]

def start_serve(args, key, port):
    m = MODELS[key]
    dev = (["--gpus", "all"] if args.nvidia else
           ["--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video", "--group-add", "render",
            "--security-opt", "seccomp=unconfined", "--cap-add=SYS_PTRACE"])
    env = sum((["-e", f"{k}={v}"] for k, v in m["env"].items()), [])
    subprocess.run(["docker", "rm", "-f", "o3serve"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "run", "-d", "--name", "o3serve", "--network", "host",
                    "--ipc=host", "--shm-size=32g", *dev, *env,
                    "-v", f"{args.cache_dir}:/root/.cache/huggingface",
                    "-e", "HF_HUB_OFFLINE=1", args.serve_image,
                    snapshot_dir(args.cache_dir, key),
                    "--served-model-name", key,
                    "--tensor-parallel-size", str(args.tp),
                    "--gpu-memory-utilization", "0.92",
                    "--max-model-len", "4096", "--port", str(port),
                    *m["serve_args"]], check=True)

def wait_ready(port, timeout_s):
    import http.client
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)
    return False

def run_eval(args, key, task, shots, limit, port, outdir):
    cmd = ["docker", "run", "--rm", "--network", "host",
           "-v", f"{args.workdir}:/work", "-w", "/work", "--entrypoint", "lm_eval",
           args.eval_image,
           "--model", "local-completions",
           "--model_args", (f"model={key},base_url=http://127.0.0.1:{port}/v1/"
                            f"completions,num_concurrent=8,max_retries=3,"
                            f"tokenized_requests=False"),
           "--tasks", task, "--num_fewshot", str(shots),
           "--limit", str(limit), "--output_path", f"/work/{outdir}",
           "--seed", "1234"]
    return subprocess.run(cmd).returncode

def parse_results(workdir, outdir, task, metric_key):
    """lm-eval writes results_*.json under output_path; pull the group metric."""
    hits = sorted(glob.glob(os.path.join(workdir, outdir, "**", "results_*.json"),
                            recursive=True))
    if not hits:
        return None, None, None
    res = json.load(open(hits[-1]))
    node = res.get("results", {}).get(task) or {}
    val = node.get(metric_key)
    err = node.get(metric_key.replace(",", "_stderr,", 1)) or \
        node.get(f"{metric_key.split(',')[0]}_stderr,{metric_key.split(',')[1]}")
    return val, err, res.get("config", {}).get("model", "")

def main():
    ap = argparse.ArgumentParser(description="O3-lite quality gate (host)")
    ap.add_argument("--serve-image", required=True)
    ap.add_argument("--eval-image", required=True)
    ap.add_argument("--models", default="qwen,qwen-fp8")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--ready-timeout", type=int, default=2400)
    ap.add_argument("--cache-dir", default="/opt/huggingface-cache")
    ap.add_argument("--workdir", default=os.path.expanduser("~/dl_script/gpubench"))
    ap.add_argument("--out", default="o3_quality.csv")
    ap.add_argument("--pod-label", default="MI300X-A")
    ap.add_argument("--environment", default="wwt-atc")
    ap.add_argument("--region-zone", default="on-prem")
    ap.add_argument("--operator", default=os.environ.get("USER", "unknown"))
    ap.add_argument("--nvidia", action="store_true")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    out_path = os.path.join(args.workdir, args.out)
    done = set()
    if os.path.exists(out_path):
        done = {(r["model_id"], r["task"]) for r in csv.DictReader(open(out_path))}

    for key in models:
        pending = [(t, s, l, mk) for t, s, l, mk in TASKS
                   if (MODELS[key]["model_id"], t) not in done]
        if not pending:
            log(f"{key}: all tasks already in CSV — skipped")
            continue
        log(f"── serving {key} (TP{args.tp})")
        start_serve(args, key, args.port)
        try:
            if not wait_ready(args.port, args.ready_timeout):
                sys.exit(f"{key} never became ready")
            for task, shots, limit, metric_key in pending:
                outdir = f"o3eval_{key}_{task}"
                log(f"  lm-eval {task} ({shots}-shot, limit {limit})")
                t0 = time.monotonic()
                started = datetime.now(timezone.utc).isoformat()
                rc = run_eval(args, key, task, shots, limit, args.port, outdir)
                if rc != 0:
                    log(f"  {task} FAILED rc={rc} — nothing recorded")
                    continue
                val, err, _ = parse_results(args.workdir, outdir, task, metric_key)
                if val is None:
                    log(f"  {task}: could not parse results — nothing recorded")
                    continue
                row = {
                    "outcome_id": "O3-partial", "environment": args.environment,
                    "region_zone": args.region_zone,
                    "instance_type_or_pod": args.pod_label,
                    "model_id": MODELS[key]["model_id"],
                    "precision": MODELS[key]["precision"],
                    "task": task, "n_shot": shots, "limit_per_subtask": limit,
                    "metric": metric_key, "value": round(float(val), 4),
                    "stderr": round(float(err), 4) if err else "",
                    "serving_stack": f"vllm-serve/{args.serve_image}",
                    "eval_harness": args.eval_image,
                    "harness_version": HARNESS_VERSION,
                    "run_start_utc": started,
                    "run_duration_s": int(time.monotonic() - t0),
                    "operator": args.operator,
                    "notes": ("non-MLPerf partial O3 quality probe; eval datasets "
                              "downloaded at runtime (datasets only — weights "
                              "remained offline); Appendix A tolerance gate "
                              "requires the full suite"),
                }
                new = not os.path.exists(out_path)
                with open(out_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                log(f"  {task}: {metric_key}={row['value']} -> {args.out}")
        finally:
            subprocess.run(["docker", "rm", "-f", "o3serve"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("O3-lite complete")

if __name__ == "__main__":
    main()
