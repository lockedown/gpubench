#!/usr/bin/env python3
"""
O9 — Model cold-start time — HOST-side orchestrator
====================================================
GPU Benchmark Methodology v1.3 | runs on the HOST (stdlib only, any
Python 3.8+; no GPU libraries needed here). Launches the in-container probe
(o9_probe.py) per (model x cache-state x repeat), dropping the OS page cache
for cold runs and metering actual NVMe bytes read via /proc/diskstats.

  sudo python3 run_o9_benchmark.py --image vllm/vllm-openai-rocm:nightly \
      --models qwen,qwen-fp8,deepseek --repeats 2 --out o9_results.csv

sudo is required for cache drops (cold cells). Runs are sequential and
foreground; use tmux. Resume-safe: completed (model,cache_state,repeat)
cells are skipped. AMD device flags default; --nvidia switches them.
"""
import argparse, csv, json, os, subprocess, sys, time  # noqa: E401
from datetime import datetime, timezone

HARNESS_VERSION = "wwt-o9-harness/1.0"
MODELS = {
    "qwen":     "models--Qwen--Qwen3.5-122B-A10B",
    "qwen-fp8": "models--Qwen--Qwen3.5-122B-A10B-FP8",
    "deepseek": "models--deepseek-ai--DeepSeek-V4-Flash",
}
CSV_FIELDS = [
    "outcome_id", "environment", "region_zone", "instance_type_or_pod",
    "model_id", "source", "format", "cache_state", "repeat",
    "import_s", "engine_build_s", "total_ready_s",
    "ttft_first_ms", "first_infer_ms", "first10_p50_ms", "steady_last20_ms",
    "bytes_read_gb", "serving_stack", "harness_version",
    "run_start_utc", "operator", "notes",
]

def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)

def sectors_read():
    """Sum sectors read across nvme/sd devices (whole disks only)."""
    total = 0
    for line in open("/proc/diskstats"):
        f = line.split()
        name = f[2]
        if (name.startswith(("nvme", "sd", "md")) and not name[-1].isdigit()) \
           or (name.startswith("nvme") and "p" not in name):
            total += int(f[5])
    return total

def drop_caches():
    subprocess.run(["sync"], check=True)
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3\n")
    log("page cache dropped")

def snapshot_dir(cache_dir, key):
    base = os.path.join(cache_dir, "hub", MODELS[key], "snapshots")
    snaps = sorted(os.listdir(base))
    return f"/root/.cache/huggingface/hub/{MODELS[key]}/snapshots/{snaps[-1]}"

def docker_cmd(args, key, probe_out):
    dev = (["--gpus", "all"] if args.nvidia else
           ["--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video", "--group-add", "render",
            "--security-opt", "seccomp=unconfined", "--cap-add=SYS_PTRACE"])
    return (["docker", "run", "--rm", "--ipc=host", "--shm-size=32g", *dev,
             "-v", f"{args.cache_dir}:/root/.cache/huggingface",
             "-v", f"{args.workdir}:/work", "-w", "/work",
             "-e", "HF_HUB_OFFLINE=1", "--entrypoint", "python3", args.image,
             "o9_probe.py", "--model-path", snapshot_dir(args.cache_dir, key),
             "--model-key", key, "--tp", str(args.tp),
             "--out", f"/work/{probe_out}"])

def done_cells(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["model_id"], r["cache_state"], r["repeat"])
                for r in csv.DictReader(f)}

def main():
    ap = argparse.ArgumentParser(description="O9 cold-start orchestrator (host)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--models", default="qwen,qwen-fp8,deepseek")
    ap.add_argument("--cache-states", default="cold,warm")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--cache-dir", default="/opt/huggingface-cache")
    ap.add_argument("--workdir", default=os.path.expanduser("~/dl_script/gpubench"))
    ap.add_argument("--out", default="o9_results.csv")
    ap.add_argument("--pod-label", default="MI300X-A")
    ap.add_argument("--environment", default="wwt-atc")
    ap.add_argument("--region-zone", default="on-prem")
    ap.add_argument("--operator", default=os.environ.get("SUDO_USER") or
                    os.environ.get("USER", "unknown"))
    ap.add_argument("--nvidia", action="store_true",
                    help="use --gpus all instead of AMD device flags")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    states = [s.strip() for s in args.cache_states.split(",")]
    for m in models:
        if m not in MODELS:
            sys.exit(f"unknown model {m}")
    if "cold" in states and os.geteuid() != 0:
        sys.exit("cold cells need root for drop_caches — run under sudo")
    if not os.path.exists(os.path.join(args.workdir, "o9_probe.py")):
        sys.exit(f"o9_probe.py not found in {args.workdir}")

    skip = done_cells(os.path.join(args.workdir, args.out))
    out_path = os.path.join(args.workdir, args.out)
    for key in models:
        for state in states:
            for rep in range(1, args.repeats + 1):
                if (key, state, str(rep)) in skip:
                    log(f"{key}/{state}/r{rep}: already in CSV — skipped")
                    continue
                log(f"── {key} | {state} | repeat {rep}")
                if state == "cold":
                    drop_caches()
                probe_out = f"o9_probe_{key}_{state}_{rep}.json"
                s0, t0 = sectors_read(), time.monotonic()
                started = datetime.now(timezone.utc).isoformat()
                rc = subprocess.run(docker_cmd(args, key, probe_out)).returncode
                gb = (sectors_read() - s0) * 512 / 1024**3
                if rc != 0:
                    log(f"  probe FAILED rc={rc} — recorded nothing; fix and re-run")
                    continue
                with open(os.path.join(args.workdir, probe_out)) as f:
                    p = json.load(f)
                lat = p["infer_ms"]
                first10 = sorted(lat[:10])[len(lat[:10]) // 2] if lat else ""
                steady = round(sum(lat[-20:]) / max(len(lat[-20:]), 1), 2) if lat else ""
                row = {
                    "outcome_id": "O9", "environment": args.environment,
                    "region_zone": args.region_zone,
                    "instance_type_or_pod": args.pod_label,
                    "model_id": key, "source": "local-nvme", "format": "safetensors",
                    "cache_state": state, "repeat": rep,
                    "import_s": p["import_s"], "engine_build_s": p["engine_build_s"],
                    "total_ready_s": p["total_ready_s"],
                    "ttft_first_ms": p.get("ttft_first_ms") or "",
                    "first_infer_ms": p["first_infer_ms"],
                    "first10_p50_ms": first10, "steady_last20_ms": steady,
                    "bytes_read_gb": round(gb, 1),
                    "serving_stack": f"vllm-on-instance/{p['vllm_version']}",
                    "harness_version": HARNESS_VERSION,
                    "run_start_utc": started, "operator": args.operator,
                    "notes": f"image={args.image}; tp={args.tp}; "
                             f"wall_s={round(time.monotonic()-t0,1)}; "
                             f"methodology=v1.3 O9 (source/format matrix reduced to "
                             f"local-nvme/safetensors — the only source on this node)",
                }
                new = not os.path.exists(out_path)
                with open(out_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                log(f"  ready={p['total_ready_s']}s first={p['first_infer_ms']}ms "
                    f"read={gb:.1f}GB -> {args.out}")
    log("O9 sweep complete")

if __name__ == "__main__":
    main()
