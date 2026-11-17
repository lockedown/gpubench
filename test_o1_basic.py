#!/usr/bin/env python3
"""Offline checks for the O1 pieces that need no torch/vLLM: platform detection,
per-node resume key, CSV layout guard. Run: python3 test_o1_basic.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_o1_benchmark_v2 as o1  # noqa: E402
import manifest  # noqa: E402

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)

for dev, gpu in [("NVIDIA B200", "b200-192gb-sxm"), ("NVIDIA B300 SXM6 AC", "b300-288gb-sxm"),
                 ("NVIDIA H200", "h200-141gb-sxm5")]:
    check(f"detect {dev}", (o1.detect_platform(dev) or {}).get("gpu_model") == gpu)
check("detect AMD arch alias", o1.detect_platform(" gfx942")["gpu_model"] == "mi300x-192gb-oam")
check("unknown GPU -> None", o1.detect_platform("NVIDIA A100") is None)

d = tempfile.mkdtemp()
p = os.path.join(d, "o1.csv")
def row(pod):
    return {f: "" for f in o1.CSV_FIELDS} | {"instance_type_or_pod": pod, "model_id": "m",
            "context_length_tokens": "2048", "kv_mode": "k"}
o1.append_row(p, row("HGX-B200-A"))
check("resume key includes node label",
      o1.done_cells(p) == {("HGX-B200-A", "m", "2048", "k")}
      and ("HGX-B300-A", "m", "2048", "k") not in o1.done_cells(p))

old = os.path.join(d, "old.csv")
open(old, "w").write("a,b\n1,2\n")
for what, fn in [("check_header", lambda: manifest.check_header(old, o1.CSV_FIELDS)),
                 ("append_row", lambda: o1.append_row(old, row("x")))]:
    try:
        fn(); check(f"{what} refuses old layout", False)
    except SystemExit:
        check(f"{what} refuses old layout", True)
manifest.check_header(os.path.join(d, "missing.csv"), o1.CSV_FIELDS)
check("check_header ok on missing file", True)

print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
