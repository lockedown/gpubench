#!/usr/bin/env python3
"""Offline test for the O9 pair: fake docker + fake diskstats + mocked vllm probe."""
import json, os, shutil, subprocess, sys, csv  # noqa: E401

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)

ROOT = "/tmp/o9_test"
shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(f"{ROOT}/bin", exist_ok=True)
os.makedirs(f"{ROOT}/work", exist_ok=True)
os.makedirs(f"{ROOT}/cache/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/abc123", exist_ok=True)

# fake docker: writes a plausible probe JSON to the mounted workdir
with open(f"{ROOT}/bin/docker", "w") as f:
    f.write(f"""#!/bin/bash
# find --out argument (last one) and the workdir mount
out=""; prev=""
for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
fname=$(basename "$out")
python3 - "$fname" << 'EOF'
import json, sys
lat = [180.0] + [12.0 + i*0.01 for i in range(99)]
json.dump(dict(import_s=8.1, engine_build_s=142.3, total_ready_s=150.4,
               first_infer_ms=lat[0], ttft_first_ms=95.2,
               infer_ms=lat, vllm_version="0.23.1rc1-fake",
               torch_version="2.11-fake"), open("{ROOT}/work/"+sys.argv[1], "w"))
EOF
echo O9_PROBE_OK
""")
os.chmod(f"{ROOT}/bin/docker", 0o755)
open(f"{ROOT}/work/o9_probe.py", "w").write("# placeholder\n")

# import orchestrator with patched primitives
sys.path.insert(0, "/sessions/practical-trusting-fermi/mnt/outputs")
import run_o9_benchmark as h  # noqa: E402

_sect = {"v": 1000}
h.sectors_read = lambda: _sect.__setitem__("v", _sect["v"] + 250_000_000) or _sect["v"]
h.drop_caches = lambda: print("  (fake cache drop)")
os.environ["PATH"] = f"{ROOT}/bin:" + os.environ["PATH"]
os.geteuid = os.geteuid  # cold check bypassed via monkeypatch below
h.os.geteuid = lambda: 0

sys.argv = ["x", "--image", "fake:img", "--models", "qwen", "--repeats", "2",
            "--cache-states", "cold,warm", "--cache-dir", f"{ROOT}/cache",
            "--workdir", f"{ROOT}/work", "--out", "o9.csv", "--operator", "test"]
h.main()

rows = list(csv.DictReader(open(f"{ROOT}/work/o9.csv")))
check("4 rows (2 states x 2 repeats)", len(rows) == 4, f"{len(rows)}")
r = rows[0]
check("CSV fields", list(r.keys()) == h.CSV_FIELDS)
check("phases recorded", r["engine_build_s"] == "142.3" and r["total_ready_s"] == "150.4")
check("first vs steady separated", float(r["first_infer_ms"]) > 100
      and float(r["steady_last20_ms"]) < 15, f"{r['first_infer_ms']} vs {r['steady_last20_ms']}")
check("bytes read metered", abs(float(r["bytes_read_gb"]) - 119.2) < 1,
      r["bytes_read_gb"])   # 250M sectors x 512B = ~119.2GB
check("cache_state cold first", r["cache_state"] == "cold" and r["repeat"] == "1")

# resume: re-run must skip all four
h.main()
rows2 = list(csv.DictReader(open(f"{ROOT}/work/o9.csv")))
check("resume-skip on re-run", len(rows2) == 4, f"{len(rows2)}")

# probe syntax check (can't execute without vllm)
rc = subprocess.run([sys.executable, "-m", "py_compile",
                     "/sessions/practical-trusting-fermi/mnt/outputs/o9_probe.py"]).returncode
check("probe compiles", rc == 0)

print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
