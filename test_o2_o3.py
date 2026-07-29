#!/usr/bin/env python3
"""Offline tests for O2-lite and O3-lite: fake SSE vLLM server + fake docker."""
import csv, json, os, shutil, sys, threading, time  # noqa: E401
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)

ROOT = "/tmp/o23_test"
shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(f"{ROOT}/work", exist_ok=True)
os.makedirs(f"{ROOT}/bin", exist_ok=True)

# ── fake vLLM SSE server: TTFT grows 20ms per concurrent request ─────────────
class Fake(BaseHTTPRequestHandler):
    active = 0
    lock = threading.Lock()
    def do_GET(self):  # /health
        self.send_response(200); self.end_headers()
    def do_POST(self):
        with Fake.lock:
            Fake.active += 1
            mine = Fake.active
        try:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            time.sleep(0.02 * mine)                    # TTFT
            for i in range(4):
                self.wfile.write(b'data: {"choices":[{"text":"x"}]}\n\n')
                self.wfile.flush()
                time.sleep(0.002)                      # ITL
            self.wfile.write(b"data: [DONE]\n\n")
        except Exception:  # noqa: BLE001
            pass
        finally:
            with Fake.lock:
                Fake.active -= 1
    def log_message(self, *a):  # silence
        pass

servers = []
for port in (8801, 8802):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    servers.append(srv)

# ── O2 e2e ────────────────────────────────────────────────────────────────────
sys.path.insert(0, "/sessions/practical-trusting-fermi/mnt/outputs")
import run_o2_cohost as o2  # noqa: E402
o2.WINDOW_S = 0.5
o2.RAMP_BUDGET_S = 0.3
o2.start_serve = lambda *a, **k: None
o2.wait_ready = lambda *a, **k: True
o2.stop_all = lambda *a, **k: None
o2.snapshot_dir = lambda *a, **k: "/fake"

print("── O2-lite e2e (fake servers, ceiling ≈ 75)")
sys.argv = ["x", "--image", "fake:img", "--pair", "qwen-fp8,deepseek",
            "--max-workers", "256", "--out", f"{ROOT}/o2.csv", "--operator", "test"]
# fake server ports must match harness's port map (8801/8802) — they do.
t0 = time.time()
o2.main()
dur = time.time() - t0
rows = list(csv.DictReader(open(f"{ROOT}/o2.csv")))
check("8 rows (2 siloed + 2 co-tenant + 2x aggressor/victim)", len(rows) == 8,
      f"{len(rows)}")
sil = [r for r in rows if r["topology"] == "siloed"]
check("siloed ceilings near simulated limit (55-95)",
      all(55 <= int(r["siloed_ceiling"]) <= 95 for r in sil),
      str([r["siloed_ceiling"] for r in sil]))
vict = [r for r in rows if r["role"] == "victim"]
check("victim rows carry isolation scores", all(r["isolation_score_ttft"] != ""
      for r in vict), str([r["isolation_score_ttft"] for r in vict]))
check("victim baseline recorded", all(r["baseline_p99_ttft_ms"] != "" for r in vict))
check("no transport errors", all("errors=0" in r["notes"] for r in rows))
check("CSV fields", list(rows[0].keys()) == o2.CSV_FIELDS)
print(f"  (o2 e2e wall time {dur:.0f}s)")

# ── O3 parse unit + e2e with fake docker ─────────────────────────────────────
print("── O3-lite")
import run_o3_quality as o3  # noqa: E402
os.makedirs(f"{ROOT}/work/o3eval_qwen_gsm8k/sub", exist_ok=True)
json.dump({"results": {"gsm8k": {"exact_match,strict-match": 0.8123,
                                 "exact_match_stderr,strict-match": 0.011}},
           "config": {"model": "local-completions"}},
          open(f"{ROOT}/work/o3eval_qwen_gsm8k/sub/results_1.json", "w"))
v, e, _ = o3.parse_results(f"{ROOT}/work", "o3eval_qwen_gsm8k", "gsm8k",
                           "exact_match,strict-match")
check("parse_results value+stderr", v == 0.8123 and e == 0.011, f"{v},{e}")

# fake docker for eval runs: writes a results json per task
with open(f"{ROOT}/bin/docker", "w") as f:
    f.write(f"""#!/bin/bash
task=""; outdir=""; prev=""
for a in "$@"; do
  [ "$prev" = "--tasks" ] && task="$a"
  [ "$prev" = "--output_path" ] && outdir="$a"
  prev="$a"
done
if [ -n "$task" ]; then
  d="{ROOT}/work/$(basename $outdir)/x"; mkdir -p "$d"
  if [ "$task" = "gsm8k" ]; then
    echo '{{"results":{{"gsm8k":{{"exact_match,strict-match":0.79,"exact_match_stderr,strict-match":0.012}}}}}}' > "$d/results_9.json"
  else
    echo '{{"results":{{"mmlu":{{"acc,none":0.7412,"acc_stderr,none":0.008}}}}}}' > "$d/results_9.json"
  fi
fi
exit 0
""")
os.chmod(f"{ROOT}/bin/docker", 0o755)
os.environ["PATH"] = f"{ROOT}/bin:" + os.environ["PATH"]
o3.start_serve = lambda *a, **k: None
o3.wait_ready = lambda *a, **k: True
sys.argv = ["x", "--serve-image", "s:img", "--eval-image", "e:img",
            "--models", "qwen", "--workdir", f"{ROOT}/work",
            "--out", "o3.csv", "--operator", "test"]
o3.main()
rows = list(csv.DictReader(open(f"{ROOT}/work/o3.csv")))
check("2 task rows", len(rows) == 2, f"{len(rows)}")
check("gsm8k + mmlu values", {r["task"]: r["value"] for r in rows} ==
      {"gsm8k": "0.79", "mmlu": "0.7412"}, str({r["task"]: r["value"] for r in rows}))
check("O3 CSV fields", list(rows[0].keys()) == o3.CSV_FIELDS)
o3.main()   # resume
rows2 = list(csv.DictReader(open(f"{ROOT}/work/o3.csv")))
check("O3 resume-skip", len(rows2) == 2, f"{len(rows2)}")

[s.shutdown() for s in servers]
print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
