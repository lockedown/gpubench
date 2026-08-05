#!/usr/bin/env python3
"""
Appendix A integrity gate — RedHatAI/DeepSeek-V4-Flash-BF16 router-tensor check
================================================================================
Verifies the open Hugging Face report (RedHatAI/DeepSeek-V4-Flash-BF16
discussion #1): the MoE router mapping tensors (`*.tid2eid`) are alleged to be
ALL ZEROS in the BF16 re-export, while the original deepseek-ai/DeepSeek-V4-Flash
checkpoint has real routing values. If true, expert routing is degenerate and
every result produced with this checkpoint is invalid (June MI300X DeepSeek-BF16
rows). This script runs the check against the LOCAL cache — fully offline, no
GPU, reads only the ~6MB router tensors from each shard, never the weights.

Usage (host, needs safetensors+numpy):
    pip install safetensors numpy --break-system-packages
    python3 check_redhat_dsv4_integrity.py [--cache-dir /opt/huggingface-cache]

Or inside the pinned container (nothing to install, run from ~/dl_script/gpubench):
    sudo docker run --rm \
      -v /opt/huggingface-cache:/root/.cache/huggingface \
      -v ~/dl_script/gpubench:/work -w /work -e HF_HUB_OFFLINE=1 \
      --entrypoint python3 \
      vllm/vllm-openai-rocm@sha256:3a83d5c1e0ee537cd123095f3ac2b78ffa2236701722569f464af8870b8c0393 \
      check_redhat_dsv4_integrity.py --cache-dir /root/.cache/huggingface

Exit codes: 0 = PASS (tensors match original)   2 = FAIL (broken checkpoint)
            3 = INDETERMINATE (original not local / no router keys found)
Writes verdict + per-layer evidence to redhat_dsv4_integrity.json next to itself.
"""
import argparse, glob, json, os, sys  # noqa: E401
from datetime import datetime, timezone

import numpy as np
from safetensors import safe_open

SUSPECT = "models--RedHatAI--DeepSeek-V4-Flash-BF16"
ORIGINAL = "models--deepseek-ai--DeepSeek-V4-Flash"
KEY_PATTERN = "tid2eid"          # router token->expert mapping (per HF report)


def snapshot(cache_dir, repo):
    base = os.path.join(cache_dir, "hub", repo, "snapshots")
    if not os.path.isdir(base):
        return None, None
    snaps = sorted(os.listdir(base))
    if not snaps:
        return None, None
    return os.path.join(base, snaps[-1]), snaps[-1]


def weight_map(snap):
    """key -> absolute shard path. Handles sharded + single-file checkpoints."""
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(snap, v) for k, v in wm.items()}
    out = {}
    for f in glob.glob(os.path.join(snap, "*.safetensors")):
        with safe_open(f, framework="np") as sf:
            for k in sf.keys():
                out[k] = f
    return out


def load(path, key):
    with safe_open(path, framework="np") as sf:
        return sf.get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/opt/huggingface-cache")
    ap.add_argument("--max-layers", type=int, default=0,
                    help="0 = check every router layer (default); N = first N only")
    args = ap.parse_args()

    sus_snap, sus_rev = snapshot(args.cache_dir, SUSPECT)
    org_snap, org_rev = snapshot(args.cache_dir, ORIGINAL)
    if not sus_snap:
        sys.exit(f"FATAL: {SUSPECT} not found under {args.cache_dir}/hub")
    print(f"suspect : {SUSPECT} @ {sus_rev}")
    print(f"original: {ORIGINAL} @ {org_rev or 'NOT LOCAL — zero-check only'}\n")

    sus_map = weight_map(sus_snap)
    keys = sorted([k for k in sus_map if KEY_PATTERN in k],
                  key=lambda k: [int(t) if t.isdigit() else t for t in k.split(".")])
    if not keys:
        cand = sorted({k for k in sus_map if ".gate." in k or "router" in k})[:10]
        print(f"NO '{KEY_PATTERN}' keys in suspect checkpoint. Router-ish keys "
              f"found instead (first 10): {cand}")
        print("VERDICT: INDETERMINATE — key naming differs from the HF report; "
              "inspect the candidates above.")
        sys.exit(3)
    if args.max_layers:
        keys = keys[:args.max_layers]
    org_map = weight_map(org_snap) if org_snap else {}

    rows, n_zero, n_mismatch, n_match = [], 0, 0, 0
    for k in keys:
        t = load(sus_map[k], k)
        all_zero = not np.any(t)
        row = {"key": k, "shape": list(t.shape), "dtype": str(t.dtype),
               "suspect_nonzero": int(np.count_nonzero(t)),
               "suspect_numel": int(t.size), "suspect_all_zero": bool(all_zero)}
        if all_zero:
            n_zero += 1
        if k in org_map:
            o = load(org_map[k], k)
            equal = t.shape == o.shape and bool(np.array_equal(t, o))
            row.update({"original_nonzero": int(np.count_nonzero(o)),
                        "equal_to_original": equal,
                        "differing_entries": (int(np.sum(t != o))
                                              if t.shape == o.shape else -1)})
            n_match += equal
            n_mismatch += (not equal)
        status = ("ALL-ZERO" if all_zero else
                  ("match" if row.get("equal_to_original") else
                   ("MISMATCH" if "equal_to_original" in row else "nonzero")))
        print(f"  {k:<44} nonzero {row['suspect_nonzero']:>9}/{t.size:<9} {status}")
        rows.append(row)

    print(f"\nlayers checked: {len(keys)} | all-zero: {n_zero} | "
          f"match original: {n_match} | mismatch: {n_mismatch}")
    if n_zero or n_mismatch:
        verdict, code = "FAIL — checkpoint router tensors broken; June DeepSeek-" \
                        "BF16 rows must be quarantined; do NOT pin this artifact", 2
    elif org_map and n_match == len(keys):
        verdict, code = "PASS — router tensors intact and identical to original; " \
                        "artifact admissible, June rows rehabilitated", 0
    else:
        verdict, code = "INDETERMINATE — suspect tensors are nonzero (HF report " \
                        "not reproduced) but original not local for equality check", 3
    print(f"VERDICT: {verdict}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "redhat_dsv4_integrity.json")
    json.dump({"checked_utc": datetime.now(timezone.utc).isoformat(),
               "suspect": {"repo": SUSPECT, "revision": sus_rev},
               "original": {"repo": ORIGINAL, "revision": org_rev},
               "verdict": verdict, "exit_code": code, "layers": rows},
              open(out, "w"), indent=1)
    print(f"evidence -> {out}")
    sys.exit(code)


if __name__ == "__main__":
    main()
