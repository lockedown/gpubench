"""Shared manifest helpers (Appendix B fields + safe CSV append). stdlib only.

Imported by every harness; keep this file next to them (O1 gets it through the
/work bind-mount of the repo)."""
import csv
import os
import subprocess


def image_sha(image):
    """container_image_sha: the image digest. Already a digest -> as is; else ask
    docker for the registry digest, falling back to the image ID for locally built
    images (wwt/vllm-bench:o1 has no RepoDigest). 'unknown' if unresolvable."""
    if image and "@sha256:" in image:
        return image
    for fmt in ("{{index .RepoDigests 0}}", "{{.Id}}"):
        try:
            out = subprocess.run(["docker", "inspect", "--format", fmt, image],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
        if "@sha256:" in out or out.startswith("sha256:"):
            return out
    return "unknown"


def firmware():
    """firmware_version: distinct GPU VBIOS strings from nvidia-smi ('unknown' on
    AMD or when nvidia-smi is absent)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=vbios_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout
        vals = sorted({v.strip() for v in out.splitlines() if v.strip()})
        return "|".join(vals) or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def reasoning_mode(model_id):
    """Appendix B reasoning_mode. DeepSeek-V4 is pinned non-think; every harness
    sends raw prompts (no chat template), so no think tokens are produced.
    shortcut: Qwen3.5 reported 'n/a' — confirm with WWT whether it counts as a reasoning model."""
    return "non-think" if str(model_id).startswith("deepseek") else "n/a"


def check_header(path, fields):
    """Exit if `path` exists with a different column layout than `fields` (a changed
    schema appended to an old CSV would silently shift every column). Call at
    start-up so a resumed run fails now, not hours in at the first write."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != list(fields):
            raise SystemExit(
                f"FATAL: {path} has a different column layout than this harness "
                f"version writes. Use a new --out file (one per node/run).")


def append_row(path, fields, row):
    """Append a row (header guard re-checked on every write)."""
    check_header(path, fields)
    new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)
