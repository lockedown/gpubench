#!/usr/bin/env bash
# Fresh HGX (B200 / B300) node -> ready to run the gpubench O1/O2/O3/O9 harnesses.
# Idempotent: safe to re-run after any failure (downloads resume, steps skip if done).
# Run as root (or via sudo) from a checkout of this repo:   sudo ./setup_hgx.sh
#
# Does NOT install the NVIDIA driver or fabric manager (vendor-matched, reboot needed).
# It checks them and stops with a clear message if they are wrong.
#
# Env overrides:
#   VLLM_IMAGE   BASE image to pull. Use the digest recorded from the H200 reference runs
#                (vllm/vllm-openai@sha256:...) so the cells are comparable. The default
#                tag is a placeholder: the base digest is resolved and saved either way.
#   CUTLASS_FIX  default 1: bake the nvidia-cutlass-dsl==4.5.1 DeepSeek fix into
#                wwt/vllm-bench:o1, exactly as the AWS runbook does for the H200 image.
#                0 = use the base image untouched.
#   CACHE_DIR    HF mirror root (default /opt/huggingface-cache) -> hub layout used by all harnesses
#   MODELS       space list of HF ids to mirror
#   HF_TOKEN     optional, only if a model is gated / you hit rate limits
#   MIN_FREE_GB  disk guard (default 800)
#   SKIP_MODELS=1  skip the (large) weight download
#   SMOKE=1      after setup, bring up DeepSeek-V4 once (never run on Blackwell) to fail fast
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shortcut: tag not verified for Blackwell here. Pin by digest once chosen (README rule 1).
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.22.1}"
CACHE_DIR="${CACHE_DIR:-/opt/huggingface-cache}"
MODELS="${MODELS:-Qwen/Qwen3.5-122B-A10B Qwen/Qwen3.5-122B-A10B-FP8 deepseek-ai/DeepSeek-V4-Flash}"
MIN_FREE_GB="${MIN_FREE_GB:-800}"
MIN_DRIVER_MAJOR=580          # CUDA 13 wheels (see requirements.txt); B300 needs a recent branch
EVAL_IMAGE="wwt/vllm-bench:o3eval"
HOST="$(hostname -s)"

step() { printf '\n== %s\n' "$*"; }
fail() { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || fail "run as root (sudo ./setup_hgx.sh)"

# ── 1. Hardware + driver + fabric ────────────────────────────────────────────
step "GPU / driver / fabric checks"
command -v nvidia-smi >/dev/null || fail "nvidia-smi missing — install the NVIDIA driver (>= ${MIN_DRIVER_MAJOR}) and reboot first"
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[ "${#GPU_NAMES[@]}" -eq 8 ] || fail "expected 8 GPUs, nvidia-smi sees ${#GPU_NAMES[@]} (MIG on? GPU fell off the bus?)"
[ "$(printf '%s\n' "${GPU_NAMES[@]}" | sort -u | wc -l)" -eq 1 ] || fail "mixed GPU models: ${GPU_NAMES[*]}"
case "${GPU_NAMES[0]}" in
  *B200*|*B300*) echo "GPU: ${GPU_NAMES[0]} x8" ;;
  *) echo "WARN: '${GPU_NAMES[0]}' is not B200/B300 — continuing" ;;
esac
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
[ "${DRV%%.*}" -ge "$MIN_DRIVER_MAJOR" ] || fail "driver $DRV < ${MIN_DRIVER_MAJOR}.x — upgrade (CUDA 13 torch in vllm 0.22.1)"
echo "driver: $DRV"
MIG="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | sort -u | tr -d ' ')"
[ "$MIG" = "Disabled" ] || [ "$MIG" = "[N/A]" ] || fail "MIG mode is '$MIG' — disable it (harness HBM floor will also reject it)"
ECC="$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | sort -u | tr -d ' ')"
[ "$ECC" = "Enabled" ] || echo "WARN: ECC is '$ECC' — methodology expects ECC on"
# NVLink: TP8 needs the NVSwitch fabric up. Fabric manager must match the driver branch.
systemctl is-active --quiet nvidia-fabricmanager \
  || fail "nvidia-fabricmanager not active — install the version matching driver $DRV and 'systemctl enable --now nvidia-fabricmanager' (Blackwell HGX may also need nvlsm per vendor guide)"
NVL="$(nvidia-smi nvlink --status -i 0)"   # not piped into grep -q: SIGPIPE + pipefail = false failure
grep -q "GB/s" <<<"$NVL" || fail "no active NVLink on GPU 0 — fabric not up"
echo "fabric manager active, NVLink up"

# ── 2. Disk ──────────────────────────────────────────────────────────────────
step "disk"
mkdir -p "$CACHE_DIR"
FREE_GB=$(df -BG --output=avail "$CACHE_DIR" | tail -1 | tr -dc 0-9)
if [ "${SKIP_MODELS:-0}" != 1 ] && [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  fail "only ${FREE_GB} GB free under $CACHE_DIR (< ${MIN_FREE_GB}); set CACHE_DIR to a bigger NVMe or lower MIN_FREE_GB"
fi
echo "${FREE_GB} GB free under $CACHE_DIR"

# ── 3. Docker + NVIDIA container toolkit ─────────────────────────────────────
step "docker + nvidia-container-toolkit"
if ! command -v docker >/dev/null; then
  command -v apt-get >/dev/null || fail "docker missing and this is not apt-based — install docker manually"
  apt-get update -y && apt-get install -y docker.io curl ca-certificates
fi
systemctl enable --now docker
if ! command -v nvidia-ctk >/dev/null; then
  command -v apt-get >/dev/null || fail "nvidia-container-toolkit missing — install it manually"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y && apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker >/dev/null
systemctl restart docker

# ── 4. vLLM image: base pinned by digest, then the same wwt/vllm-bench:o1 build the
#       H200 reference used (O1 runbook §4.2) ──────────────────────────────────
step "vLLM image"
docker pull "$VLLM_IMAGE"
BASE="$(docker inspect --format='{{index .RepoDigests 0}}' "$VLLM_IMAGE")"
[ -n "$BASE" ] || fail "could not resolve a digest for $VLLM_IMAGE"
case "$VLLM_IMAGE" in *@sha256:*) ;; *) echo "WARN: base is a tag, not the reference digest — cells won't be comparable to H200 unless $BASE equals the artefact-repo digest" ;; esac
mkdir -p "$REPO_DIR/results/$HOST"
echo "$BASE" > "$REPO_DIR/results/$HOST/base_image_digest.txt"
if [ "${CUTLASS_FIX:-1}" = 1 ]; then
  TMP="$(mktemp -d)"
  cat > "$TMP/Dockerfile" <<EOF2
FROM $BASE
RUN pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 \\
 && pip install nvidia-cutlass-dsl==4.5.1
EOF2
  docker build -t wwt/vllm-bench:o1 "$TMP"
  rm -rf "$TMP"
  # shortcut: fix validated on H200 only; if the DeepSeek smoke (SMOKE=1) fails on Blackwell, retry with CUTLASS_FIX=0
  DIGEST="$(docker inspect --format='{{.Id}}' wwt/vllm-bench:o1)"   # local build: no RepoDigest, image ID is the pin
else
  DIGEST="$BASE"
fi
echo "$DIGEST" > "$REPO_DIR/results/$HOST/image_digest.txt"
echo "base:   $BASE"
echo "pinned: $DIGEST  (record as container_image_sha)"

# ── 5. Weights -> HF hub layout (O1 + O2/O3/O9 all read <CACHE_DIR>/hub/models--*) ─
if [ "${SKIP_MODELS:-0}" != 1 ]; then
  step "model weights (resumable; can take hours)"
  for M in $MODELS; do
    echo "-- $M"
    docker run --rm -e HF_HOME=/hf ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
      -v "$CACHE_DIR:/hf" --entrypoint python3 "$DIGEST" \
      -c "import sys; from huggingface_hub import snapshot_download as d; print(d(sys.argv[1]))" "$M"
    SNAP="$CACHE_DIR/hub/models--${M//\//--}/snapshots"
    ls "$SNAP"/*/*.safetensors >/dev/null 2>&1 || fail "$M: no .safetensors after download"
    [ -f "$(ls -d "$SNAP"/*/ | tail -1)config.json" ] || fail "$M: config.json missing"
  done
fi

# ── 6. lm-eval image for O3 (CPU-only; recipe from run_o3_quality.py header) ──
step "o3eval image"
if ! docker image inspect "$EVAL_IMAGE" >/dev/null 2>&1; then
  TMP="$(mktemp -d)"
  printf 'FROM python:3.12-slim\nRUN pip install --no-cache-dir "lm-eval[api]==0.4.9" "transformers<5"\n' > "$TMP/Dockerfile.o3eval"
  docker build -f "$TMP/Dockerfile.o3eval" -t "$EVAL_IMAGE" "$TMP"
  rm -rf "$TMP"
fi
docker run --rm --entrypoint lm_eval "$EVAL_IMAGE" --help >/dev/null && echo "o3eval ok"

# ── 7. Validate inside the pinned image (same path O1 uses) ──────────────────
step "O1 validate-only inside the pinned container"
KEYS=""
for M in $MODELS; do
  case "$M" in
    Qwen/Qwen3.5-122B-A10B) KEYS="$KEYS,qwen" ;;
    Qwen/Qwen3.5-122B-A10B-FP8) KEYS="$KEYS,qwen-fp8" ;;
    deepseek-ai/DeepSeek-V4-Flash) KEYS="$KEYS,deepseek" ;;
  esac
done
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v "$CACHE_DIR:/root/.cache/huggingface" -v "$REPO_DIR:/work" -w /work \
  -e HF_HUB_OFFLINE=1 --entrypoint python3 "$DIGEST" \
  run_o1_benchmark_v2.py --validate-only --models "${KEYS#,}" \
  || fail "validate-only failed — read the PASS/FATAL lines above"

# ── 8. Optional smoke: DeepSeek-V4 has never been run on Blackwell. Catch a
#       kernel/engine failure now (minutes) instead of hours into an O1 sweep.
if [ "${SMOKE:-0}" = 1 ] && [ "${SKIP_MODELS:-0}" != 1 ]; then
  step "smoke: deepseek engine bring-up (3 inferences)"
  SNAP="$(ls -d "$CACHE_DIR"/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/*/ | tail -1)"
  docker run --rm --gpus all --ipc=host --shm-size=32g \
    -v "$CACHE_DIR:/root/.cache/huggingface" -v "$REPO_DIR:/work" -w /work \
    -e HF_HUB_OFFLINE=1 --entrypoint python3 "$DIGEST" \
    o9_probe.py --model-path "/root/.cache/huggingface/${SNAP#"$CACHE_DIR"/}" \
    --model-key deepseek --inferences 3 --out "/work/results/$HOST/smoke.json" \
    || fail "deepseek smoke failed — engine does not come up on this GPU/image; see log above"
fi

cat <<EOF

DONE. Image: $DIGEST
Per-node results dir: $REPO_DIR/results/$HOST  (use it for every --out — see audit finding 2)

O1:  docker run -d --name o1run --gpus all --ipc=host --shm-size=32g \\
       -v $CACHE_DIR:/root/.cache/huggingface -v $REPO_DIR:/work -w /work \\
       -e HF_HUB_OFFLINE=1 --entrypoint python3 \$(cat results/$HOST/image_digest.txt) \\
       run_o1_benchmark_v2.py --models qwen --profile methodology --image-sha \$(cat results/$HOST/image_digest.txt) \\
       --pod-label <HGX-B200-A|HGX-B300-A> --operator <you> --out /work/results/$HOST/o1.csv
     (--profile methodology = 5 min warm-up / 15 min capture / 3 repeats, the v1.3 rule. 'standard' is dry-run only.)
O9:  sudo python3 run_o9_benchmark.py --nvidia --pod-label <label> --image \$(cat results/$HOST/image_digest.txt) \\
       --workdir $REPO_DIR --out results/$HOST/o9.csv --operator <you>
O3:  python3 run_o3_quality.py --nvidia --pod-label <label> --serve-image \$(cat results/$HOST/image_digest.txt) \\
       --eval-image $EVAL_IMAGE --workdir $REPO_DIR --out results/$HOST/o3.csv --operator <you>
O2:  sudo python3 run_o2_cohost.py --nvidia --pod-label <label> --image \$(cat results/$HOST/image_digest.txt) \\
       --pair qwen-fp8,qwen --gpus 0,1,2,3 --out results/$HOST/o2.csv --operator <you>
EOF
