#!/bin/bash
set -euo pipefail

ENV_PATH="/home/mihir.more/.conda/envs/gencast-gfs"
CONDA_BASE="${CONDA_BASE:-/home/mihir.more/miniconda3}"
ASSETS="/Datastorage/mihir.more/gencast-gfs/ai_models_assets"
LEAD=120   # hours
INIT_TIME=0000

# Disk-space guard settings
MIN_FREE_GB=50
WAIT_SEC=300
TMP_BASE="/Datastorage/mihir.more/tmp"
mkdir -p "$TMP_BASE"

has_space() {
  local path="$1"
  local need_kb=$((MIN_FREE_GB * 1024 * 1024))
  local avail_kb
  avail_kb=$(df -Pk "$path" | awk 'NR==2{print $4}')
  [[ "${avail_kb:-0}" -ge "$need_kb" ]]
}

wait_for_space() {
  local path="$1"
  while ! has_space "$path"; do
    echo "Low disk space on $path; waiting ${WAIT_SEC}s..." >&2
    sleep "$WAIT_SEC"
  done
}

run_with_space_retry() {
  local path="$1"; shift
  while :; do
    if "$@"; then return 0; fi
    if has_space "$path"; then return 1; fi
    echo "No space left on device; waiting ${WAIT_SEC}s then retrying..." >&2
    sleep "$WAIT_SEC"
  done
}

# activate conda env so runtime libs (e.g., libmpi) are available
if [[ -x "$CONDA_BASE/bin/conda" ]]; then
  # avoid nounset failures in conda activate.d scripts
  set +u
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
  conda activate "$ENV_PATH"
  set -u
else
  echo "conda not found at $CONDA_BASE; falling back to PATH/LD_LIBRARY_PATH" >&2
  export PATH="$ENV_PATH/bin:$PATH"
  export LD_LIBRARY_PATH="$ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
fi

export PATH="$ENV_PATH/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

command -v ai-models-gfs >/dev/null 2>&1 || { echo "ai-models-gfs not found in env"; exit 1; }
EXTRACT_TOOL=""
TP_WHERE=""
TP_MATCH=""

if command -v grib_copy >/dev/null 2>&1; then
  if ! grib_copy -V >/dev/null 2>&1; then
    echo "grib_copy found but not runnable; check eccodes install." >&2
    exit 1
  fi
  EXTRACT_TOOL="grib_copy"
  TP_WHERE="shortName=tp/TP"
elif command -v wgrib2 >/dev/null 2>&1; then
  EXTRACT_TOOL="wgrib2"
  TP_MATCH=":(TP|tp):"
  if ! wgrib2 -version >/dev/null 2>&1; then
    echo "wgrib2 failed to start (missing runtime libs like libmpi.so); trying module load openmpi." >&2
    if [[ -f /etc/profile.d/modules.sh ]]; then
      set +u
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
      set -u
    elif [[ -f /etc/profile.d/lmod.sh ]]; then
      set +u
      # shellcheck disable=SC1091
      source /etc/profile.d/lmod.sh
      set -u
    fi
    if command -v module >/dev/null 2>&1; then
      set +u
      module load openmpi >/dev/null 2>&1 || true
      set -u
    fi
    if ! wgrib2 -version >/dev/null 2>&1; then
      echo "wgrib2 still failed; install openmpi in the env or load an MPI module." >&2
      exit 1
    fi
  fi
else
  echo "No GRIB tool found; install eccodes (grib_copy) or wgrib2." >&2
  exit 1
fi

for day in $(seq -w 13 31); do
  INIT_DATE="202508${day}"
  OUTPUT_DIR="/Datastorage/mihir.more/gencast-ggfs-masked-run-august-2025/gencast-gfs-run-${INIT_DATE}-0000-masked-sst-5dayslead"
  TMP_DIR="$(mktemp -d -p "$TMP_BASE")"

  mkdir -p "$OUTPUT_DIR"

  wait_for_space "$TMP_DIR"
  wait_for_space "$OUTPUT_DIR"

  AI_MODELS_ASSETS="$ASSETS" \
  run_with_space_retry "$TMP_DIR" \
  "$ENV_PATH/bin/ai-models-gfs" gencast \
    --input gfs \
    --date "$INIT_DATE" \
    --time "$INIT_TIME" \
    --lead-time "$LEAD" \
    --assets "$ASSETS" \
    --assets-sub-directory \
    --download-assets \
    --num-ensemble-members 20 \
    --output file \
    --path "$TMP_DIR/gc_gfs_{step}.grib2"

  # Keep only total precipitation
  for f in "$TMP_DIR"/*.grib2; do
    out="$OUTPUT_DIR/${f##*/}"
    wait_for_space "$OUTPUT_DIR"
    if [[ "$EXTRACT_TOOL" == "grib_copy" ]]; then
      run_with_space_retry "$OUTPUT_DIR" grib_copy -w "$TP_WHERE" "$f" "$out"
    else
      run_with_space_retry "$OUTPUT_DIR" wgrib2 "$f" -match "$TP_MATCH" -grib "$out"
    fi
  done

  rm -rf "$TMP_DIR"
done
