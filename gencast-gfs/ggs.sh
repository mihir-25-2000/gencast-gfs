#!/bin/bash
set -euo pipefail

ENV_PATH="/home/mihir.more/.conda/envs/gencast-gfs"
ASSETS="/Datastorage/mihir.more/gencast-gfs/ai_models_assets"
OUTPUT_DIR="/Datastorage/mihir.more/gencast-gfs-run-20251129-0000-sst"
SST_PATH="/Datastorage/mihir.more/sst.day.2025-11-28.nc"
INIT_DATE=20251129
INIT_TIME=0000

export PATH="$ENV_PATH/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export SST_INJECTION_PATH="$SST_PATH"
export SST_INJECTION_DATE="$INIT_DATE"

mkdir -p "$OUTPUT_DIR"
AI_MODELS_ASSETS="$ASSETS" \
"$ENV_PATH/bin/ai-models-gfs" gencast \
  --input gfs \
  --date "$INIT_DATE" \
  --time "$INIT_TIME" \
  --lead-time 24 \
  --assets "$ASSETS" \
  --assets-sub-directory \
  --download-assets \
  --num-ensemble-members 20 \
  --output file \
  --path "$OUTPUT_DIR/gc_gfs_{step}.grib2"
