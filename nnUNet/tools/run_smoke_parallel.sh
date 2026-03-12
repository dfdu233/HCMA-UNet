#!/bin/bash
set -euo pipefail

cd /root/HCMA-UNet/nnUNet

MAX_JOBS=2
EPOCHS=2
ITERS=2
MODELS=(
  SwinHR
  SwinUNETRv2
  nn
  MedNeXt
  Mamba3d
  segFormer3d
  SingleBaselinev5
)

mkdir -p /root/tf-logs/smoke

for model in "${MODELS[@]}"; do
  while [ "$(jobs -p | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 2
  done

  echo "[SMOKE] start ${model}"
  /root/miniconda3/envs/mamba2/bin/python /root/HCMA-UNet/nnUNet/tools/smoke_test_model.py \
    --model "$model" \
    --epochs "$EPOCHS" \
    --iters "$ITERS" \
    > "/root/tf-logs/smoke/${model}.log" 2>&1 &
done

wait

echo "[SMOKE] done"
for model in "${MODELS[@]}"; do
  if grep -q "\[${model}\] OK" "/root/tf-logs/smoke/${model}.log"; then
    echo "[SMOKE] ${model}: OK"
  else
    echo "[SMOKE] ${model}: FAILED"
    tail -n 30 "/root/tf-logs/smoke/${model}.log" || true
  fi
done
