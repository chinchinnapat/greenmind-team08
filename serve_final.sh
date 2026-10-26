#!/usr/bin/env bash
# Config ที่ใช้ส่ง Final รอบสุดท้าย: เหมือน serve_best.sh แต่เพิ่ม gpu-memory-utilization เป็น 0.95
# (KV cache 744K -> 780K โทเค็น) แล้วรัน tools/warmup.py หลัง server ขึ้นก่อนส่ง grader
source ~/venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 400
vllm serve /models/mango-w8a8-gdn --served-model-name mango --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-model-len 40960 --max-num-batched-tokens 4096 --max-num-seqs 256 \
  --long-prefill-token-threshold 2048 --gpu-memory-utilization 0.95 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --async-scheduling
