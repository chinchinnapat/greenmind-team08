#!/usr/bin/env bash
# Best Final config (S = 77.6): INT8 W8A8 quantized by us from official MANGO BF16
source ~/venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
vllm serve /models/mango-w8a8 --served-model-name mango --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-model-len 40960 --max-num-batched-tokens 4096 --max-num-seqs 256 \
  --limit-mm-per-prompt '{"image":0,"video":0}'
