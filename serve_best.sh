#!/usr/bin/env bash
# Best Final config (S = 93.9): INT8 W8A8 (ours) + long-prefill threshold, GPU power limit 367 W
source ~/venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 367
vllm serve /models/mango-w8a8 --served-model-name mango --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-model-len 40960 --max-num-batched-tokens 4096 --max-num-seqs 256 \
  --long-prefill-token-threshold 2048 \
  --limit-mm-per-prompt '{"image":0,"video":0}'
