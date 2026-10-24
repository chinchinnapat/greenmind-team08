# GreenMind LLM Inference Optimization – Team 08

Energy-efficient serving of **CMKL/MANGO1.5-Qwen3.5-9B** on 1× NVIDIA A100-SXM4 40GB with vLLM.

**Best Final score: S = 77.6** — A 1.00 · η 2.29× baseline · SLO-met 252/300 · p95 TTFT 1.28 s · p95 TPOT 47 ms · 225 J/request

## Final configuration
- **Model:** INT8 W8A8 (GPTQ weights, per-token dynamic INT8 activations), quantized by us during the event from the official BF16 weights. MTP head kept in BF16.
- **Engine:** vLLM 0.32.0 with FlashInfer kernels, prefix caching on (19% hit rate observed in a graded run)
- **Command:** [`serve_best.sh`](serve_best.sh)
- **Quantization script:** [`quant/quant_w8a8.py`](quant/quant_w8a8.py)
- **Weights:** Hugging Face link — TODO

## Why it works
With thinking mode on at 4 req/s, hundreds of long generations run at once during the peak phase.
At that batch size A100 decoding is **compute-bound**, not memory-bound:
- BF16 and AWQ-W4A16 still run matmuls on FP16 tensor cores (312 TFLOPS) → p95 TPOT 79–82 ms at peak → most requests miss the 50 ms SLO.
- W8A8 runs on INT8 tensor cores (624 TOPS, 2×) → p95 TPOT 47 ms → SLO-met 125 → 252 / 300.
- The energy score only counts on-time requests, so fixing SLO misses roughly doubled efficiency.

## Experiment log
| # | Track | Config | S | η vs base | J/req | p95 TTFT | p95 TPOT | SLO-met |
|---|---|---|---|---|---|---|---|---|
| V1 | Validation | BF16, vLLM defaults | 0.0 | 0.90× | 463 | 2.05 s | 47 ms | 64/100 |
| V2 | Validation | BF16, batched-tokens 8192, max-seqs 512 | 0.0 | 0.97× | 447 | 1.76 s | 51 ms | 67/100 |
| V3 | Validation | AWQ-W4A16 (organizer), 8192 / 512 | 43.9 | 1.57× | 252 | 2.18 s | 55 ms | 61/100 |
| V4 | Validation | AWQ-W4A16, 4096 / 192 | 41.8 | 1.53× | 255 | 2.25 s | 55 ms | 60/100 |
| F1 | Final | AWQ-W4A16, 8192 / 512 | 27.8 | 1.37× | 187 | 2.21 s | 79 ms | 125/300 |
| F2 | Final | AWQ-W4A16, 4096 / 192 | 19.3 | 1.19× | 210 | 2.46 s | 82 ms | 122/300 |
| **F3** | **Final** | **W8A8 (ours), 4096 / 256** | **77.6** | **2.29×** | **225** | **1.28 s** | **47 ms** | **252/300** |

## What did not work (and why)
- **Scheduler tuning on BF16** (V1→V2): small gain only; the GPU was already saturated.
- **Lowering max-num-seqs 512 → 192 on AWQ** (F1→F2): TPOT did not improve (79→82 ms) and score dropped. The bottleneck is per-step compute, not queue depth.
- **AWQ-W4A16:** cut energy per request (463 → 187 J) but kept FP16 compute, so peak TPOT stayed far above 50 ms.
- **Query router (small vs large model) and prompt compression** — analysed and rejected:
  - The rules require a MANGO-derived model and a single serving process; a second model breaks both.
  - Energy is dominated by decoding, not prompt reading (example: 19 prompt tokens vs 312 completion tokens, 289 of them reasoning), so compressing prompts saves <1%.
  - Compressing RAG / tool-calling prompts risks accuracy and the 0.575 s TTFT budget.
  - We kept the ideas at token level instead: speculative decoding (MTP) and vLLM prefix caching.

## Setup issues we hit
- `Could not find nvcc` during FlashInfer JIT → installed `ninja`, `flashinfer-cubin` and CUDA toolkit 12.9, set `CUDA_HOME`.
- After quantization, vLLM could not load the multimodal processor → copied `preprocessor_config.json`, `processor_config.json`, `chat_template.jinja` from the BF16 checkpoint and served text-only with `--limit-mm-per-prompt '{"image":0,"video":0}'`.
- llm-compressor tried to load the video processor (needs torchvision) → passed the tokenizer directly with `processor=tok`.

## Rules compliance
- Model derived from the official `CMKL/MANGO1.5-Qwen3.5-9B` by post-training quantization only, made during the event.
- Calibration data (open): `HuggingFaceH4/ultrachat_200k` (384 samples) + `airesearch/WangchanThaiInstruct` (128 samples). No hidden test items.
- Open-source software only: vLLM, llm-compressor, transformers, FlashInfer.
- One vLLM serving process on one GPU, no response caching (prefix caching only), default GPU power settings, grader-controlled sampling not overridden.
