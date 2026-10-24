# GreenMind LLM Inference Optimization – Team 08

Energy-efficient serving of **CMKL/MANGO1.5-Qwen3.5-9B** on 1× NVIDIA A100-SXM4 40GB with vLLM.

**Best Final score: S = 93.9** — A 1.00 · η 2.76× baseline · SLO-met 285/300 · p95 TTFT 0.86 s · p95 TPOT 45 ms · 211 J/request
≈ **64% less energy per on-time answer** than the untuned baseline (≈613 J → ≈222 J).

## Final configuration
- **Model:** INT8 W8A8 (GPTQ weights, per-token dynamic INT8 activations), quantized by us during the event from the official BF16 weights. Linear-attention (GDN) projections, `lm_head` and MTP head kept in BF16 to protect accuracy.
- **Engine:** vLLM 0.32.0, FlashInfer kernels, prefix caching on (≈19% hit rate)
- **Scheduler:** `--max-num-batched-tokens 4096 --max-num-seqs 256 --long-prefill-token-threshold 2048`
- **GPU:** power limit 367 W, persistence mode on, held constant for the whole run
- **Command:** [`serve_best.sh`](serve_best.sh) · **Quantization:** [`quant/quant_w8a8.py`](quant/quant_w8a8.py)
- **Weights:** Hugging Face link — TODO

## What worked
### 1. Self-made INT8 W8A8 (S 27.8 → 77.6)
With thinking mode on at 4 req/s, hundreds of long generations run at once during the peak phase, so A100 decoding is **compute-bound**.
AWQ-W4A16 shrinks weights but still multiplies on FP16 tensor cores (312 TFLOPS); W8A8 uses INT8 tensor cores (624 TOPS, 2×).
Peak p95 TPOT dropped 79 → 47 ms and on-time requests rose 125 → 252 / 300 with no accuracy loss.

### 2. Measure, then fix TTFT (S 77.6 → 93.9)
vLLM `/metrics` showed average TTFT = **19 ms queueing + 154 ms prefill** — the queue was not the problem; long prompts were hogging the prefill budget.
`--long-prefill-token-threshold 2048` caps each long prompt's chunk per step so short prompts get through.
p95 TTFT 1.28 → 0.86 s, latency score 0.83 → 1.00, on-time 252 → 285 / 300.

## Experiment log
| # | Track | Config | S | η vs base | J/req | p95 TTFT | p95 TPOT | SLO-met |
|---|---|---|---|---|---|---|---|---|
| V1 | Validation | BF16, vLLM defaults | 0.0 | 0.90× | 463 | 2.05 s | 47 ms | 64/100 |
| V2 | Validation | BF16, batched-tokens 8192, max-seqs 512 | 0.0 | 0.97× | 447 | 1.76 s | 51 ms | 67/100 |
| V3 | Validation | AWQ-W4A16 (organizer), 8192 / 512 | 43.9 | 1.57× | 252 | 2.18 s | 55 ms | 61/100 |
| V4 | Validation | AWQ-W4A16, 4096 / 192 | 41.8 | 1.53× | 255 | 2.25 s | 55 ms | 60/100 |
| F1 | Final | AWQ-W4A16, 8192 / 512 | 27.8 | 1.37× | 187 | 2.21 s | 79 ms | 125/300 |
| F2 | Final | AWQ-W4A16, 4096 / 192 | 19.3 | 1.19× | 210 | 2.46 s | 82 ms | 122/300 |
| F3 | Final | W8A8 (ours), 4096 / 256 | 77.6 | 2.29× | 225 | 1.28 s | 47 ms | 252/300 |
| F4 | Final | W8A8 + MTP (1 spec token) | 67.1 | 2.08× | 219 | 1.59 s | 47 ms | 223/300 |
| **F7** | **Final** | **W8A8 + long-prefill threshold 2048, 367 W** | **93.9** | **2.76×** | **211** | **0.86 s** | **45 ms** | **285/300** |
| F8 | Final | same as F7 (repeat) | 82.0 | 2.31× | 246 | 1.04 s | 48 ms | 277/300 |

## What did not work (and why)
- **Scheduler tuning on BF16** (V1→V2): small gain only; the GPU was already saturated.
- **Lowering max-num-seqs 512 → 192** (F1→F2): score dropped. The bottleneck was per-step compute, not queue depth.
- **AWQ-W4A16:** cut energy per request but kept FP16 compute, so peak TPOT stayed far above 50 ms.
- **MTP speculative decoding** (F3→F4): J/request barely moved (225→219) but verification ate peak compute, so on-time fell 252→223.
- **Query router (small vs large model) and prompt compression** — analysed and rejected: a second model breaks the single-process / MANGO-derived rules, and energy is dominated by decoding (example: 19 prompt tokens vs 312 completion tokens, 289 of them reasoning), so prompt compression saves <1% while risking RAG/tool accuracy.

## Run-to-run variance
The same config scored **93.9 and 82.0** (F7 vs F8). Batch-dependent numerics change thinking lengths even at temperature 0; one long-thinking request at the end stretches the drain window and adds ~10 kJ. Every run still beat the baseline by 2.3–2.8×.

## Setup issues we hit
- `Could not find nvcc` during FlashInfer JIT → installed `ninja`, `flashinfer-cubin` and CUDA toolkit 12.9, set `CUDA_HOME`.
- llm-compressor tried to load the video processor (needs torchvision) → passed the tokenizer directly with `processor=tok`.
- vLLM could not load the multimodal processor after quantization → copied `preprocessor_config.json`, `processor_config.json`, `chat_template.jinja` from the BF16 checkpoint; served text-only with `--limit-mm-per-prompt '{"image":0,"video":0}'`.
- GPU power limit silently reset → enabled persistence mode (`nvidia-smi -pm 1`) before setting it.

## Rules compliance
- Model derived from the official `CMKL/MANGO1.5-Qwen3.5-9B` by post-training quantization only, made during the event.
- Calibration data (open): `HuggingFaceH4/ultrachat_200k` (384 samples) + `airesearch/WangchanThaiInstruct` (128 samples). No hidden test items.
- Open-source software only: vLLM, llm-compressor, transformers, FlashInfer.
- One vLLM serving process on one GPU, no response caching (prefix caching only), power limit held constant for the whole run, grader-controlled sampling not overridden.
