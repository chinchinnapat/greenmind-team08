# GreenMind LLM Inference Optimization – ทีม 08

[English](README.md)

ให้บริการโมเดล **CMKL/MANGO1.5-Qwen3.5-9B** บน NVIDIA A100-SXM4 40GB หนึ่งตัวด้วย vLLM โดยเน้นประหยัดพลังงาน

**คะแนน Final สูงสุด: S = 99.2** — A 1.00 · ประสิทธิภาพพลังงาน 3.41 เท่าของ baseline · ตอบทันเวลา 284/300 · p95 TTFT 0.81 วินาที · p95 TPOT 44 ms · 171 จูลต่อ request
ใช้พลังงานต่อคำตอบที่ทันเวลา **น้อยกว่า baseline ประมาณ 71%** (~613 จูล → ~180 จูล)

## ชุดที่ได้คะแนนสูงสุด → [`serve_best.sh`](serve_best.sh)
- **โมเดล: INT8 W8A8 รุ่น v2** ที่ทีม quantize เองระหว่างงานจาก weight ทางการ ด้วย GPTQ (weight INT8 + activation INT8 แบบ dynamic ต่อโทเค็น)
  - เป็น INT8: MLP ทั้ง 32 ชั้น, full attention 8 ชั้น และ projection ตัวใหญ่ของ linear attention (`in_proj_qkv`, `in_proj_z`, `out_proj`) 24 ชั้น
  - คงเป็น BF16: ตัวคำนวณ gate ของ linear attention (`in_proj_a`, `in_proj_b`), `lm_head` และ MTP head
  - script: [`quant/quant_w8a8_gdn.py`](quant/quant_w8a8_gdn.py)
- **ข้อมูล calibration แบบเปิด:** `HuggingFaceH4/ultrachat_200k` 384 ตัวอย่าง + `airesearch/WangchanThaiInstruct` 128 ตัวอย่าง
- **Engine:** vLLM 0.32.0, FlashInfer, prefix caching (hit rate 19–51%)
- **Scheduler:** `--max-num-batched-tokens 4096 --max-num-seqs 256 --long-prefill-token-threshold 2048 --async-scheduling`
- **GPU:** power limit 400 W (ค่าเริ่มต้นของ GPU), เปิด persistence mode
- **Weight บน Hugging Face:** https://huggingface.co/chinnakrub/MANGO1.5-Qwen3.5-9B-W8A8-GDN-team08

ชุดก่อนหน้า **v1** (สูงสุด 93.9): โมเดล INT8 ที่ linear attention ยังเป็น BF16 ([`quant/quant_w8a8.py`](quant/quant_w8a8.py)), 367 W, ไม่มี async scheduling

## สิ่งที่ได้ผล

### 1. quantize เป็น INT8 W8A8 เอง (27.8 → 77.6)
เมื่อเปิด thinking และมี request เข้า 4 ตัวต่อวินาที ช่วง peak จะมีงานคิดยาวทำพร้อมกันหลายร้อยตัว A100 จึงถูกจำกัดด้วย**กำลังคำนวณ** ไม่ใช่หน่วยความจำ
AWQ 4-bit ทำให้ weight เล็กลง แต่ยังคูณ matrix ด้วย FP16 (312 TFLOPS) ส่วน W8A8 ใช้หน่วยคำนวณ INT8 (624 TOPS) ซึ่งเร็วกว่า 2 เท่า
p95 TPOT ตอน peak ลดจาก 79 เหลือ 47 ms ตอบทันเพิ่มจาก 125 เป็น 252 จาก 300 โดยความแม่นยำไม่ลด

### 2. วัดก่อนแล้วค่อยแก้ TTFT (77.6 → 93.9)
`/metrics` ของ vLLM แสดงว่า TTFT เฉลี่ย = **รอคิว 19 ms + prefill 154 ms** ปัญหาจึงไม่ใช่คิว แต่เป็น prompt ยาวที่แย่งโควตา prefill
`--long-prefill-token-threshold 2048` จำกัดโควตาของ prompt ยาวต่อรอบ ให้ prompt สั้นแทรกเข้ามาได้
p95 TTFT ลดจาก 1.28 เหลือ 0.86 วินาที คะแนน latency ขึ้นจาก 0.83 เป็น 1.00

### 3. แปลง linear attention เป็น INT8 ด้วย (v2: สูงสุด 99.2)
24 จาก 32 ชั้นเป็น linear attention ซึ่ง v1 ยังคำนวณด้วย BF16 พอแปลงเป็น INT8 แล้ว TPOT ลดจาก 47–48 เหลือ 43–45 ms และพลังงานลดจาก 236–246 เหลือ 171–216 จูลต่อ request โดยความแม่นยำเท่าเดิม

### 4. เมื่อโมเดลเร็วพอ การจำกัดไฟก็ไม่จำเป็น
บน v2 ลองที่ 330, 370 และ 400 W พลังงานต่อ request แทบเท่ากัน (171–176 จูล) เพราะ GPU ใช้ไฟเฉลี่ยต่ำกว่าเพดานอยู่แล้ว แต่ที่ 400 W ช่วง peak เร็วกว่า พลังงานที่ประหยัดได้จึงมาจากประสิทธิภาพของซอฟต์แวร์ ไม่ใช่การบีบ GPU

## บันทึกการทดลอง
| รอบ | ประเภท | ชุดตั้งค่า | S | เทียบ baseline | จูล/request | p95 TTFT | p95 TPOT | ตอบทัน |
|---|---|---|---|---|---|---|---|---|
| V1 | Validation | BF16 ค่าเริ่มต้นของ vLLM | 0.0 | 0.90× | 463 | 2.05 วิ | 47 ms | 64/100 |
| V2 | Validation | BF16, batched-tokens 8192, max-seqs 512 | 0.0 | 0.97× | 447 | 1.76 วิ | 51 ms | 67/100 |
| V3 | Validation | AWQ 4-bit (ของผู้จัด), 8192 / 512 | 43.9 | 1.57× | 252 | 2.18 วิ | 55 ms | 61/100 |
| V4 | Validation | AWQ 4-bit, 4096 / 192 | 41.8 | 1.53× | 255 | 2.25 วิ | 55 ms | 60/100 |
| F1 | Final | AWQ 4-bit, 8192 / 512 | 27.8 | 1.37× | 187 | 2.21 วิ | 79 ms | 125/300 |
| F2 | Final | AWQ 4-bit, 4096 / 192 | 19.3 | 1.19× | 210 | 2.46 วิ | 82 ms | 122/300 |
| F3 | Final | INT8 v1, 4096 / 256 | 77.6 | 2.29× | 225 | 1.28 วิ | 47 ms | 252/300 |
| F4 | Final | INT8 v1 + MTP | 67.1 | 2.08× | 219 | 1.59 วิ | 47 ms | 223/300 |
| F7 | Final | INT8 v1 + long-prefill 2048, 367 W | 93.9 | 2.76× | 211 | 0.86 วิ | 45 ms | 285/300 |
| F8 | Final | เหมือน F7 (ส่งซ้ำ) | 82.0 | 2.31× | 246 | 1.04 วิ | 48 ms | 277/300 |
| F10 | Final | เหมือน F7 แต่ 350 W | 84.7 | 2.45× | 225 | 0.87 วิ | 50 ms | 269/300 |
| F11 | Final | เหมือน F7 แต่ 370 W | 85.2 | 2.37× | 239 | 0.85 วิ | 48 ms | 276/300 |
| F12 | Final | เหมือน F11 (ส่งซ้ำ) | 86.4 | 2.43× | 236 | 0.84 วิ | 47 ms | 280/300 |
| F14 | Final | INT8 v2 + async, 370 W | 93.0 | 2.70× | 216 | 0.80 วิ | 43 ms | 285/300 |
| F16 | Final | INT8 v2 + async, 370 W | 98.1 | 3.30× | 176 | 0.82 วิ | 44 ms | 284/300 |
| F17 | Final | INT8 v2 + async, 330 W | 98.9 | 3.37× | 172 | 0.97 วิ | 45 ms | 283/300 |
| **F18** | **Final** | **INT8 v2 + async, 400 W** | **99.2** | **3.41×** | **171** | **0.81 วิ** | **44 ms** | **284/300** |

## สิ่งที่ไม่ได้ผลและเหตุผล
- **จูน scheduler บน BF16** (V1 → V2): ดีขึ้นนิดเดียว เพราะ GPU ทำงานเต็มกำลังอยู่แล้ว
- **ลดงานพร้อมกันจาก 512 เหลือ 192** (F1 → F2): คะแนนลดลง เพราะคอขวดคือกำลังคำนวณต่อรอบ ไม่ใช่ความยาวคิว
- **AWQ 4-bit:** พลังงานต่อ request ลดลง แต่ยังคำนวณด้วย FP16 TPOT ตอน peak จึงเกิน 50 ms
- **MTP speculative decoding** (F3 → F4): พลังงานแทบไม่เปลี่ยน แต่การตรวจโทเค็นที่เดาไว้กินกำลังคำนวณตอน peak ตอบทันจึงลดจาก 252 เหลือ 223
- **ลดไฟเป็น 350 W บน v1** (F10): TPOT ชนเพดาน 50 ms ต้องทำให้โมเดลเร็วขึ้น (v2) ก่อนถึงจะลดไฟได้โดยไม่เสีย SLO
- **router แยกโมเดลเล็กกับโมเดลใหญ่ และการย่อ prompt:** วิเคราะห์แล้วไม่ทำ เพราะผิดกติกา (ต้องเป็น serving process เดียวและเป็นโมเดลที่สร้างจาก MANGO) และพลังงานส่วนใหญ่อยู่ที่การสร้างคำตอบ (ตัวอย่าง: prompt 19 โทเค็น เทียบกับคำตอบ 312 โทเค็น ซึ่ง 289 โทเค็นเป็นการคิด)

## ความแกว่งของผล
ชุดตั้งค่าเดียวกันได้คะแนนต่างกันระหว่างรอบ (v1: 82.0–93.9, v2: 93.0–99.2) ถึงจะใช้ temperature 0 แต่การจัด batch ที่ต่างกันทำให้ความยาวของการคิดเปลี่ยนไป request ที่คิดยาวผิดปกติช่วงท้ายจะยืดเวลา drain และเพิ่มพลังงาน แต่ทุกรอบของ v2 ยังดีกว่า baseline 2.7–3.4 เท่า

## ปัญหาที่เจอและวิธีแก้
- vLLM หา `nvcc` ไม่เจอตอนคอมไพล์ kernel → ติดตั้ง `ninja`, `flashinfer-cubin` และ CUDA toolkit 12.9 แล้วตั้ง `CUDA_HOME`
- llm-compressor พยายามโหลด video processor (ต้องใช้ torchvision) → ส่ง tokenizer เข้าไปตรง ๆ ด้วย `processor=tok`
- vLLM เปิดโมเดลที่ quantize แล้วไม่ได้ → คัดลอก `preprocessor_config.json`, `processor_config.json` และ `chat_template.jinja` จากโมเดลต้นฉบับ และปิดการรับรูปภาพด้วย `--limit-mm-per-prompt '{"image":0,"video":0}'`
- ค่าจำกัดไฟ GPU ถูกรีเซ็ต → เปิด persistence mode ก่อนตั้งค่า
- process เก่าของ vLLM ค้างและจอง GPU → ปิดก่อนเปิด server ใหม่ทุกครั้ง

## วิธีทำซ้ำ
1. ติดตั้ง vLLM 0.32.0 ใน `~/venv` และ llm-compressor ใน `~/venv-quant` แยกกัน, ติดตั้ง CUDA toolkit 12.9
2. ดาวน์โหลด `CMKL/MANGO1.5-Qwen3.5-9B` ไว้ที่ `/models/mango-bf16`
3. `TOKENIZERS_PARALLELISM=false python quant/quant_w8a8_gdn.py` (~15 นาทีบน A100)
4. คัดลอก `preprocessor_config.json`, `processor_config.json`, `chat_template.jinja` จาก `/models/mango-bf16` ไปที่ `/models/mango-w8a8-gdn`
5. `./serve_best.sh`

## การปฏิบัติตามกติกา
- โมเดลสร้างจาก `CMKL/MANGO1.5-Qwen3.5-9B` ทางการ ด้วยการ quantize หลังการเทรนเท่านั้น และทำระหว่างงาน
- ข้อมูล calibration เป็นชุดข้อมูลเปิด ไม่มีข้อสอบลับปน
- ใช้ซอฟต์แวร์ open source ทั้งหมด: vLLM, llm-compressor, transformers, FlashInfer
- มี vLLM serving process เดียวบน GPU ตัวเดียว ไม่ cache คำตอบ (ใช้แค่ prefix caching) ค่า power limit คงที่ตลอดการรัน และไม่แก้ค่าการสุ่มที่ grader กำหนด

## หลังจาก F18: การทดลองเพิ่มเติม (ไม่มีรอบไหนเกิน 99.2)
| รอบ | ชุดตั้งค่า | S | R | เทียบ baseline | จูล/request | p95 TTFT | p95 TPOT | ตอบทัน | ข้อสังเกต |
|---|---|---|---|---|---|---|---|---|---|
| F19 | เหมือน F18 (ส่งซ้ำ) | 95.0 | 0.966 | 3.11× | 182 | 0.97 วิ | 47 ms | 277/300 | ความแม่นยำแกว่งลง |
| F20 | เหมือน F18 | 98.4 | 0.975 | 3.43× | 170 | 0.94 วิ | 47 ms | 285/300 | |
| F21 | budget 2048 / threshold 1024 | 97.8 | 0.974 | 3.15× | 181 | 0.97 วิ | 47 ms | 278/300 | TPOT ไม่ลดลง |
| F23 | เหมือน F18 (รอบแรกหลังรีสตาร์ท) | 92.0 | 0.997 | 2.64× | 218 | 0.75 วิ | 39 ms | 282/300 | พลังงานพุ่ง |
| F25 | budget 8192 / threshold 2048 | 89.0 | 0.968 | 2.63× | 218 | 0.84 วิ | 42 ms | 280/300 | KV cache 631K, prefix hit 18% |
| V38 | threshold 1024, gpu-mem 0.95 (Validation) | 65.5 | 0.984 | 1.75× | 346 | 0.85 วิ | 33 ms | 93/100 | prefix hit 16% |

### ข้อค้นพบ
- **prefix cache ของโมเดล hybrid ขึ้นกับขนาดก้อน prefill:** budget 4096 + threshold 2048 ได้ hit ~47–51% ส่วน threshold 1024 (16%) และ budget 8192 (18%, KV cache เล็กลงเหลือ 631K) ทำให้ต้อง prefill ซ้ำเยอะ พลังงานและ TTFT จึงแย่ลง
- **power cap ไม่ใช่ตัวแปรหลัก:** บน v2 ที่ 330/370/400 W ได้ 171–176 J/request ส่วน 400 W ช่วง peak เร็วที่สุด simulator ยืนยันว่า 375 W ทำให้ s_T ลดลงมากกว่าไฟที่ประหยัดได้
- **ความแกว่งระหว่างรอบ:** config เดียวกันได้ 92–99.2 เพราะ R แกว่ง 0.959–1.004 (การจัด batch ทำให้ thinking เปลี่ยนเส้นทาง) และ request ประมาณ 2% ที่คิดยาวเกิน 5,000 โทเค็นทำให้ช่วงท้ายรอบยาว 115–160 วินาที
- **รอบแรกหลังรีสตาร์ทมักใช้พลังงานมากกว่า** (216–218 J/request ใน 3 จาก 4 ครั้ง) จึงต้อง warm-up (`tools/warmup.py`) และไม่รีสตาร์ทระหว่างรอบ
- **Validation ใช้ทำนาย Final ไม่ได้:** J/request ของโมเดลเดียวกันแกว่ง 241–346
- **Monte Carlo (`sim/mc_real.py`):** โอกาสต่อรอบที่จะได้ ≥ 99.3 ด้วย config นี้ ~6.5% ตัวที่มีผลมากที่สุดคือ s_T (ถ้า s_T +0.02 โอกาสขึ้นเป็น ~23%)

### เครื่องมือใน repo
| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `serve_best.sh` | config ที่ได้ 99.2 (F18) |
| `serve_final.sh` | config รอบสุดท้าย (เพิ่ม `--gpu-memory-utilization 0.95`) |
| `quant/quant_w8a8.py`, `quant/quant_w8a8_gdn.py` | quantize v1 และ v2 |
| `quant/post_quant.sh` | คัดลอกไฟล์ processor หลัง quantize (ไม่อย่างนั้น vLLM เปิดไม่ขึ้น) |
| `tools/warmup.py` | warm-up server หลังเปิดก่อนส่ง grader |
| `tools/loadtest.py` | จำลองโหลดแบบ grader วัด SLO และจูล (ห้ามรันระหว่าง grader วัดผล) |
| `sim/sim.py`, `sim/sweep2.py` | simulator ของ vLLM scheduler และ power cap ที่ calibrate กับค่าที่วัดจริง |
| `sim/mc_real.py` | Monte Carlo จากผล Final จริง (ต้องมี numpy) |

### ปัญหาเพิ่มเติมที่เจอ
- อัปโหลด Hugging Face ไม่ได้ (403) → token เป็นแบบ Read และชื่อ namespace ผิด แก้โดยสร้าง token แบบ Write แล้วอัปโหลดไปที่ `chinnakrub/...`
- process ของ vLLM ตัวเก่าค้างและจอง GPU → ปิดก่อนเปิดใหม่ทุกครั้ง (`pgrep -fa "vllm serve"`)
- เห็นบอทภายนอกเรียก endpoint ใน log → ระบบจริงต้องมี API key และ firewall
