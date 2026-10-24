import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

SRC = "/models/mango-bf16"
DST = "/models/mango-w8a8-gdn"

model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained(SRC)

# Same open calibration data as v1, so only the ignore list changes
en = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=0).select(range(384))
en = en.map(lambda x: {"text": tok.apply_chat_template(x["messages"], tokenize=False)}, remove_columns=en.column_names)
th = load_dataset("airesearch/WangchanThaiInstruct", split="train").shuffle(seed=0).select(range(128))
th = th.map(lambda x: {"text": tok.apply_chat_template(
    [{"role": "user", "content": x["Instruction"] + "\n" + (x["Input"] or "")},
     {"role": "assistant", "content": x["Output"]}], tokenize=False)}, remove_columns=th.column_names)
ds = concatenate_datasets([en, th])
ds = ds.map(lambda x: tok(x["text"], max_length=2048, truncation=True, add_special_tokens=False),
            remove_columns=["text"])

# v2: also quantize the large linear-attention (GDN) projections; keep tiny gate projections in BF16
ignore = ["lm_head", "re:.*mtp.*", "re:.*visual.*"]
gdn_big, gdn_small = [], []
for name, m in model.named_modules():
    if isinstance(m, nn.Linear) and "linear_attn" in name:
        (gdn_small if m.out_features < 1024 else gdn_big).append(name)
ignore += gdn_small
print(f"GDN Linear -> INT8: {len(gdn_big)}   kept BF16: {len(gdn_small)}")
print("INT8 types:", sorted({n.split('.')[-1] for n in gdn_big}))
print("BF16 types:", sorted({n.split('.')[-1] for n in gdn_small}))
assert gdn_big, "No linear_attn Linear layers found - stop and report"

recipe = GPTQModifier(targets="Linear", scheme="W8A8", ignore=ignore)
oneshot(model=model, processor=tok, dataset=ds, recipe=recipe, max_seq_length=2048,
        num_calibration_samples=len(ds))
model.save_pretrained(DST, save_compressed=True)
tok.save_pretrained(DST)
print("DONE", DST)
