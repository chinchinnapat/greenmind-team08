from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

SRC = "/models/mango-bf16"
DST = "/models/mango-w8a8"

model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained(SRC)

# Open calibration data: English chat + Thai instructions (keeps Thai accuracy)
en = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=0).select(range(384))
en = en.map(lambda x: {"text": tok.apply_chat_template(x["messages"], tokenize=False)}, remove_columns=en.column_names)
th = load_dataset("airesearch/WangchanThaiInstruct", split="train").shuffle(seed=0).select(range(128))
th = th.map(lambda x: {"text": tok.apply_chat_template(
    [{"role": "user", "content": x["Instruction"] + "\n" + (x["Input"] or "")},
     {"role": "assistant", "content": x["Output"]}], tokenize=False)}, remove_columns=th.column_names)
ds = concatenate_datasets([en, th])
ds = ds.map(lambda x: tok(x["text"], max_length=2048, truncation=True, add_special_tokens=False),
            remove_columns=["text"])

recipe = GPTQModifier(
    targets="Linear",
    scheme="W8A8",
    # Keep sensitive / non-standard parts in BF16
    ignore=["lm_head", "re:.*linear_attn.*", "re:.*mtp.*", "re:.*visual.*"],
)

oneshot(model=model, processor=tok, dataset=ds, recipe=recipe, max_seq_length=2048, num_calibration_samples=len(ds))
model.save_pretrained(DST, save_compressed=True)
tok.save_pretrained(DST)
print("DONE", DST)
