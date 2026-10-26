#!/usr/bin/env bash
# Copy the processor files vLLM needs into a quantized checkpoint (the quantizer only saves the tokenizer)
# Usage: quant/post_quant.sh /models/mango-w8a8-gdn
DST=${1:-/models/mango-w8a8-gdn}
for f in preprocessor_config.json processor_config.json chat_template.jinja; do
  [ -f /models/mango-bf16/$f ] && cp -n /models/mango-bf16/$f "$DST"/ && echo "copied $f"
done
