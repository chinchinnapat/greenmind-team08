#!/usr/bin/env python3
"""Warm up a freshly started vLLM server (kernel JIT, CUDA graphs) before a graded run.
Sends a short mixed burst: short chat, Thai, 3K/12K/50K-character prompts and tool calls."""
import argparse, json, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
ap.add_argument("--model", default="mango")
args = ap.parse_args()

S = "The city council reviewed the budget, transport, flood prevention, schools and public health. "
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

def req(content, tools=None, max_tokens=128):
    body = {"model": args.model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    r = urllib.request.Request(args.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        for _ in resp:
            pass

jobs = [("Explain photosynthesis briefly.", None)] * 8 + [("อธิบายวิธีทำไข่เจียว", None)] * 4 \
     + [(S * 60 + "\nSummarize.", None)] * 4 + [(S * 250 + "\nSummarize.", None)] * 2 \
     + [(S * 1100 + "\nSummarize.", None)] + [("Weather in Tokyo?", TOOLS)] * 4
t0 = time.time()
threads = [threading.Thread(target=req, args=j) for j in jobs]
for t in threads:
    t.start(); time.sleep(0.1)
for t in threads:
    t.join()
print(f"warm-up {len(jobs)} requests done in {time.time() - t0:.1f}s")
