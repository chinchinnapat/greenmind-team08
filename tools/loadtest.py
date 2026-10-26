#!/usr/bin/env python3
"""Open-loop load test that mimics the GreenMind grader: ramp -> peak -> quiet.
Measures SLO pass rate, TTFT/TPOT and GPU joules (NVML) so configs can be compared
without spending grader submissions. Numbers are relative, not equal to the grader's.
Never run it while the grader is measuring."""
import argparse, asyncio, json, random, time
from collections import Counter

import httpx
import pynvml
from openai import AsyncOpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000/v1")
ap.add_argument("--model", default="mango")
ap.add_argument("--ramp", type=float, default=40)
ap.add_argument("--peak", type=float, default=40)
ap.add_argument("--quiet", type=float, default=60)
ap.add_argument("--peak-rate", type=float, default=4.0)
ap.add_argument("--quiet-rate", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--label", default="run")
args = ap.parse_args()

SENTS = [
    "The city council met on Tuesday to review the annual budget and the new bus routes.",
    "Flood prevention along the river was delayed because two contractors withdrew their bids.",
    "Three primary schools will be renovated before the rainy season starts next year.",
    "The public health campaign focused on dengue prevention and free vaccination days.",
    "Local businesses asked for lower parking fees in the old town during weekends.",
    "A new recycling program collected forty tons of plastic in its first two months.",
    "The mayor proposed solar panels on all government buildings by the end of 2028.",
    "Residents complained about noise from night construction near the central market.",
    "The water authority reported that leakage in old pipes wastes almost a fifth of supply.",
    "Tourism revenue grew strongly after the airport added direct flights to three cities.",
]
CHAT = ["Give me three tips to sleep better.", "Explain what a VPN does in simple terms.",
        "Write a short polite email asking my manager for a day off on Friday.",
        "What is the difference between weather and climate?", "Suggest a 3-day itinerary for Chiang Mai."]
THAI = ["อธิบายวิธีทำไข่เจียวให้ฟูกรอบ", "สรุปข้อดีข้อเสียของการทำงานที่บ้าน",
        "แนะนำวิธีออมเงินสำหรับนักศึกษา", "อธิบายความแตกต่างระหว่าง AI กับ Machine Learning แบบเข้าใจง่าย"]
MATH = ["A shop sells pens at 12 baht each and gives 1 free pen for every 5 bought. How much do 23 pens cost?",
        "Solve for x: 3x^2 - 12x + 9 = 0.",
        "What is the sum of all integers from 1 to 250 that are divisible by 3 or 5?",
        "Write a Python function that returns the n-th Fibonacci number in O(log n) time."]
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_flights", "description": "Search flights",
     "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "dest": {"type": "string"},
                    "date": {"type": "string"}}, "required": ["origin", "dest", "date"]}}},
]


def doc(rng, words):
    out, n = [], 0
    while n < words:
        s = rng.choice(SENTS)
        out.append(s)
        n += len(s.split())
    return " ".join(out)


def make_request(rng):
    # Same class mix as the Final track: chat 30%, RAG 20%, tools 20%, reasoning 15%, Thai 10%, long 5%
    r = rng.random()
    if r < 0.30:
        return "chat", [{"role": "user", "content": rng.choice(CHAT)}], None
    if r < 0.50:
        q = "\n\nQuestion: According to the text above, what problems did the city report?"
        return "rag", [{"role": "user", "content": doc(rng, rng.randint(3000, 5000)) + q}], None
    if r < 0.70:
        q = "Find me a flight from Bangkok to Tokyo on 2026-11-02 and tell me the weather in Tokyo."
        return "tool", [{"role": "user", "content": q}], TOOLS
    if r < 0.85:
        return "math", [{"role": "user", "content": rng.choice(MATH)}], None
    if r < 0.95:
        return "thai", [{"role": "user", "content": rng.choice(THAI)}], None
    q = "\n\nQuestion: Summarize the text above in one sentence."
    return "longctx", [{"role": "user", "content": doc(rng, rng.randint(12000, 16000)) + q}], None


def schedule():
    t, out = 0.0, []
    while t < args.ramp + args.peak + args.quiet:
        if t < args.ramp:
            phase, rate = "ramp", max(args.peak_rate * t / args.ramp, 0.25)
        elif t < args.ramp + args.peak:
            phase, rate = "peak", args.peak_rate
        else:
            phase, rate = "quiet", args.quiet_rate
        out.append((t, phase))
        t += 1.0 / rate
    return out


async def one(client, t0, at, phase, kind, msgs, tools):
    await asyncio.sleep(max(0.0, t0 + at - time.perf_counter()))
    start = time.perf_counter()
    first = last = usage = finish = err = None
    kw = dict(model=args.model, messages=msgs, stream=True, stream_options={"include_usage": True},
              max_tokens=8192, temperature=0,
              extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    if tools:
        kw.update(tools=tools, tool_choice="auto")
    try:
        async for ch in await client.chat.completions.create(**kw):
            if ch.usage:
                usage = ch.usage
            if not ch.choices:
                continue
            c = ch.choices[0]
            d = c.delta
            if d.content or d.tool_calls or getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None):
                last = time.perf_counter()
                first = first or last
            finish = c.finish_reason or finish
    except Exception as e:
        err = repr(e)[:200]
    pt = usage.prompt_tokens if usage else 0
    out = usage.completion_tokens if usage else 0
    ttft = (first - start) if first else None
    tpot = (last - first) / (out - 1) if first and out > 1 else None
    ttft_ratio = ttft / (0.5 + 0.25 * pt / 1000) if ttft is not None else None
    if err:
        fail = "error"
    elif first is None:
        fail = "no_token"
    elif finish == "length":
        fail = "cutoff"
    elif ttft_ratio > 1:
        fail = "ttft"
    elif tpot is not None and tpot > 0.05:
        fail = "tpot"
    else:
        fail = None
    return dict(phase=phase, kind=kind, ok=fail is None, fail=fail, prompt=pt, out=out,
                ttft=ttft, ttft_ratio=ttft_ratio, tpot=tpot, err=err)


def p95(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[int(0.95 * (len(xs) - 1))] if xs else float("nan")


def report(res, joules, wall):
    met = sum(r["ok"] for r in res)
    peak = [r for r in res if r["phase"] == "peak"]
    print(f"\n=== {args.label} ===")
    print(f"SLO-met        {met}/{len(res)}   (peak {sum(r['ok'] for r in peak)}/{len(peak)})")
    print(f"GPU energy     {joules:,.0f} J over {wall:.0f} s  (avg {joules / wall:.0f} W)")
    print(f"J/request      {joules / len(res):.0f}")
    print(f"eta            {met / joules:.5f} SLO-met req/J   <- higher is better")
    print(f"p95 TTFT/SLO   {p95([r['ttft_ratio'] for r in res]):.2f}   (<=1 is on target)")
    print(f"p95 TPOT       {p95([r['tpot'] for r in res]) * 1000:.0f} ms")
    print(f"avg out tokens {sum(r['out'] for r in res) / len(res):.0f}")
    print(f"fail reasons   {dict(Counter(r['fail'] for r in res if not r['ok']))}")
    for k in sorted({r["kind"] for r in res}):
        rs = [r for r in res if r["kind"] == k]
        print(f"  {k:8s} {sum(r['ok'] for r in rs)}/{len(rs)}")
    with open(f"loadtest_{args.label}.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")


async def main():
    rng = random.Random(args.seed)
    plan = [(t, ph, *make_request(rng)) for t, ph in schedule()]
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    http = httpx.AsyncClient(limits=httpx.Limits(max_connections=2000, max_keepalive_connections=2000),
                             timeout=httpx.Timeout(900.0))
    client = AsyncOpenAI(base_url=args.url, api_key="none", http_client=http)
    print(f"{args.label}: {len(plan)} requests over {args.ramp + args.peak + args.quiet:.0f}s + drain ...", flush=True)
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
    t0 = time.perf_counter() + 1.0
    res = await asyncio.gather(*(one(client, t0, *p) for p in plan))
    joules = (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) - e0) / 1000
    report(res, joules, time.perf_counter() - t0)


asyncio.run(main())
