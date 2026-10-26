"""Discrete-step simulator of vLLM V1 chunked-prefill serving on one A100, replaying a
GreenMind-shaped trace (ramp -> peak 4 req/s -> quiet 1 req/s, 300 requests, thinking on).

Calibrated to what we measured on the VM: p95 TPOT ~40-45 ms, p95 TTFT ~0.8 s, ~285/300 on time,
~50-65 kJ per Final, idle 71 W, prefix-cache hit ~47%. Absolute numbers are approximate; use it for
comparisons between settings, not as a prediction of the grader.
Known gap: the prefix-cache hit rate is fixed here, so it cannot show that threshold 1024 or a
budget of 8192 collapse the hit rate on this hybrid model (measured on the VM: ~50% -> 16-18%)."""
import heapq, math, random, sys
from statistics import mean

# --- workload (Final track class mix) ---
CLASSES = [  # name, count, prompt sampler, prefix-hit fraction, output median, output sigma
    ("chat",      90, lambda r: r.lognormvariate(math.log(250), 0.5),  0.10,  600, 0.75),
    ("rag",       60, lambda r: r.uniform(3000, 9000),                 0.30,  700, 0.70),
    ("agentic",   60, lambda r: r.uniform(2000, 8000),                 0.60,  500, 0.80),
    ("reasoning", 45, lambda r: r.lognormvariate(math.log(400), 0.4),  0.10, 2600, 0.80),
    ("thai",      30, lambda r: r.lognormvariate(math.log(200), 0.4),  0.10,  700, 0.75),
    ("longctx",   15, lambda r: r.uniform(12000, 32000),               0.20,  800, 0.70),
]
T_RAMP, T_PEAK, RATE_PEAK, RATE_QUIET = 60.0, 35.0, 4.0, 1.0
MAX_TOKENS = 8192

def make_trace(seed):
    r = random.Random(seed)
    kinds = [c for c in CLASSES for _ in range(c[1])]
    r.shuffle(kinds)
    # open-loop arrivals following the ramp/peak/quiet rate shape
    t, arrivals = 0.0, []
    while len(arrivals) < len(kinds):
        if t < T_RAMP: rate = max(RATE_PEAK * t / T_RAMP, 0.3)
        elif t < T_RAMP + T_PEAK: rate = RATE_PEAK
        else: rate = RATE_QUIET
        t += r.expovariate(rate)
        arrivals.append(t)
    reqs = []
    for a, (name, _, ps, hit, med, sig) in zip(arrivals, kinds):
        prompt = int(ps(r))
        out_raw = int(r.lognormvariate(math.log(med), sig))
        reqs.append(dict(arr=a, kind=name, prompt=prompt, pf=max(16, int(prompt * (1 - hit))),
                         out=max(1, min(out_raw, MAX_TOKENS)), cut=out_raw >= MAX_TOKENS,
                         peak=T_RAMP <= a < T_RAMP + T_PEAK))
    return reqs

# --- GPU cost / power model (full clock) ---
T0, C_DEC, C_PF = 0.013, 0.00010, 0.000045   # s per step, per decode token, per prefill token
P_IDLE, P_STEP_BASE = 71.0, 135.0            # W idle; W floor while stepping
E_DEC, P_STATIC = 0.030, 70.0                # J per decode token; static W (does not scale with clock)
ALPHA, SCALE_FRAC = 2.5, 0.75                # dyn power ~ f^alpha; share of step time that scales with clock

def run(reqs, budget=4096, threshold=2048, max_seqs=256, cap=400.0, e_pf=0.012):
    reqs = sorted(reqs, key=lambda q: q["arr"])
    n, i, now, energy = len(reqs), 0, reqs[0]["arr"], 0.0
    waiting, decoding, heap, step = [], 0, [], 0
    done, start = 0, reqs[0]["arr"]
    while done < n:
        while i < n and reqs[i]["arr"] <= now:
            q = reqs[i]; q["left"] = q["pf"]; waiting.append(q); i += 1
        if decoding == 0 and not waiting:
            nxt = reqs[i]["arr"]; energy += P_IDLE * (nxt - now); now = nxt; continue
        # schedule one step: decodes first, then FCFS chunked prefill with the leftover budget
        B, left, pf_tokens, finished_pf = decoding, budget - decoding, 0, []
        slots = max_seqs - decoding
        for q in waiting:
            if left <= 0 or slots <= 0: break
            cap_chunk = threshold if q["pf"] > threshold else q["left"]
            c = min(q["left"], cap_chunk, left)
            if c <= 0: break
            q["left"] -= c; left -= c; pf_tokens += c; slots -= 1
            if q["left"] == 0: finished_pf.append(q)
        t = T0 + C_DEC * B + C_PF * pf_tokens
        p_dem = P_STEP_BASE + (E_DEC * B + e_pf * pf_tokens) / t
        if p_dem > cap:  # power cap: clock drops until dynamic power fits under the cap
            f = ((cap - P_STATIC) / (p_dem - P_STATIC)) ** (1 / ALPHA)
            t = t * (1 - SCALE_FRAC) + t * SCALE_FRAC / f
            p = cap
        else:
            p = p_dem
        now += t; energy += p * t; step += 1
        # finish decodes due this step
        while heap and heap[0][0] <= step:
            _, _, q = heapq.heappop(heap)
            q["end"] = now; decoding -= 1; done += 1
        for q in finished_pf:
            waiting.remove(q)
            q["first"] = now
            if q["out"] <= 1:
                q["end"] = now; done += 1
            else:
                decoding += 1; heapq.heappush(heap, (step + q["out"] - 1, id(q), q))
    # SLO bookkeeping like the grader
    met, peak_n, peak_met, ttft_r, tpots = 0, 0, 0, [], []
    for q in reqs:
        ttft = q["first"] - q["arr"]
        slo = 0.5 + 0.25 * q["prompt"] / 1000
        tpot = (q["end"] - q["first"]) / (q["out"] - 1) if q["out"] > 1 else 0.0
        ok = (not q["cut"]) and ttft <= slo and (q["out"] <= 1 or tpot <= 0.050)
        met += ok; ttft_r.append(ttft / slo); tpots.append(tpot)
        if q["peak"]: peak_n += 1; peak_met += ok
    p95 = lambda xs: sorted(xs)[int(0.95 * (len(xs) - 1))]
    sL = math.sqrt(min(1, 1 / p95(ttft_r)) * min(1, 0.050 / max(p95(tpots), 1e-9)))
    return dict(met=met, sT=peak_met / peak_n, sL=sL, E=energy, window=now - start,
                ttft_p95=p95([q["first"] - q["arr"] for q in reqs]), tpot_p95=p95(tpots),
                fail_ttft=sum((q["first"] - q["arr"]) > 0.5 + 0.25 * q["prompt"] / 1000 for q in reqs),
                fail_tpot=sum(q["out"] > 1 and (q["end"] - q["first"]) / (q["out"] - 1) > 0.05 for q in reqs),
                fail_cut=sum(q["cut"] for q in reqs))

def summarize(label, results):
    m = lambda k: mean(r[k] for r in results)
    print(f"{label:34s} met {m('met'):6.1f}  sT {m('sT'):.3f}  sL {m('sL'):.3f}  E {m('E')/1000:5.1f} kJ  "
          f"win {m('window'):5.0f}s  TTFTp95 {m('ttft_p95'):.2f}s  TPOTp95 {m('tpot_p95')*1000:4.1f}ms  "
          f"fails ttft/tpot/cut {m('fail_ttft'):.1f}/{m('fail_tpot'):.1f}/{m('fail_cut'):.1f}")

if __name__ == "__main__":
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
    traces = [make_trace(s) for s in seeds]
    import copy
    def batch(**kw): return [run(copy.deepcopy(t), **kw) for t in traces]
    print("== calibration (current config 4096/2048, 400 W) ==")
    summarize("4096/2048 @400W", batch())
    print("\n== power cap sweep (4096/2048) ==")
    for cap in (330, 350, 375, 400):
        summarize(f"4096/2048 @{cap}W", batch(cap=cap))
    print("\n== sensitivity: prefill energy per token (how hot prefill runs) ==")
    for e in (0.008, 0.018):
        for cap in (375, 400):
            summarize(f"e_pf={e} @{cap}W", batch(cap=cap, e_pf=e))
    print("\n== scheduler options @400W ==")
    for b, th in ((2048, 1024), (4096, 1024), (4096, 2048), (6144, 2048), (8192, 2048)):
        summarize(f"budget {b} / threshold {th}", batch(budget=b, threshold=th))
