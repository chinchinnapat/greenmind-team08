"""Second sweep: long-prefill threshold at a fixed 4096 budget (same GPU memory, so the KV/prefix
cache is unchanged), across both prefill-power assumptions and power caps."""
import copy, sys
from sim import make_trace, run, summarize

seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
traces = [make_trace(s + 1000) for s in seeds]  # fresh seeds, not the ones used to calibrate
def batch(**kw): return [run(copy.deepcopy(t), **kw) for t in traces]

for e_pf in (0.012, 0.018):
    print(f"\n== budget 4096, e_pf={e_pf} ==")
    for th in (512, 768, 1024, 1536, 2048):
        for cap in (375, 400):
            summarize(f"threshold {th:4d} @{cap}W", batch(threshold=th, cap=cap, e_pf=e_pf))
