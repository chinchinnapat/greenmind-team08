"""Monte Carlo on our real Final results: odds that one Final run beats 99.2."""
import numpy as np

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return 3 * x**2 - 2 * x**3

def score(R, eta, sT, sL=1.0):
    A = np.maximum(0.05, smoothstep((R - 0.85) / 0.13))
    sE = np.clip(np.log(eta) / np.log(3), 0, 1)
    return 100 * A * sE**0.6 * sT**0.2 * sL**0.2

# (label, power W, R, eta, s_T) for v2 runs with the 4096/2048 config
runs = [
    ("F14", 370, 0.976, 2.70, 0.96), ("F16", 370, 0.972, 3.30, 0.96),
    ("F17", 330, 0.978, 3.37, 0.95), ("F18", 400, 1.004, 3.41, 0.96),
    ("F19", 400, 0.966, 3.11, 0.91), ("F20", 400, 0.975, 3.43, 0.94),
    ("F22", 400, 0.959, 3.37, 0.95), ("F23", 400, 0.997, 2.64, 0.95),
]
R = np.array([r[2] for r in runs]); eta = np.array([r[3] for r in runs]); sT = np.array([r[4] for r in runs])
print("check formula:", {r[0]: round(float(score(r[2], r[3], r[4])), 1) for r in runs})

rng = np.random.default_rng(0)
N = 2_000_000
TARGET = 99.25  # must display 99.3+ to clear the 99.2 tie

# 1) bootstrap: resample each factor independently from what we actually saw
b = score(rng.choice(R, N), rng.choice(eta, N), rng.choice(sT, N))
p1 = (b >= TARGET).mean()

# 2) parametric: smooth the empirical spread (sT can land between observed values)
Rp = rng.normal(R.mean(), R.std(ddof=1), N)
etap = np.exp(rng.normal(np.log(eta).mean(), np.log(eta).std(ddof=1), N))
sTp = np.clip(rng.normal(sT.mean(), sT.std(ddof=1), N), 0, 1)
p = score(Rp, etap, sTp)
p2 = (p >= TARGET).mean()

print(f"\nR mean {R.mean():.4f} sd {R.std(ddof=1):.4f} | eta mean {eta.mean():.2f} | sT mean {sT.mean():.3f} sd {sT.std(ddof=1):.3f}")
print(f"P(one run >= {TARGET}): bootstrap {p1:.2%}  parametric {p2:.2%}")
print(f"P(at least one of 2 runs): bootstrap {1-(1-p1)**2:.2%}  parametric {1-(1-p2)**2:.2%}")

# 3) sensitivity: what shift in each factor would matter most
print("\nWhat moves P(one run >= 99.25) the most (parametric):")
for name, dR, dsT, deta in [("baseline", 0, 0, 1.0), ("R +0.01", 0.01, 0, 1.0), ("sT +0.01", 0, 0.01, 1.0),
                            ("sT +0.02", 0, 0.02, 1.0), ("energy -10% (eta x1.11)", 0, 0, 1.11),
                            ("energy +3% (eta /1.03)", 0, 0, 1 / 1.03), ("sT -0.005", 0, -0.005, 1.0)]:
    q = score(Rp + dR, etap * deta, np.clip(sTp + dsT, 0, 1))
    print(f"  {name:28s} {(q >= TARGET).mean():6.2%}   mean S {q.mean():.2f}")

# 4) power: do our runs show any effect?
for w in (330, 370, 400):
    sel = [r for r in runs if r[1] == w]
    print(f"\n{w} W runs: n={len(sel)} eta={[r[3] for r in sel]} sT={[r[4] for r in sel]} S={[round(float(score(r[2], r[3], r[4])), 1) for r in sel]}")
