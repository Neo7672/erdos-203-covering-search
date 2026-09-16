"""INDEPENDENT VERIFICATION of the M=5040 covering for Erdos #203.

1. Re-verify every prime's group data from scratch (orders, generator, dlogs).
2. Re-check the covering: for ALL (k,l) in (Z/5040)^2, some prime's coset contains it.
   - fast numpy full-torus check
   - slow pure-int modular spot checks (including edges)
3. Construct m via CRT (+ m=1 mod 2, m=1 mod 3) and verify m*2^k*3^l+1 is
   divisible by some p_i (and > p_i) for a large random sample AND structurally.
"""
import json
from math import gcd
import numpy as np
from sympy import isprime, primerange

M = 5040
covering = json.load(open("covering_M5040_seed42.json"))
inventory = json.load(open("inventory_full.json"))

# the primes used in the covering
used = []
for p_str, c in covering.items():
    p = int(p_str)
    o2, o3, h, alpha, beta = inventory[p_str]
    used.append(dict(p=p, c=int(c), o2=int(o2), o3=int(o3), h=int(h),
                     alpha=int(alpha), beta=int(beta)))
used.sort(key=lambda d: d["h"])
print(f"=== {len(used)} primes in covering ===")

# ---- 1. verify group data from scratch ----
print("\n[1] verifying group data for each prime...")
for d in used:
    p, h, a, b = d["p"], d["h"], d["alpha"], d["beta"]
    assert isprime(p), f"p={p} not prime!"
    assert p not in (2, 3)
    # order of 2 and 3
    def order(x):
        o = p - 1
        for q, e in (lambda n: __import__("sympy").factorint(n))(p - 1).items():
            for _ in range(e):
                if pow(x, o // q, p) == 1:
                    o //= q
                else:
                    break
        return o
    assert order(2) == d["o2"], (p, "ord2")
    assert order(3) == d["o3"], (p, "ord3")
    # g = 2^? ... recover g from dlogs: we know 2 = g^a, 3 = g^b. Then g = 2^u where
    # a*u = 1 (mod h) if gcd(a,h)=1... general check: g^h = 1 and g generates
    # simplest: verify consistency: (2^k * 3^l) for (k,l) with a*k+b*l = c (mod h)
    # equals g^c for ALL c. Recover g: g = 2^(a^{-1} mod h) if gcd(a,h)=1.
    # Instead verify the STRUCTURAL property directly without g:
    # for all (k,l): 2^k 3^l depends only on (a k + b l) mod h.
    # Check: if (a*k1+b*l1) = (a*k2+b*l2) mod h then 2^k1 3^l1 = 2^k2 3^l2 (mod p).
    import random
    rng = random.Random(p)
    ok = True
    for _ in range(200):
        k1, l1 = rng.randrange(0, 4 * h), rng.randrange(0, 4 * h)
        # find (k2,l2) with same form value
        dk, dl = rng.randrange(-3, 4), rng.randrange(-3, 4)
        # want a*dk + b*dl = 0 (mod h)
        if (a * dk + b * dl) % h != 0:
            continue
        k2, l2 = k1 + dk, l1 + dl
        if (pow(2, k1, p) * pow(3, l1, p)) % p != (pow(2, k2, p) * pow(3, l2, p)) % p:
            ok = False
            print(f"  !! p={p}: inconsistency at {(k1,l1)} vs {(k2,l2)}")
            break
    assert ok, f"p={p} form consistency FAILED"
    # gcd condition (surjectivity of form map)
    g = gcd(gcd(a, b), h)
    print(f"  p={p:>5} ok: ord2={d['o2']:>3} ord3={d['o3']:>3} h={h:>3} "
          f"form=({a},{b}) c={d['c']:>3} gcd(a,b,h)={g}")
    assert g == 1, f"p={p}: gcd(a,b,h)={g} != 1 -- form map not surjective!"

# ---- 2. full torus covering check (numpy) ----
print("\n[2] full-torus covering check (numpy)...")
uncovered = np.ones((M, M), dtype=bool)
masks = []
for d in used:
    h, a, b, c = d["h"], d["alpha"], d["beta"], d["c"]
    k = np.arange(M, dtype=np.int64)
    lf = (a * k[None, :] + b * k[:, None]) % h
    mask = (lf == c)
    masks.append(mask)
    uncovered &= ~mask
n_unc = int(uncovered.sum())
print(f"    uncovered after all 24 cosets: {n_unc}")
assert n_unc == 0, "COVERING INCOMPLETE!"
print("    >>> FULL TORUS COVERED: VERIFIED <<<")

# per-cell coverage count stats (how many primes cover the average cell)
cnt = np.zeros((M, M), dtype=np.int16)
for mask in masks:
    cnt += mask
print(f"    coverage multiplicity: min={cnt.min()} mean={cnt.mean():.3f} max={cnt.max()}")

# ---- 3. pure-int spot verification ----
print("\n[3] pure-int modular spot checks...")
rng = np.random.default_rng(7)
fails = 0
for _ in range(20000):
    k = int(rng.integers(0, M)); l = int(rng.integers(0, M))
    hit = any((d["alpha"] * k + d["beta"] * l - d["c"]) % d["h"] == 0 for d in used)
    if not hit:
        fails += 1
print(f"    random cells checked: 20000, failures: {fails}")
assert fails == 0

# ---- 4. construct m via CRT ----
print("\n[4] constructing m via CRT...")
from sympy.ntheory.modular import crt
mods = [d["p"] for d in used]
res = [-pow(pow(d["g_c"] if False else 0, 0, 1), 0) for d in used]  # placeholder
# m = -g^{-c} (mod p) where g generates <2,3> and g^c = -m^{-1}
# recover g: g = 2^{a^{-1} mod h} * 3^{0}? No -- need actual g from inventory? 
# We did NOT save g. Recompute: find x with 2^x = 3 (mod p) => then g can be 2 itself
# IF 3 in <2>: for family-A primes 3 = 2^x. General: g exists with 2=g^a, 3=g^b.
# g = 2^(a^-1 mod h) works when gcd(a,h)=1. Then check 3 = g^b.
def find_g(d):
    p, h, a, b = d["p"], d["h"], d["alpha"], d["beta"]
    if gcd(a, h) == 1:
        ainv = pow(a, -1, h)
        g = pow(2, ainv, p)
        assert pow(g, h, p) == 1
        assert pow(g, a, p) == 2 % p and pow(g, b, p) == 3 % p
        return g
    if gcd(b, h) == 1:
        binv = pow(b, -1, h)
        g = pow(3, binv, p)
        assert pow(g, h, p) == 1
        assert pow(g, a, p) == 2 % p and pow(g, b, p) == 3 % p
        return g
    # gcd(a,h)>1 and gcd(b,h)>1: find g by brute force among 2^u 3^v
    import itertools
    for u in range(h):
        for v in range(h):
            g = pow(2, u, p) * pow(3, v, p) % p
            if pow(g, a, p) == 2 % p and pow(g, b, p) == 3 % p:
                return g
    raise RuntimeError(f"no g found for p={p}")

residues = []
for d in used:
    g = find_g(d)
    d["g"] = g
    # m = -g^{-c} (mod p)
    m_p = (-pow(g, -d["c"], d["p"])) % d["p"]
    residues.append(m_p)
    # sanity: m*2^k*3^l = -1 mod p for ALL (k,l) with a k + b l = c (mod h)
    rng2 = np.random.default_rng(d["p"])
    for _ in range(50):
        k = int(rng2.integers(0, 3 * d["h"])); l = int(rng2.integers(0, 3 * d["h"]))
        if (d["alpha"] * k + d["beta"] * l) % d["h"] == d["c"] % d["h"]:
            v = (m_p * pow(2, k, d["p"]) * pow(3, l, d["p"]) + 1) % d["p"]
            assert v == 0, (d["p"], k, l)
print("    per-prime residue checks passed")

r, mod = crt(mods, residues)
mod = int(mod); r = int(r) % mod
# extend: m = r + t*mod, want m = 1 (mod 2) and 1 (mod 3) => m = 1 (mod 6)
# mod is coprime to 6 (all p_i > 3). Solve r + t*mod = 1 (mod 6)
mt6 = pow(mod % 6, -1, 6)
t0 = ((1 - r) % 6) * mt6 % 6
m = r + t0 * mod
assert m % 2 == 1 and m % 3 == 1 and m >= 1
assert all(m % d["p"] == d["m_p"] for d in used)
print(f"    m = {m}")
print(f"    m has {len(str(m))} digits; gcd(m,6) = {gcd(m, 6)}")
print(f"    m+1 = {m+1}  isprime={isprime(m+1)}")

# ---- 5. end-to-end verification: for random (k,l), m*2^k*3^l+1 composite ----
print("\n[5] end-to-end spot verification (random k,l up to 10^6)...")
rng3 = np.random.default_rng(2024)
bad = 0
for _ in range(3000):
    k = int(rng3.integers(0, 10**6)); l = int(rng3.integers(0, 10**6))
    N = m * 2**k * 3**l + 1
    # find the covering prime (via reduced residues)
    kr, lr = k % M, l % M
    div = None
    for d in used:
        if (d["alpha"] * kr + d["beta"] * lr - d["c"]) % d["h"] == 0:
            div = d["p"]; break
    assert div is not None
    if N % div != 0 or N == div:
        bad += 1
        print(f"    !! FAIL at k={k} l={l} p={div}")
print(f"    checked 3000 random (k,l) in [0,10^6)^2: failures={bad}")
assert bad == 0
print("\n=== ALL VERIFICATION PASSED ===")
json.dump({"m": str(m), "primes": [{"p": d["p"], "h": d["h"], "alpha": d["alpha"],
                                     "beta": d["beta"], "c": d["c"], "g": d["g"]} for d in used]},
          open("solution_erdos203.json", "w"), indent=1)
print("saved solution_erdos203.json")
