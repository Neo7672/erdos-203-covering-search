"""Erdos #203: search for m with m*2^k*3^l+1 always composite via 2D covering system.

Theory: prime p is usable with density 1/h_p where h_p = lcm(ord_p(2), ord_p(3)).
Condition: alpha*k + beta*l = c_p (mod h_p) where 2=g^alpha, 3=g^beta, g gen of <2,3>.
m is then fixed mod p by m = -g^{-c_p}. Covering of Z^2 by the cosets gives the result.
"""
import numpy as np
from sympy import primerange, factorint
from math import gcd, lcm

def multiplicative_order(a, p, phi_factors):
    o = p - 1
    for q, e in phi_factors.items():
        for _ in range(e):
            if pow(a, o // q, p) == 1:
                o //= q
            else:
                break
    return o

def build_inventory(pmax):
    inv = {}
    for p in primerange(5, pmax):
        phi = factorint(p - 1)
        o2 = multiplicative_order(2, p, phi)
        o3 = multiplicative_order(3, p, phi)
        h = lcm(o2, o3)
        if h > 400: continue
        # find generator of <2,3>: element of order h; try 2^a*3^b
        gen = None
        for a in range(h):
            for b in range(h):
                g = (pow(2, a, p) * pow(3, b, p)) % p
                if multiplicative_order(g, p, phi) == h:
                    gen = g; break
            if gen: break
        if gen is None: continue
        # dlogs of 2 and 3 base gen (brute force, h<=400)
        table = {}
        x = 1
        for j in range(h):
            table[x] = j
            x = (x * gen) % p
        alpha = table[2]; beta = table[3]
        assert gcd(alpha, beta, h) == 1, (p, alpha, beta, h)
        inv[p] = (o2, o3, h, gen, alpha, beta)
    return inv

if __name__ == "__main__":
    import sys
    pmax = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    print(f"scanning primes < {pmax} ...")
    inv = build_inventory(pmax)
    # save
    import json
    with open("inventory_full.json", "w") as f:
        json.dump({str(p): list(v[:3]) + [v[4], v[5]] for p, v in inv.items()}, f)
    # group by h
    from collections import defaultdict
    byh = defaultdict(list)
    for p, (o2, o3, h, gen, alpha, beta) in inv.items():
        byh[h].append((p, alpha, beta))
    print(f"total primes with h<=400: {len(inv)}")
    tot = sum(1.0 / h for h in byh for _ in byh[h])
    print(f"total density budget: {tot:.4f}")
    # check budget for M-candidates
    for M in (5040, 55440, 720720, 110880, 166320):
        sub = [(p, h, a, b) for p, (o2, o3, h, g, a, b) in inv.items() if h % M == 0 or M % h == 0]
        bud = sum(1.0 / h for _, h, _, _ in sub)
        print(f"M={M:>7}: {len(sub):3d} primes, budget={bud:.4f}")
    print()
    print("h -> primes (h<=130):")
    for h in sorted(byh):
        if h <= 130:
            print(f"  h={h:>4}: " + "  ".join(f"{p}(a={a},b={b})" for p, a, b in byh[h]))
