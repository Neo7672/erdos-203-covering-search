"""LNS (Large Neighborhood Search) for the 2D covering problem, Erdos #203.

Move: pick a random subset R of primes (ruin), remove their cosets, then
re-insert ALL primes greedily in random order (recreate), each choosing its
best coset against the current uncovered set. Accept if uncovered decreases
(or equal, with probability, to drift). Track best solution.
"""
import numpy as np, json, sys, time, random
from math import gcd

def load_primes(M, f="inventory_full.json"):
    inv = json.load(open(f))
    out = []
    for p, (o2, o3, h, a, b) in inv.items():
        p, h, a, b = int(p), int(h), int(a), int(b)
        if h and M % h == 0:
            out.append(dict(p=p, h=h, a=a % h or h and a % h, b=b % h))
    out.sort(key=lambda d: d["h"])
    return out

class LNS:
    def __init__(self, primes, M, seed=0):
        self.primes = primes
        self.M = M
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.k = np.arange(M, dtype=np.int64)
        self.c = {pr["p"]: 0 for pr in primes}
        self.CNT = np.zeros((M, M), dtype=np.int8)
        for pr in primes:
            self.CNT += self.mask(pr, self.c[pr["p"]])

    def lf(self, pr):
        return ((pr["a"] * self.k[None, :] + pr["b"] * self.k[:, None]) % pr["h"]).astype(np.int16)

    def mask(self, pr, c):
        return (self.lf(pr) == c)

    def uncovered(self):
        return int((self.CNT == 0).sum())

    def best_c(self, pr):
        h = pr["h"]
        lf = self.lf(pr)
        cnts = np.bincount(lf[self.CNT == 0].astype(np.int64), minlength=h)
        return int(cnts.argmax()), int(cnts.max())

    def remove(self, subset):
        for pr in subset:
            self.CNT -= self.mask(pr, self.c[pr["p"]])

    def reinsert_greedy(self, order):
        for pr in order:
            c, gain = self.best_c(pr)
            self.CNT += self.mask(pr, c)
            self.c[pr["p"]] = c

    def lns_round(self, ruin=6):
        subset = self.pyrng.sample(self.primes, min(ruin, len(self.primes)))
        u0 = self.uncovered()
        self.remove(subset)
        rest = [pr for pr in self.primes if pr not in subset]
        order = rest + self.pyrng.sample(subset, len(subset))
        # re-greedily re-place EVERYTHING in random order (cheap improvement pass)
        for pr in order:
            c, _ = self.best_c(pr)
            if c != self.c[pr["p"]]:
                self.CNT -= self.mask(pr, self.c[pr["p"]])
                self.CNT += self.mask(pr, c)
                self.c[pr["p"]] = c
        u1 = self.uncovered()
        return u0, u1

    def run(self, minutes, ruin=(4, 9), verbose=True):
        t0 = time.time()
        best = (self.uncovered(), dict(self.c))
        rnd = 0
        while time.time() - t0 < minutes * 60:
            r = self.pyrng.randint(*ruin)
            u0, u1 = self.lns_round(r)
            rnd += 1
            if u1 < best[0]:
                best = (u1, dict(self.c))
                if verbose:
                    print(f"  round {rnd:>5} ruin={r} {u0}->{u1}  BEST={u1} "
                          f"({u1/(self.M*self.M):.4%}) t={time.time()-t0:.0f}s", flush=True)
            elif u1 > u0:
                # revert: restore best cosets (cheap: re-place all from best)
                for pr in self.primes:
                    bc = best[1][pr["p"]]
                    if self.c[pr["p"]] != bc:
                        self.CNT -= self.mask(pr, self.c[pr["p"]])
                        self.CNT += self.mask(pr, bc)
                        self.c[pr["p"]] = bc
            if best[0] == 0:
                break
        # ensure final state = best
        for pr in self.primes:
            bc = best[1][pr["p"]]
            if self.c[pr["p"]] != bc:
                self.CNT -= self.mask(pr, self.c[pr["p"]])
                self.CNT += self.mask(pr, bc)
                self.c[pr["p"]] = bc
        return best[0]

if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 5040
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 10
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    primes = load_primes(M)
    bud = sum(1 / pr["h"] for pr in primes)
    print(f"M={M} primes={len(primes)} budget={bud:.4f}")
    s = LNS(primes, M, seed)
    u = s.uncovered()
    print(f"start (all c=0): uncovered={u} ({u/M/M:.4%})")
    u = s.run(minutes)
    print(f"final uncovered: {u}")
    if u == 0:
        json.dump({str(pr["p"]): s.c[pr["p"]] for pr in primes},
                  open(f"covering_M{M}_lns{seed}.json", "w"))
        print(f"*** COVERING FOUND -> covering_M{M}_lns{seed}.json ***")
