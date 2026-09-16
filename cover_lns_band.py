"""Banded LNS engine for M=55440 (8GB RAM budget).

CNT (int8) full torus resident: 55440^2 = 3.07GB.
lf/masks computed per band (2000 rows) on the fly; never materialized fully.
"""
import numpy as np, json, sys, time, random

BAND = 2000

def load_primes(M, f="inventory_full.json"):
    inv = json.load(open(f))
    out = []
    for p, (o2, o3, h, a, b) in inv.items():
        p, h, a, b = int(p), int(h), int(a), int(b)
        if h and M % h == 0:
            out.append(dict(p=p, h=h, a=a % h, b=b % b0(h) if False else b % h))
    out.sort(key=lambda d: d["h"])
    return out

def b0(h): return h

class BandedLNS:
    def __init__(self, primes, M, seed=0):
        self.primes = primes; self.M = M
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.c = {pr["p"]: 0 for pr in primes}
        self.CNT = np.zeros((M, M), dtype=np.int8)
        for pr in primes:
            self._apply_mask(pr, self._mask(pr, self.c[pr["p"]]), +1)

    def _bands(self):
        M, B = self.M, BAND
        for l0 in range(0, M, B):
            l1 = min(l0 + B, M)
            yield l0, l1

    def _lf_band(self, pr, l0, l1):
        h, a, b = pr["h"], pr["a"], pr["b"]
        k = np.arange(self.M, dtype=np.int64)
        l = np.arange(l0, l1, dtype=np.int64)
        return ((a * k[None, :] + b * l[:, None]) % h).astype(np.int16)

    def _mask(self, pr, c):
        """generator of band masks (bool arrays) for coset value c"""
        for l0, l1 in self._bands():
            yield self._lf_band(pr, l0, l1) == c

    def _apply_mask(self, pr, maskgen, sign):
        i = 0
        for l0, l1 in self._bands():
            m = next(maskgen)
            self.CNT[l0:l1] += sign * m
            i += 1

    def uncovered(self):
        return int((self.CNT == 0).sum())

    def set_coset(self, pr, c_new):
        c_old = self.c[pr["p"]]
        if c_old == c_new: return
        for l0, l1 in self._bands():
            lf = self._lf_band(pr, l0, l1)
            self.CNT[l0:l1] += ((lf == c_new).astype(np.int8)
                                - (lf == c_old).astype(np.int8))
        self.c[pr["p"]] = c_new

    def best_c(self, pr):
        h = pr["h"]
        cnts = np.zeros(h, dtype=np.int64)
        for l0, l1 in self._bands():
            lf = self._lf_band(pr, l0, l1)
            unc = self.CNT[l0:l1] == 0
            if unc.any():
                cnts += np.bincount(lf[unc].astype(np.int64), minlength=h)
        return int(cnts.argmax()), int(cnts.max())

    def lns_round(self, ruin):
        subset = self.pyrng.sample(self.primes, min(ruin, len(self.primes)))
        order = self.pyrng.sample(self.primes, len(self.primes))
        for pr in order:
            c, _ = self.best_c(pr)
            self.set_coset(pr, c)

    def run(self, minutes, ruin=(4, 9), verbose=True):
        t0 = time.time()
        best = (self.uncovered(), dict(self.c))
        rnd = 0
        while time.time() - t0 < minutes * 60:
            rnd += 1
            self.lns_round(self.pyrng.randint(*ruin))
            u = self.uncovered()
            if u < best[0]:
                best = (u, dict(self.c))
                if verbose:
                    print(f"  round {rnd:>4} BEST={u} ({u/(self.M*self.M):.4%}) "
                          f"t={time.time()-t0:.0f}s", flush=True)
            else:
                for pr in self.primes:
                    self.set_coset(pr, best[1][pr["p"]])
            if best[0] == 0: break
        for pr in self.primes:
            self.set_coset(pr, best[1][pr["p"]])
        return best[0]

if __name__ == "__main__":
    M = int(sys.argv[1]); minutes = float(sys.argv[2]); seed = int(sys.argv[3])
    primes = load_primes(M)
    bud = sum(1 / pr["h"] for pr in primes)
    print(f"M={M} primes={len(primes)} budget={bud:.4f}", flush=True)
    s = BandedLNS(primes, M, seed)
    u = s.uncovered()
    print(f"start: uncovered={u} ({u/M/M:.4%})", flush=True)
    u = s.run(minutes)
    print(f"final uncovered: {u}")
    if u == 0:
        json.dump({str(pr["p"]): s.c[pr["p"]] for pr in primes},
                  open(f"covering_M{M}_lns{seed}.json", "w"))
        print("*** COVERING FOUND ***")
