"""Non-parametric paired statistics for repeated build wall-clock.
Input: one or more manifests whose results.rows have
{block, system, build_s} -- produced by bench/paired_wallclock.py (laptop)
or by the server-side paired-build driver (manifests
results/paired_wallclock/graft-build-paired-*-d0-s1.json).

For every (A, B) pair requested (default: every GRAFT profile against every
baseline) it reports, over the blocks in which both were built:
  median and IQR of each system's wall-clock,
  the per-block ratio A/B: median, Hodges-Lehmann estimate (median of
  Walsh averages of log-ratios, exponentiated) with its 95% CI,
  Wilcoxon signed-rank on log-ratios (exact, two-sided) and the sign test,
  and the number of blocks in which A was slower than B.
Usage:
    python3 bench/paired_stats.py results/paired_wallclock/*.json

Requires numpy and scipy.
"""
import itertools
import json
import sys

import numpy as np
from scipy import stats


def walsh_hl(x, conf=0.95):
    """Hodges-Lehmann one-sample estimate and exact CI from Walsh averages."""
    x = np.asarray(x); n = len(x)
    w = np.sort([(x[i] + x[j]) / 2 for i in range(n) for j in range(i, n)])
    m = len(w)
    # exact CI: rank bounds from the signed-rank distribution
    k = _signed_rank_crit(n, conf)
    lo = w[k - 1] if k >= 1 else w[0]; hi = w[m - k] if k >= 1 else w[-1]
    return float(np.median(w)), float(lo), float(hi)


def _signed_rank_crit(n, conf):
    # smallest k such that P(W <= k-1) <= (1-conf)/2 under H0
    tot = n * (n + 1) // 2
    pmf = np.zeros(tot + 1); pmf[0] = 1.0
    for r in range(1, n + 1):
        new = pmf.copy(); new[r:] += pmf[:tot + 1 - r]; pmf = new / 2
    cdf = np.cumsum(pmf); alpha = (1 - conf) / 2
    k = int(np.searchsorted(cdf, alpha, side="right"))
    return k


def pairs_of(systems):
    g = [s for s in systems if s.startswith("graft")]
    b = [s for s in systems if not s.startswith("graft")]
    return [(a, c) for a in g for c in b] or list(itertools.combinations(systems, 2))


def analyse(man):
    rows = man["results"]["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["system"], {})[r["block"]] = r["build_s"]
    systems = list(by)
    out = []
    ds = man.get("dataset", "?")
    name = ds.get("name", "?") if isinstance(ds, dict) else ds
    print(f"\n== {name}  ({man['config'].get('blocks_done', '?')} blocks; "
          f"hash identical across blocks: {man['results'].get('hash_identical_across_blocks')})")
    print(f"{'system':26s} {'n':>3s} {'median s':>9s} {'IQR':>15s}")
    for s in systems:
        v = np.array(list(by[s].values()))
        q1, q3 = np.percentile(v, [25, 75])
        print(f"{s:26s} {len(v):3d} {np.median(v):9.1f} [{q1:6.1f}, {q3:6.1f}]")
    print(f"\n{'A vs B':45s} {'n':>3s} {'med A/B':>8s} {'HL A/B [95% CI]':>22s} {'Wilcoxon p':>11s} {'sign p':>8s} {'A>B':>5s}")
    for a, c in pairs_of(systems):
        blocks = sorted(set(by[a]) & set(by[c]))
        if len(blocks) < 2:
            continue
        ra = np.array([by[a][b] for b in blocks]); rc = np.array([by[c][b] for b in blocks])
        lr = np.log(ra / rc)
        hl, lo, hi = walsh_hl(lr)
        wp = stats.wilcoxon(lr, alternative="two-sided", method="exact").pvalue if np.any(lr != 0) else 1.0
        npos = int((lr > 0).sum()); nn = int((lr != 0).sum())
        sp = stats.binomtest(npos, nn, 0.5).pvalue if nn else 1.0
        print(f"{a + ' vs ' + c:45s} {len(blocks):3d} {np.exp(np.median(lr)):8.2f} "
              f"{np.exp(hl):7.2f} [{np.exp(lo):5.2f}, {np.exp(hi):5.2f}] {wp:11.4f} {sp:8.4f} {npos:2d}/{len(blocks)}")
        out.append({"A": a, "B": c, "n": len(blocks), "median_ratio": float(np.exp(np.median(lr))),
                    "hl_ratio": float(np.exp(hl)), "ci": [float(np.exp(lo)), float(np.exp(hi))],
                    "wilcoxon_p": float(wp), "sign_p": float(sp), "A_slower": npos})
    return out


if __name__ == "__main__":
    for f in sys.argv[1:]:
        analyse(json.load(open(f)))
