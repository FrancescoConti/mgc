#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""llama_ffn - extrapolate the Llama3 SwiGLU FFN GEMMs (S=512) from mm_os_mgc slice runs of mgc_sweep.

The FFN of a layer is gate and up, [S x E] @ [E x F], and down, [S x F] @ [F x E]. Each GEMM runs on
the mesh as `passes` output-static passes of K_sim = tiles * Kt output columns (Kt per tile, sized on
the L1 with T_SIZE=64, see PLAN); the cycles of one pass are a + b * N, fitted from the mm_os_mgc runs
with M=S, K=K_sim and N=256, 512. When the N=512 run is missing, b is taken from the N=256 run and
the mean steady-state efficiency of the other shapes of the same mesh side.

The attention block (real GQA, best S_BLOCK config, ms per layer at 1 GHz) is taken from the
mhsa_pipeopt3 sweeps, scaled to Hq > Hkv (see ATT).

    mgc/scripts/mgc_sweep.py --tests mm_os_mgc --tiles 8 --M 512 --N 256 512 \\
        --K 2048 3072 4096 8192 13312 14336 16384 --T_SIZE 64 --l2-size 0x10000000
    mgc/scripts/llama_ffn.py mgc_sweep/<run>/results.csv [...]
"""

import csv
import sys

S = 512
PEAK = 64  # RedMulE MAC/cycle per tile
MODELS = {  # E, F (d_ff), layers
    '1B': (2048, 8192, 16), '3B': (3072, 8192, 28), '8B': (4096, 14336, 32),
    '70B': (8192, 28672, 80), '405B': (16384, 53248, 126)}
# tiles -> {GEMM output columns K: (K_sim, passes)}
PLAN = {
    4: {8192: (4096, 2), 14336: (3584, 4), 2048: (2048, 1), 3072: (3072, 1), 4096: (4096, 1)},
    8: {8192: (8192, 1), 14336: (14336, 1), 28672: (14336, 2), 53248: (13312, 4),
        2048: (2048, 1), 3072: (3072, 1), 4096: (4096, 1), 16384: (16384, 1)}}
# attention block per layer, ms at 1 GHz: mhsa_pipeopt3, S=512, H=Hkv=8, scaled to Hq
# (mgc_sweep/20261017_183630 for tiles=4, mgc_sweep/20261017_192646 for tiles=8)
ATT = {4: {'1B': 7.52, '3B': 15.90, '8B': 26.27},
       8: {'1B': 2.83, '3B': 5.17, '8B': 8.09, '70B': 25.82, '405B': 94.38}}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs = {}  # (tiles, K, N) -> row
    for f in sys.argv[1:]:
        for r in csv.DictReader(open(f)):
            if r['test'] == 'mm_os_mgc' and r['cycles'] and int(r['M'] or 0) == S:
                runs[(int(r['tiles']), int(r['K']), int(r['N']))] = r

    print('slice runs:')
    for k in sorted(runs):
        r = runs[k]
        c = int(r['cycles'])
        print('  tiles=%d K=%5d N=%3d  %-5s %9d cyc  %7.1f MAC/cyc  RedMulE %s%%  iDMA %s%%'
              % (*k, r['status'], c, S * k[2] * k[1] / c, r['redmule_pct'], r['idma_pct']))

    # (tiles, K_sim) -> (a, b, steady-state efficiency)
    fit = {}
    for (tiles, ks, n), r in runs.items():
        r2 = runs.get((tiles, ks, 512))
        if n == 256 and r2:
            c1 = int(r['cycles'])
            b = (int(r2['cycles']) - c1) / 256
            fit[(tiles, ks)] = (c1 - 256 * b, b, S * ks / (tiles * tiles * PEAK) / b)
    print('\nper-pass fit, cycles = a + b * N:')
    for (tiles, ks), (a, b, e) in sorted(fit.items()):
        print('  tiles=%d K_sim=%5d  a=%7d  b=%7.1f  steady state %.2f%% of peak' % (tiles, ks, a, b, 100 * e))
    for (tiles, ks, n), r in runs.items():
        if n == 256 and (tiles, ks) not in fit:
            effs = [e for (t, _), (_, _, e) in fit.items() if t == tiles]
            if effs:
                e = sum(effs) / len(effs)
                b = S * ks / (tiles * tiles * PEAK) / e
                fit[(tiles, ks)] = (int(r['cycles']) - 256 * b, b, e)
                print('  tiles=%d K_sim=%5d  no N=512 run: b from the mean efficiency %.2f%% (estimated)'
                      % (tiles, ks, 100 * e))

    def gemm(tiles, n, k):
        ks, p = PLAN[tiles].get(k, (None, 0))
        if (tiles, ks) not in fit:
            return None
        a, b, _ = fit[(tiles, ks)]
        return p * (a + b * n)

    print('\nFFN per layer (gate + up + down) and attention, ms at 1 GHz:')
    print('%-5s %5s %12s %9s %10s %10s %9s %10s %10s'
          % ('model', 'tiles', 'FFN cycles', 'MAC/cyc', 'FFN ms', 'FFN all', 'att ms', 'layer ms', 'model ms'))
    for tiles in sorted(PLAN):
        for m, (e, f, layers) in MODELS.items():
            up, down = gemm(tiles, e, f), gemm(tiles, f, e)
            if up is None or down is None:
                continue
            cyc = 2 * up + down
            ffn = cyc / 1e6
            att = ATT.get(tiles, {}).get(m)
            print('%-5s %5d %12d %9.1f %10.2f %10.1f %9s %10s %10s'
                  % (m, tiles, cyc, 3 * S * e * f / cyc, ffn, ffn * layers,
                     '-' if att is None else '%.2f' % att, '-' if att is None else '%.2f' % (ffn + att),
                     '-' if att is None else '%.1f' % ((ffn + att) * layers)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
