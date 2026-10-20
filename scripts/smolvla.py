#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""smolvla - SmolVLA (lerobot/smolvla_base) inference on MAGIA, from mgc_sweep runs of the layer shapes.

Model (lerobot SmolVLAConfig defaults, SmolVLM2-500M-Video-Instruct):
  vision: SigLIP-B/16 at 512x512 -> 1024 patches, E=768, 12 heads x 64, MLP 3072, 12 layers, per camera
  connector: pixel shuffle x4 -> 64 tokens x 12288 -> Linear 12288 -> 960
  VLM text: first 16 of 32 SmolLM2-360M layers, E=960, 15 q / 5 kv heads x 64, MLP 2560 (SwiGLU)
  prefix: 64 tokens per camera + <=48 language + 1 state, run once (KV cached)
  action expert: 16 layers, E=720, 15 q / 5 kv heads x 64, MLP 2048 (SwiGLU); 50 action tokens,
    10 flow-matching steps; odd layers cross-attend to the VLM prefix K/V (re-projected 320 -> 320),
    even layers self-attend over prefix + suffix

Simulated: VLM prefix padded to 256 tokens (3 cameras: 241, fewer cameras are padded to 256 too),
expert tokens padded to 64, expert width 720 padded to 768 where it is the reduction. VLM attention
from gqa_pipeopt3_masked (H=15, HKV=5, the prefix mask), or else gqa_pipeopt3 (no mask load), or else
mhsa_pipeopt3 with 15 kv heads scaled by the GQA / MHA MAC ratio. Expert attention core (50 queries over
the prefix) costed at the MAC/cycle of the VLM S=256 attention block. Cycles at 1 GHz.

Non-matmul ops (LayerNorm, GELU and residual adds in vision; RMSNorm, RoPE on Q/K, SwiGLU and residual
adds in the VLM and the expert) are counted only from --ops files, CSV with columns tiles,op,rows,cols,
cycles (op: layernorm gelu add rmsnorm rope swiglu; rope cols = heads x head_dim; cycles = slowest
tile). A shape not measured is costed linearly in rows x cols from the measured ones (a + b n with two
or more sizes, proportional with one). Ops with no data are listed and not counted. Pixel shuffle is
not counted. See smolvla_pending.txt for the runs.

    tests/magia/mgc/scripts/mgc_sweep.py --tests mm_os_mgc --tiles <N> --M 64 --N 768 --K 320 960 2048 ...
        (M x N x K: 1024 x 768 x 768/3072, 1024 x 3072 x 768, 64 x 12288 x 960, 256 x 960 x 2560,
         256 x 2560 x 960, 256 x 320 x 320, 64 x 768 x 320/960/2048, 64 x 960 x 720, 64 x 2048 x 720;
         T_SIZE=64, --no-profile --l2-size 0x10000000)
    tests/magia/mgc/scripts/mgc_sweep.py --tests mhsa_pipeopt3 --tiles <N> --S 1024 --D 768 --P 64 --H 12 \
        --S_BLOCK_SIZE_Q 512 --S_BLOCK_SIZE_KV 256 --no-profile --l2-size 0x10000000
    tests/magia/mgc/scripts/mgc_sweep.py --tests gqa_pipeopt3[_masked] --tiles <N> --S 256 --D 960 --P 64 --H 15 \
        --HKV 5 --S_BLOCK_SIZE_Q 256 --S_BLOCK_SIZE_KV 256 --no-profile --l2-size 0x10000000
    tests/magia/mgc/scripts/smolvla.py mgc_sweep/<run>/results.csv [...] [--ops ops.csv [...]]
"""
import argparse
import csv
import sys

if len(sys.argv) < 2:
    sys.exit(__doc__)
ap = argparse.ArgumentParser(usage=__doc__)
ap.add_argument('results', nargs='+', help='mgc_sweep results.csv')
ap.add_argument('--ops', nargs='+', default=[], help='non-matmul op cycles, CSV tiles,op,rows,cols,cycles')
args = ap.parse_args()

runs = {}
for f in args.results:
    for r in csv.DictReader(open(f)):
        if not r['cycles']:
            continue
        t, c = int(r['tiles']), int(r['cycles'])
        if r['test'] == 'mm_os_mgc':
            key = (t, 'mm', int(r['M']), int(r['N']), int(r['K']))
        else:
            kind = 'gqa' if r['test'].startswith('gqa') else 'att'
            key = (t, kind + ('_masked' if r['test'].endswith('_masked') else ''), int(r['S']), int(r['D']))
        runs[key] = min(c, runs.get(key, c))

ops = {}  # (tiles, op) -> {rows x cols: cycles}
for f in args.ops:
    for r in csv.DictReader(open(f)):
        d = ops.setdefault((int(r['tiles']), r['op']), {})
        n, c = int(r['rows']) * int(r['cols']), int(r['cycles'])
        d[n] = min(c, d.get(n, c))


def op_cost(t, op, rows, cols, missing):
    d = ops.get((t, op))
    if not d:
        missing.add(op)
        return 0
    n = rows * cols
    if n in d:
        return d[n]
    if len(d) == 1:
        (n0, c0), = d.items()
        return c0 * n / n0
    xs, ys = list(d), list(d.values())
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my + b * (n - mx)


L_TXT = 48
VIS_MAC = 12 * (4 * 1024 * 768 * 768 + 2 * 1024 * 1024 * 768 + 2 * 1024 * 768 * 3072) \
    + 1024 * 768 * 768 + 64 * 12288 * 960
VLM_ATT_SIM = 4 * 256 * 960 * 960 + 2 * 256 * 256 * 960  # MHA, H=15
VLM_ATT_GQA = 256 * (2 * 960 * 960 + 2 * 960 * 320) + 2 * 256 * 256 * 960


def macs(cams):
    p = 64 * cams + L_TXT + 1
    vlm = 16 * (p * (2 * 960 * 960 + 2 * 960 * 320 + 3 * 960 * 2560) + 2 * p * p * 960)
    core = 8 * 2 * 50 * p * 960 + 8 * 2 * 50 * (p + 50) * 960
    exp = 10 * (16 * 50 * (2 * 720 * 960 + 2 * 720 * 320 + 3 * 720 * 2048) + 8 * 2 * p * 320 * 320 + core)
    return cams * VIS_MAC, vlm, exp, core


def model(t, missing):
    mm = lambda m, n, k: runs[(t, 'mm', m, n, k)]
    op = lambda name, rows, cols: op_cost(t, name, rows, cols, missing)
    b = {}
    b['vision att / layer'] = runs[(t, 'att', 1024, 768)]
    b['vision MLP / layer'] = mm(1024, 768, 3072) + mm(1024, 3072, 768)
    b['patch embed'] = mm(1024, 768, 768)
    b['connector'] = mm(64, 12288, 960)
    # 2 LayerNorm, GELU, 2 residual adds per layer; post-LayerNorm once
    b['vision ops / layer'] = 2 * op('layernorm', 1024, 768) + op('gelu', 1024, 3072) + 2 * op('add', 1024, 768)
    b['vision / camera'] = 12 * (b['vision att / layer'] + b['vision MLP / layer'] + b['vision ops / layer']) \
        + op('layernorm', 1024, 768) + b['patch embed'] + b['connector']
    if (t, 'gqa_masked', 256, 960) in runs:  # gqa_pipeopt3_masked, H=15, HKV=5
        b['VLM att / layer (GQA, masked)'] = runs[(t, 'gqa_masked', 256, 960)]
    elif (t, 'gqa', 256, 960) in runs:  # gqa_pipeopt3, no mask load
        b['VLM att / layer (GQA, no mask)'] = runs[(t, 'gqa', 256, 960)]
    else:  # mhsa_pipeopt3 with 15 kv heads, scaled to the GQA MACs
        b['VLM att / layer (GQA est., no mask)'] = runs[(t, 'att', 256, 960)] * VLM_ATT_GQA / VLM_ATT_SIM
    vlm_att = [v for k, v in b.items() if k.startswith('VLM att')][0]
    b['VLM MLP / layer'] = 2 * mm(256, 960, 2560) + mm(256, 2560, 960)
    # 2 RMSNorm, RoPE on Q (15 heads) and K (5 heads), SwiGLU, 2 residual adds per layer
    b['VLM ops / layer'] = 2 * op('rmsnorm', 256, 960) + op('rope', 256, 960) + op('rope', 256, 320) \
        + op('swiglu', 256, 2560) + 2 * op('add', 256, 960)
    b['VLM prefix, 16 layers'] = 16 * (vlm_att + b['VLM MLP / layer'] + b['VLM ops / layer'])
    b['expert QKVO / layer'] = mm(64, 768, 960) + 2 * mm(64, 768, 320) + mm(64, 960, 720)
    b['expert MLP / layer'] = 2 * mm(64, 768, 2048) + mm(64, 2048, 720)
    b['cross K/V / cross layer'] = 2 * mm(256, 320, 320)
    b['expert att core / step (est.)'] = macs(3)[3] / (VLM_ATT_GQA / vlm_att)
    # as the VLM (RoPE on Q and K in all 16 layers), at 64 tokens x 720; final RMSNorm once per step
    b['expert ops / layer'] = 2 * op('rmsnorm', 64, 720) + op('rope', 64, 960) + op('rope', 64, 320) \
        + op('swiglu', 64, 2048) + 2 * op('add', 64, 720)
    b['expert / flow step'] = 16 * (b['expert QKVO / layer'] + b['expert MLP / layer'] + b['expert ops / layer']) \
        + 8 * b['cross K/V / cross layer'] + b['expert att core / step (est.)'] + op('rmsnorm', 64, 720)
    b['expert, 10 steps'] = 10 * b['expert / flow step']
    return b


for t in sorted({k[0] for k in runs}):
    missing = set()
    try:
        b = model(t, missing)
    except KeyError as e:
        print('tiles=%d: missing run %s' % (t, e))
        continue
    print('tiles=%d (%d tiles, peak %d MAC/cycle), cycles and ms at 1 GHz:' % (t, t * t, 64 * t * t))
    for name, c in b.items():
        print('  %-36s %12d  %9.3f' % (name, c, c / 1e6))
    if missing:
        print('  not counted (no --ops data): %s' % ' '.join(sorted(missing)))
    print('  %4s %10s %10s %10s %10s %8s %9s %7s' % ('cams', 'vision', 'VLM', 'expert', 'total ms', 'GMAC',
                                                   'MAC/cyc', '%peak'))
    for cams in (1, 2, 3):
        vm, lm, em, _ = macs(cams)
        cyc = cams * b['vision / camera'] + b['VLM prefix, 16 layers'] + b['expert, 10 steps']
        mac = vm + lm + em
        print('  %4d %10.2f %10.2f %10.2f %10.2f %8.1f %9.1f %7.1f'
              % (cams, cams * b['vision / camera'] / 1e6, b['VLM prefix, 16 layers'] / 1e6,
                 b['expert, 10 steps'] / 1e6, cyc / 1e6, mac / 1e9, mac / cyc, 100 * mac / cyc / (64 * t * t)))
    print()
