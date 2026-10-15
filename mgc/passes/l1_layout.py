# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""l1-layout: one L1 layout, identical on every tile.

Buffers and their slots are bump-allocated from `l1_tile_base` in declaration
order, except that buffers pinned to loops (`pin=`, checked by pin-check) share
L1 with the buffers of the other loops: each buffer goes right above the earlier
buffers whose loops overlap its own (an unpinned buffer overlaps every loop, so
without pins this is the plain bump allocation). Sizes are only known at run
time, so when several of those could be the highest, the start address is their
largest end, computed at run time. When a buffer is accessed on another tile (`buf.at_rel(dy, dx)`,
`buf.at(y, x)`, `buf.row`, `buf.col`), the other tile's copy must be at the
same offset, so slot sizes use the *maximum* tile sizes (`tile_h_max`): with
actual sizes a clipped edge tile would lay its buffers out differently and
receive data at the wrong address.

`buf.at_rel(dy, dx)` resolves to the neighbour's L1 base plus the local offset;
the base is computed once per direction (`l1_base_s` = tile below, ...).
Absolute and multicast targets compute the base where they are used.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.passes import ModulePass

from .. import ir
from ..errors import MgcError
from ..expr import Const, Raw, Sym, product
from .common import esize, max_sizes, tensors

DIRS = {(1, 0): 's', (-1, 0): 'n', (0, 1): 'e', (0, -1): 'w'}


def _overlap(a, b):
    """True if two loop intervals (`None`: every loop) overlap."""
    return a is None or b is None or (a[0] <= b[1] and b[0] <= a[1])


def _max(ends):
    """C expression for the largest of the end addresses `ends`."""
    out = f'({ends[0]})'
    for e in ends[1:]:
        out = f'(({out}) > ({e}) ? ({out}) : ({e}))'
    return out


@dataclass(frozen=True)
class L1Layout(ModulePass):
    """Assigns an L1 address to every buffer slot and resolves neighbour bases.

    Example: buffers `w` (1 slot, 4 KiB) and `x` (2 slots, 2 KiB each) get
    `l1_tile_base`, `l1_addr_w + 4096`, `l1_addr_x_0 + 2048`. Sets `addrs` on
    each `mg.alloc` and `base` (e.g. `l1_base_s`, the tile below) on each `mg.remote`.
    """
    name = 'l1-layout'

    def apply(self, ctx, module):
        ts = tensors(module)
        remotes = [o for o in module.walk() if isinstance(o, ir.RemoteOp)]
        uniform = bool(remotes)
        placed = []  # (loops, end address, ids of the buffers known to lie below its start)
        for a in [o for o in module.walk() if isinstance(o, ir.AllocOp)]:
            shp = a.view.data.shape
            if uniform:
                shp = [max_sizes(s) for s in shp]
            nbytes = product(*shp, Const(esize(a, ts)))
            depth = a.depth.data
            loops = (a.pin_lo.data, a.pin_hi.data) if a.pin_lo is not None else None
            under = [b for b in placed if _overlap(loops, b[0])]
            below = set().union(*(b[2] for b in under)) if under else set()
            tops = [b for b in under if id(b) not in below]
            if not tops:
                start = '(l1_tile_base)'
            elif len(tops) == 1:
                start = tops[0][1]
            else:
                start = _max([b[1] for b in tops])
            addrs = []
            for k in range(depth):
                slot = f'l1_addr_{a.sym.data}' if depth == 1 else f'l1_addr_{a.sym.data}_{k}'
                addrs.append(Raw(start))
                start = f'{slot} + ({nbytes.c()})'
            a.attributes['addrs'] = ir.exprs(addrs)
            entry = (loops, start, below | {id(b) for b in under})
            placed.append(entry)
        for r in remotes:
            if r.mode.data != 'rel':
                continue
            dy, dx = r.dy.data.value(), r.dx.data.value()
            if dy is None or dx is None:
                raise MgcError(r, 'neighbour offsets must be compile-time constants')
            suffix = DIRS.get((dy, dx), f'{dy}_{dx}'.replace('-', 'm'))
            r.attributes['base'] = StringAttr(f'l1_base_{suffix}')
