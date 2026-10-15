# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""pin-check: buffers pinned to loops, so that l1-layout can share L1 between loops.

`L1.single_buffer(view, pin="qk")` pins a buffer to the loop `pipeline(n, name="qk")`
(also `systolic()`); `pin=("a", "b")` to several loops, which must be back-to-back.
Named loops sit directly in the `tiles()` body, where they split it into stretches:
the stretch of loop k runs from the end of loop k - 1 to the start of loop k + 1, so
the statements between two loops belong to both. A buffer pinned to loops lo..hi:

* is used only in the stretches of lo..hi. A transfer or job uses its buffers until
  its `.wait()`;
* in the statements between loops k and k + 1, buffers pinned up to k are used
  before buffers pinned from k + 1 on;
* is not accessed by other tiles (`.at()`, `.at_rel()`, `.row`, `.col`): a slower
  tile could still write into memory this tile already reuses.

Every tile only reuses memory it alone touches, in program order, so the layout is
safe without any cross-tile timing. The pass records `pin_lo`..`pin_hi` (indices of
the named loops) on each pinned `mg.alloc`.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import IntAttr
from xdsl.passes import ModulePass

from .. import ir
from ..errors import MgcError
from .common import alloc_of


def _issues(op):
    """True if `op` completes on an event that a `.wait()` consumes."""
    if isinstance(op, ir.IdmaOp):
        return True
    if isinstance(op, ir.JobOp):
        return op.event is not None
    return isinstance(op, ir.HwpeOp) and op.action.data in ('enqueue', 'oneshot') and op.event is not None


def _evkey(e):
    return e.key() + (e.index.c() if e.index is not None else None,)


@dataclass(frozen=True)
class PinCheck(ModulePass):
    """Checks the `pin=` buffers against the stretches of their named loops."""
    name = 'pin-check'

    def apply(self, ctx, module):
        allocs = [o for o in module.walk() if isinstance(o, ir.AllocOp)]
        pinned = {id(a): a for a in allocs if a.pin is not None}
        loops = [o for o in module.walk() if isinstance(o, ir.PipelineOp) and o.label is not None]
        if not pinned and not loops:
            return
        tiles = next(o for o in module.walk() if isinstance(o, ir.TilesOp))
        body = tiles.body.block
        top = list(body.ops)
        pos = {id(o): k for k, o in enumerate(top)}

        names = {}
        for lp in loops:
            if lp.parent_block() is not body:
                raise MgcError(lp, f'named {lp.kind.data}() loops go directly in the tiles() body')
            if lp.label.data in names:
                raise MgcError(lp, f'two loops are named "{lp.label.data}"')
            names[lp.label.data] = len(names)
        at = [pos[id(lp)] for lp in loops]  # position of each named loop in the tiles() body

        for a in pinned.values():
            ks = []
            for x in a.pin.data:
                if x.data not in names:
                    raise MgcError(a, f'`{a.sym.data}` is pinned to "{x.data}", but no pipeline() or systolic() '
                                      f'has name="{x.data}"')
                ks.append(names[x.data])
            ks = sorted(set(ks))
            if ks != list(range(ks[0], ks[-1] + 1)):
                order = ', '.join(f'"{lp.label.data}"' for lp in loops)
                raise MgcError(a, f'`{a.sym.data}` is pinned to loops that are not back-to-back (in order: '
                                  f'{order}): pin it to the loops in between too')
            a.attributes['pin_lo'] = IntAttr(ks[0])
            a.attributes['pin_hi'] = IntAttr(ks[-1])

        def where(op):
            while op.parent_block() is not body:
                op = op.parent_op()
            return pos[id(op)]

        # top-level positions where each pinned buffer is used (transfers and jobs until their wait)
        ops = list(module.walk())
        uses = {k: [] for k in pinned}
        for n, op in enumerate(ops):
            if isinstance(op, ir.RemoteOp) and id(alloc_of(op.res)) in pinned:
                raise MgcError(op, f'`{alloc_of(op.res).sym.data}` is pinned to a loop and cannot be accessed by '
                                   'other tiles')
            if isinstance(op, ir.IdmaOp):
                bufs = [op.local]
            elif isinstance(op, (ir.HwpeOp, ir.JobOp, ir.CallOp)):
                bufs = list(op.bufs)
            else:
                continue
            mine = [alloc_of(v) for v in bufs if id(alloc_of(v)) in pinned]
            if not mine:
                continue
            ps = [where(op)]
            if _issues(op):
                key = _evkey(op.event.data)
                w = next((x for x in ops[n + 1:] if isinstance(x, ir.WaitOp) and _evkey(x.event.data) == key), None)
                ps.append(where(w) if w is not None else len(top))
            for a in mine:
                uses[id(a)] += [(p, op) for p in ps]

        for k, a in pinned.items():
            lo, hi = a.pin_lo.data, a.pin_hi.data
            start = at[lo - 1] + 1 if lo > 0 else 0
            end = at[hi + 1] - 1 if hi + 1 < len(at) else len(top) - 1
            span = '", "'.join(lp.label.data for lp in loops[lo:hi + 1])
            for p, op in uses[k]:
                if not start <= p <= end:
                    raise MgcError(op, f'`{a.sym.data}` is pinned to "{span}" but used (or waited) outside '
                                       'the stretch of the tiles() body around that loop')

        # statements between loops k and k + 1: buffers pinned up to k first
        for k in range(len(at) - 1):
            before = [(p, op) for i, a in pinned.items() if a.pin_hi.data == k for p, op in uses[i]]
            after = [(p, op) for i, a in pinned.items() if a.pin_lo.data == k + 1 for p, op in uses[i]]
            if before and after:
                last, first = max(before, key=lambda u: u[0]), min(after, key=lambda u: u[0])
                if last[0] >= first[0]:
                    raise MgcError(first[1], f'between loops "{loops[k].label.data}" and "{loops[k + 1].label.data}", '
                                             f'a buffer pinned up to "{loops[k].label.data}" is still used here or '
                                             f'later: it may share L1 with the buffers of '
                                             f'"{loops[k + 1].label.data}"')
