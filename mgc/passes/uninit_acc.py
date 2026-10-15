# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""uninit-acc: accumulating jobs on an accumulator that holds nothing yet.

A job with an `if_uninit` alternative in its descriptor (RedMulE `gemm`,
`y += x @ w`, has `matmul`, `z = x @ w`) reads its `inout` operands. When no
write of such a buffer can reach the job, what it reads is not initialized,
and the alternative computes the same thing without reading it (no Y traffic,
no wait for Y on RedMulE). This pass marks those jobs.

Writes of a buffer: DMA loads into it, jobs and L1 kernels with it as an `out`
or `inout` operand (the job itself included), on any slot of the buffer. The
job reads its accumulator while it runs, from its start (which, for an
`enqueue`, comes later, at the commit) until its `.wait()`. A write reaches it
if it comes before that wait in the iteration order (textual order; stage,
then textual order in a pipeline body), e.g. a `memset_zero(y)` between the
`enqueue` and the `commit_start`, or if it is in a loop around the job (an
earlier iteration). A job whose wait is not found after it in its loops is
left as it is. So the job switches:

* always, if it is in no loop;
* in the first iteration of every loop around it, otherwise: the job gets
  `alt` (the alternative) and `alt_if` (`i == 0 && z == 0`), which the
  pipeline pass instantiates with the iteration of each step. job-lower then
  folds it with `fold()` (in the prologue of a pipeline `i == 0` holds, in its
  step loop `i + 1 == 0` does not), and the C emitter prints what is left as
  a choice between the two calls.

A buffer that another tile can write (`idma.store` into `buf.at_rel(...)` and
the like) is never switched: the other tile runs on its own time line.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr
from xdsl.passes import ModulePass

from .. import devices
from .. import ir
from ..expr import Cmp, Const, Logic, Sym, const_of, sub
from .common import alloc_of, kernels
from .pipeline import _roles


def _job(op):
    """(accelerator, job) of a job op, None for other ops and software kernels."""
    if isinstance(op, ir.HwpeOp) and op.action.data == 'enqueue':
        return op.acc.data, op.job.data
    if isinstance(op, ir.JobOp) and '.' in op.kernel.data:
        return tuple(op.kernel.data.split('.'))
    return None


def _writes(op, kdefs):
    """(local, remote): the allocations `op` writes on this tile and on other tiles."""
    if isinstance(op, ir.IdmaOp):
        if op.remote is not None:
            return [], [alloc_of(op.remote)]
        return ([alloc_of(op.local)] if op.dir.data == 0 else []), []
    if isinstance(op, (ir.JobOp, ir.HwpeOp)) and len(op.bufs):
        return [alloc_of(v) for v, r in zip(op.bufs, _roles(op, kdefs)) if r in ('out', 'inout')], []
    return [], []


def _path(op):
    """Position of `op` as (parent op, key) pairs from the outermost: `key`
    orders the children of one parent in iteration order."""
    out = []
    while True:
        parent = op.parent_op()
        if parent is None:
            break
        idx = list(op.parent_block().ops).index(op)
        if isinstance(parent, ir.PipelineOp) and isinstance(op, ir.StageOp):
            key = (op.k.data, idx)
        else:
            key = (parent.regions.index(op.parent_region()), idx)
        out.append((parent, key))
        op = parent
    return out[::-1]


def _precedes(a, b):
    """True if `a` runs before `b` in one iteration of their common loops."""
    for (pa, ka), (pb, kb) in zip(_path(a), _path(b)):
        assert pa is pb
        if ka != kb:
            return ka < kb
    return False


def _end(op, waits):
    """The first `.wait()` of the event of job `op` after it, None if there is none."""
    e = op.event.data if op.event is not None else None
    if e is None:
        return None
    idx = e.index.c() if e.index is not None else None
    after = [w for w in waits if w.event.data.key() == e.key()
             and (w.event.data.index.c() if w.event.data.index is not None else None) == idx
             and not _precedes(w, op)]
    first = [w for w in after if not any(_precedes(v, w) for v in after if v is not w)]
    return first[0] if first else None


def _first_iteration(op):
    """Condition that every loop around `op` is in its first iteration (None if none)."""
    conds = []
    for parent, _ in _path(op):
        if isinstance(parent, ir.ForOp):
            conds.append(Cmp('==', Sym(parent.var.data), parent.lo.data))
        elif isinstance(parent, ir.PipelineOp):
            conds.append(Cmp('==', Sym(parent.var.data), Const(0)))
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else Logic('&&', conds)


@dataclass(frozen=True)
class UninitAcc(ModulePass):
    """Marks the accumulating jobs whose accumulator is not initialized."""
    name = 'uninit-acc'

    def apply(self, ctx, module):
        kdefs = kernels(module)
        writers, remote = {}, set()
        waits = [op for op in module.walk() if isinstance(op, ir.WaitOp)]
        for op in module.walk():
            local, rem = _writes(op, kdefs)
            for a in local:
                writers.setdefault(a, []).append(op)
            remote.update(rem)

        for op in list(module.walk()):
            j = _job(op)
            if j is None or getattr(op, 'alt', None) is not None:
                continue  # `uninit=` given by the user
            acc, job = j
            alt = devices.HWPES[acc].jobs[job].if_uninit
            if not alt:
                continue
            end = _end(op, waits)
            if end is None:
                continue
            roles = _roles(op, kdefs)
            accs = [alloc_of(v) for v, r in zip(op.bufs, roles) if r == 'inout']
            if any(a in remote or any(_precedes(w, end) for w in writers.get(a, ()) if w is not op)
                   for a in accs):
                continue
            cond = _first_iteration(op)
            if cond is None:
                if isinstance(op, ir.HwpeOp):
                    op.attributes['job'] = StringAttr(alt)
                else:
                    op.attributes['kernel'] = StringAttr(f'{acc}.{alt}')
            else:
                op.attributes['alt'] = StringAttr(alt)
                op.attributes['alt_if'] = ir.E(cond)


def fold(op):
    """Fold the `alt_if` of `op` once its loops are final (after the pipeline
    pass): 1 / 0 if it always / never holds, else the condition left. A term
    `v + c == 0` is false when `v` is the variable of a loop around `op`
    starting at `lo` with `lo + c > 0` (the loops only count up)."""
    lows = {}
    p = op.parent_op()
    while p is not None:
        if isinstance(p, ir.ForOp) and const_of(p.lo.data) is not None:
            lows.setdefault(p.var.data, const_of(p.lo.data))
        p = p.parent_op()

    def value(c):
        v = c.value()
        if v is not None or not (isinstance(c, Cmp) and c.op == '=='):
            return v
        d = sub(c.a, c.b, fold=False)
        if len(d.syms()) == 1:
            s = next(iter(d.syms()))
            k = const_of(sub(d, Sym(s), fold=False))
            if k is not None and s in lows and lows[s] + k > 0:
                return 0
        return None

    cond = op.alt_if.data
    keep = []
    for c in (cond.xs if isinstance(cond, Logic) and cond.op == '&&' else [cond]):
        v = value(c)
        if v == 0:
            return 0
        if v is None:
            keep.append(c)
    if not keep:
        return 1
    return keep[0] if len(keep) == 1 else Logic('&&', keep)
