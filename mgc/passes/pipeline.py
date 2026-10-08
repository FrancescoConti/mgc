# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""pipeline: `mg.pipeline` -> explicit, predicated software pipeline.

The body is written for one iteration `i`, as `with stage(k):` blocks. Stage
`k` of iteration `i` runs `k` steps after its stage 0. The pass generates a
prologue (the stage-0 blocks of iteration 0) and a step loop: in step `s`,
every block runs in textual order on its own iteration `s + 1 - k`, guarded
by `0 <= s + 1 - k < n`. A stage may have several blocks, e.g. a stage-0 block
at the end of the step waits the DMAs issued by the first one.

Checks:
* depth: a storage indexed `[i + c]` (multi-buffer slot or event-array entry)
  touched in stages a..b holds b - a + 1 iterations at once, so it needs
  depth >= b - a + 1. A use of `[i + c]` in stage k counts as stage k - c
  (it is iteration i + c's slot). A transfer or job keeps its operands (and its event)
  until the stage of its wait. An omitted depth is inferred.
* a single buffer written in the loop is touched in one stage only (else the
  next iteration overwrites it): use a multi-buffer indexed `[i]`. Operands of
  in-order HWPE jobs count at their enqueue only.
* every event issued in the body is waited later in the same iteration. An
  event waited in a later stage than the one issuing it becomes an event
  array indexed `[i]` (DMA events too: `mg_idma_wait` waits by transfer id).
* scalars declared in a block are only used in that block.
* HWPE protocol: per job enqueue -> commit [-> start] -> wait; one simulated
  run checks that the in-order queue commits/starts the intended jobs and
  never holds more than `queue_depth` of them.
* `sync()` in a `pipeline()` stage needs a trip count that is the same on all
  tiles; `systolic()` stages do not take `sync()`.

`systolic` (with `skew`) wraps the steps in a global time loop, identical on
all tiles, in which the tile is active when 0 <= t - skew <= last step, with a
barrier ending every time step. Its length is max(skew + n) + S - 2 over the
mesh: tile-dependent scalars are inlined, and skew + n must be affine in the
tile ids, so the maximum is at a corner of the mesh.

Code shape matters to GCC (it changes register allocation and the cycle count),
so a step with a single guard takes the shape of the hand-written 2-stage
tests, `if (next) {...; wait} else {wait}`; otherwise consecutive ops with the
same guard share one `if`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter

from .. import devices
from .. import ir
from ..errors import MgcError
from ..expr import Cmp, Const, Expr, Logic, Sym, add, const_of, sub
from .common import alloc_of, kernels

# ----------------------------------------------------------------------------
# small IR constructors
# ----------------------------------------------------------------------------


def wait(ev: ir.EvtRef):
    """New `mg.wait` op on event `ev`."""
    return ir.WaitOp.create(attributes={'event': ir.EvtAttr(ev)})


def if_(cond, then, els=()):
    """New `mg.if` op with the given `then` / optional `else` op lists."""
    return ir.IfOp.create(attributes={'cond': ir.E(cond)},
                          regions=[_region(then), _region(els)])


def _region(ops):
    from xdsl.ir import Block, Region
    return Region(Block(list(ops)))


# ----------------------------------------------------------------------------
# substitution of the iteration variable
# ----------------------------------------------------------------------------


def _subst_attr(a, m):
    """Apply symbol substitution `m` inside an attribute (expressions, views,
    event indices, arrays of those)."""
    if isinstance(a, ir.ExprAttr):
        return ir.ExprAttr(a.data.subst(m))
    if isinstance(a, ir.ViewAttr):
        v = a.data
        return ir.ViewAttr(ir.View(v.tensor, [s.subst(m) for s in v.starts], [s.subst(m) for s in v.sizes],
                                   list(v.kept)))
    if isinstance(a, ir.EvtAttr):
        e = a.data
        return ir.EvtAttr(ir.EvtRef(e.kind, e.name, e.dir, e.index.subst(m) if e.index is not None else None))
    if isinstance(a, ArrayAttr):
        return ArrayAttr([_subst_attr(x, m) for x in a.data])
    return a


def instantiate(op, var, value: Expr):
    """Clone `op` for iteration `var = value`.

    E.g. cloning `idma.load(x[pt], X[pt*8:(pt+1)*8, :])` with `value = step + 1`
    yields the load for the next iteration: `x[step + 1]`, rows `(step+1)*8 ...`.
    """
    c = op.clone()
    m = {var: value}
    for o in c.walk():
        for k, a in list(o.attributes.items()):
            o.attributes[k] = _subst_attr(a, m)
        o.lineno = getattr(op, 'lineno', None)
    return c


def _syms(op):
    """Names used by the expressions of `op` and of the ops nested in it."""
    out = set()

    def visit(a):
        if isinstance(a, ir.ExprAttr):
            out.update(a.data.syms())
        elif isinstance(a, ir.ViewAttr):
            for e in a.data.starts + a.data.sizes:
                out.update(e.syms())
        elif isinstance(a, ir.EvtAttr) and a.data.index is not None:
            out.update(a.data.index.syms())
        elif isinstance(a, ArrayAttr):
            for x in a.data:
                visit(x)

    for o in op.walk():
        for a in o.attributes.values():
            visit(a)
    return out


# ----------------------------------------------------------------------------
# body analysis
# ----------------------------------------------------------------------------


@dataclass(eq=False)
class Block:
    """A `with stage(k):` block; `pos` is its textual position in the body."""
    k: int
    pos: int
    op: ir.StageOp

    @property
    def ops(self):
        return list(self.op.body.block.ops)


@dataclass(eq=False)
class Use:
    """One op of the body, in iteration order: (stage, block, position)."""
    op: object
    block: Block
    order: tuple

    @property
    def k(self):
        return self.block.k


@dataclass(eq=False)
class Issue:
    """An op completing on an event (DMA, accelerator job), with the wait(s)
    that end it; `last` is the stage of the (latest) wait."""
    use: Use
    key: tuple
    waits: List[Use] = field(default_factory=list)

    @property
    def last(self):
        return max([w.k for w in self.waits], default=self.use.k)


def _slot(idx: Expr, var: str):
    """Slot index -> ('rel', c) for var + c, ('abs', text) otherwise."""
    if var in idx.syms():
        c = const_of(sub(idx, Sym(var), fold=False))
        if c is None:
            raise MgcError(None, f'slot index `{idx.c()}` must be `{var} + constant`')
        return ('rel', c)
    return ('abs', idx.c())


def _roles(job, kdefs):
    """Operand roles (`in`/`out`/`inout`) of a job, from the accelerator
    descriptor or the L1-kernel declaration."""
    if isinstance(job, ir.HwpeOp):
        return [r for _, r, _ in devices.HWPES[job.acc.data].jobs[job.job.data].operands]
    k = job.kernel.data
    if '.' in k:
        acc, j = k.split('.')
        return [r for _, r, _ in devices.HWPES[acc].jobs[j].operands]
    return [o.data.partition(':')[2] for o in kdefs[k].operands_.data]


def _evkey(e: ir.EvtRef, var):
    """Identity of the storage an event names, within one iteration."""
    if e.kind == 'array':
        return ('array', e.name, None, _slot(e.index, var))
    return (e.kind, e.name, e.dir, None)


def _issues(op):
    """True if `op` completes on an event that a `.wait()` must consume."""
    if isinstance(op, ir.IdmaOp):
        return True
    if isinstance(op, ir.JobOp):
        return op.event is not None
    return isinstance(op, ir.HwpeOp) and op.action.data in ('enqueue', 'oneshot')


def _buffers(op, var, kdefs, remote):
    """(alloc, slot, written) for each L1 buffer `op` names. `remote`: also
    the destination of a store to other tiles (same step on every tile)."""
    out = []
    if isinstance(op, ir.IdmaOp):
        slot = _slot(op.slot.data, var)
        out.append((alloc_of(op.local), slot, op.dir.data == 0))
        if remote and op.remote is not None:
            r = op.remote.owner
            out.append((alloc_of(op.remote), ('abs', 'block') if r.view is not None else slot, True))
        return out
    if isinstance(op, (ir.JobOp, ir.HwpeOp)) and len(op.bufs):
        for v, s, role in zip(op.bufs, op.slots.data, _roles(op, kdefs)):
            out.append((alloc_of(v), _slot(s.data, var), role in ('out', 'inout')))
    return out


# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pipeline(ModulePass):
    """Expands `for i in pipeline(n)` / `systolic(n, skew=...)` into an explicit
    software pipeline (see the module docstring).

    Example, output-static GEMM: `stage(0)` loads `x[i]`, `w[i]`, enqueues the
    job on `ev[i]`, waits the loads and commits; `stage(1)` waits `ev[i]`. The
    result is a prologue that loads and commits iteration 0, then a loop where
    step `s` loads and commits iteration `s + 1` while RedMulE computes
    iteration `s`, then waits for it.
    """
    name = 'pipeline'

    def apply(self, ctx, module):
        pipes = [op for op in module.walk() if isinstance(op, ir.PipelineOp)]
        if len(pipes) > 1:
            raise MgcError(pipes[1], 'only one pipeline() or systolic() per test is supported for now')
        for p in pipes:
            self.expand(module, p)
        for o in module.walk():
            if isinstance(o, ir.AllocOp) and o.depth.data == 0:
                raise MgcError(o, f'`{o.sym.data}` is not used by a pipeline(): give it a depth= '
                                  '(only pipeline() and systolic() infer it)')
            if isinstance(o, ir.EventsOp) and o.depth.data == 0:
                raise MgcError(o, f'`{o.sym.data}` is not used by a pipeline(): give it a size, '
                                  f'{o.kind.data}.events(n)')

    def expand(self, module, p):
        """Replace the `mg.pipeline` op `p` with prologue + loop."""
        var = p.var.data
        n = p.n.data
        systolic = p.kind.data == 'systolic'
        kdefs = kernels(module)
        docs, blocks = [], []
        for op in list(p.body.block.ops):
            if isinstance(op, ir.StageOp):
                blocks.append(Block(op.k.data, len(blocks), op))
            elif isinstance(op, ir.CommentOp) and not blocks:
                op.detach()
                docs.append(op)
        S = 1 + max(b.k for b in blocks)
        if S < 2:
            raise MgcError(p, f'{p.kind.data}() needs at least two stages: stage(0) and stage(1)')

        # every op in iteration order: by stage, then by block, then in the block
        uses = []
        for b in sorted(blocks, key=lambda b: (b.k, b.pos)):
            for j, o in enumerate(o for top in b.ops for o in top.walk()):
                uses.append(Use(o, b, (b.k, b.pos, j)))

        self.check_syncs(p, uses, systolic)
        self.check_scalars(blocks)
        issues = self.match_events(uses, var)
        self.check_hwpe(uses, issues, blocks, var)
        self.cross_stage_events(module, p, issues, var)
        table = self.depths(module, uses, issues, var, kdefs, remote=not systolic)

        step = Sym(var, None, 'int')
        out = self.prologue(p, blocks, var)
        out += docs
        out.append(self.schedule_doc(p, S, table))
        body = self.step(blocks, var, step, n, S)

        if not systolic:
            hi = add(n, Const(S - 2)) if S > 2 else n
            loop_body = body
            if p.sync is not None:
                loop_body = body + [self.sync(p)]
            out.append(ir.ForOp.create(attributes={'var': StringAttr(var), 'lo': ir.E(Const(0)),
                                                   'hi': ir.E(hi), 'ctype': StringAttr(step.ctype)},
                                       regions=[_region(loop_body)]))
        else:
            skew = p.skew.data
            t = p.time.data if p.time is not None else 't'
            steps = add(self.mesh_max(p, add(skew, n)), Const(S - 2))
            last = add(n, Const(S - 3)) if S >= 3 else sub(n, Const(1))
            active = Logic('&&', [Cmp('>=', step, Const(0)), Cmp('<=', step, last)])
            loop_body = [ir.ScalarOp.create(attributes={'sym': StringAttr(var), 'ctype': StringAttr('int'),
                                                        'value': ir.E(sub(Sym(t), skew)), 'decl': IntAttr(1)}),
                         if_(active, body),
                         self.sync(p)]
            out.append(ir.ForOp.create(attributes={'var': StringAttr(t), 'lo': ir.E(Const(0)),
                                                   'hi': ir.E(steps), 'ctype': StringAttr('int')},
                                       regions=[_region(loop_body)]))
        for o in out:
            for x in o.walk():
                if getattr(x, 'lineno', None) is None:
                    x.lineno = getattr(p, 'lineno', None)
        Rewriter.insert_op(out, InsertPoint.before(p))
        p.detach()
        p.erase()

    # -- checks --------------------------------------------------------------
    def check_syncs(self, p, uses, systolic):
        """`sync()` inside stages: every tile of the scope must run the same
        steps, so the trip count may not depend on the tile; never in systolic()."""
        syncs = [u.op for u in uses if isinstance(u.op, ir.SyncOp)]
        if not syncs:
            return
        if systolic:
            raise MgcError(syncs[0], 'no sync() inside systolic(): tiles run different iterations in the same '
                                     'step; the barrier ending each step is `sync=`')
        mesh, dep, defs = self.tile_deps(p)
        e = self.inline(p.n.data, defs)
        bad = e.syms() & (dep | set(mesh))
        if bad:
            raise MgcError(syncs[0], f'sync() inside pipeline() needs a trip count that is the same on every tile, '
                                     f'but it depends on {", ".join(sorted(bad))}')

    @staticmethod
    def check_scalars(blocks):
        """A scalar declared in a block is used only in that block: elsewhere
        it would belong to another iteration (or to another C scope)."""
        decl = {}
        for b in blocks:
            for o in (o for top in b.ops for o in top.walk()):
                if isinstance(o, ir.ScalarOp) and o.decl.data:
                    if o.sym.data in decl and decl[o.sym.data][0] is not b:
                        raise MgcError(o, f'`{o.sym.data}` is declared in two stage blocks: use two names')
                    decl[o.sym.data] = (b, o)
        for b in blocks:
            for top in b.ops:
                for name in _syms(top):
                    if name in decl and decl[name][0] is not b:
                        raise MgcError(top, f'`{name}` is declared in a stage({decl[name][0].k}) block and used in '
                                            f'another block, where it would be stale: compute it in this block')

    @staticmethod
    def match_events(uses, var) -> List[Issue]:
        """Pair each `.wait()` with the transfers/jobs issued earlier in the same
        iteration on the same event storage (both branches of an `if` may issue
        it); every issue must be waited."""
        pending, issues = {}, []
        for u in uses:
            if _issues(u.op):
                it = Issue(u, _evkey(u.op.event.data, var))
                pending.setdefault(it.key, []).append(it)
                issues.append(it)
            elif isinstance(u.op, ir.WaitOp):
                key = _evkey(u.op.event.data, var)
                its = pending.pop(key, None)
                if not its:
                    raise MgcError(u.op, f'this wait (stage {u.k}) has no transfer or job issued before it in the same '
                                         f'iteration: waits go in the issuing stage or a later one')
                for it in its:
                    it.waits.append(u)
        for its in pending.values():
            raise MgcError(its[0].use.op, f'this transfer/job (stage {its[0].use.k}) is never waited in the loop body')
        return issues

    def check_hwpe(self, uses, issues, blocks, var):
        """Protocol of every explicit job (enqueue -> commit [-> start] -> wait,
        in iteration order), then one simulated run of the schedule: the in-order
        queue must commit/start the job each step means, and hold at most
        `queue_depth` jobs (enqueued, not yet waited)."""
        waits = {id(w.op): it for it in issues for w in it.waits}
        jobs = {}  # acc -> jobs of this iteration, each {'enqueue': Use, 'commit': ..., 'start': ..., 'wait': ...}
        role = {}  # id(op) -> (acc, job, step name)
        for u in uses:
            o = u.op
            if isinstance(o, ir.HwpeOp) and o.action.data == 'enqueue':
                j = {'enqueue': u}
                jobs.setdefault(o.acc.data, []).append(j)
                role[id(o)] = (o.acc.data, j, 'enqueue')
            elif isinstance(o, ir.JobOp) and '.' in o.kernel.data:
                j = {'enqueue': u, 'commit': u, 'start': u}
                jobs.setdefault(o.kernel.data.split('.')[0], []).append(j)
                role[id(o)] = (o.kernel.data.split('.')[0], j, 'oneshot')
            elif isinstance(o, ir.HwpeOp):
                acc, act = o.acc.data, o.action.data
                steps = {'commit': ['commit'], 'start': ['start'], 'commit_start': ['commit', 'start']}[act]
                for s in steps:
                    prev = 'enqueue' if s == 'commit' else 'commit'
                    j = next((j for j in jobs.get(acc, []) if prev in j and s not in j), None)
                    if j is None:
                        raise MgcError(o, f'{acc}.{act}() needs an earlier {acc}.{prev}() in the same iteration '
                                          '(stage order, then block order)')
                    j[s] = u
                role[id(o)] = (acc, j, act)
            elif isinstance(o, ir.WaitOp) and id(o) in waits:
                src = waits[id(o)].use.op
                if id(src) in role:
                    acc, j, _ = role[id(src)]
                    if 'start' not in j:
                        raise MgcError(o, f'this {acc} job is waited before it is started')
                    j['wait'] = u
                    role[id(o)] = (acc, j, 'wait')
        for acc, js in jobs.items():
            for j in js:
                for s, how in (('commit', 'commit_start() or commit()'), ('start', 'start()'), ('wait', 'a .wait()')):
                    if s not in j:
                        raise MgcError(j['enqueue'].op, f'this {acc} job needs {how} in the same iteration')
        if not role:
            return

        # simulated run: 2S + 2 iterations, prologue (step -1) included
        S = 1 + max(b.k for b in blocks)
        n = 2 * S + 2
        queue = {acc: [] for acc in jobs}  # entries [iteration, job, committed, started]
        for s in range(-1, n + S - 2):
            for b in blocks:
                it = s + 1 - b.k
                if not 0 <= it < n or (s == -1 and b.k != 0):
                    continue
                for o in (o for top in b.ops for o in top.walk()):
                    if id(o) not in role:
                        continue
                    acc, j, act = role[id(o)]
                    q = queue[acc]
                    if act in ('enqueue', 'oneshot'):
                        q.append([it, j, act == 'oneshot', act == 'oneshot'])
                        qd = devices.HWPES[acc].queue_depth
                        if len(q) > qd:
                            raise MgcError(o, f'{acc} holds {qd} jobs, but {len(q)} would be in flight here '
                                              f'(iterations {", ".join(str(e[0]) for e in q)}): wait a job earlier')
                    if act in ('commit', 'commit_start'):
                        self.hw_step(o, acc, q, it, j, 2, 'commit')
                    if act in ('start', 'commit_start'):
                        self.hw_step(o, acc, q, it, j, 3, 'start')
                    if act == 'wait':
                        q[:] = [e for e in q if not (e[0] == it and e[1] is j)]

    @staticmethod
    def hw_step(o, acc, q, it, j, f, what):
        """Commit/start the oldest job not yet committed/started, as the hardware
        does; it must be the job of iteration `it` this statement belongs to."""
        e = next((e for e in q if not e[f]), None)
        if e is None or e[0] != it or e[1] is not j:
            got = 'no job' if e is None else f'the job of iteration {e[0]} (stage {e[1]["enqueue"].k})'
            raise MgcError(o, f'{acc}.{what}() of iteration {it} would {what} {got}: the hardware queue '
                              f'{what}s jobs in enqueue order')
        e[f] = True

    @staticmethod
    def cross_stage_events(module, p, issues, var):
        """An event waited in a later stage than the one issuing it is in flight
        for several iterations: give it per-iteration storage, a new event
        array indexed by `var` (DMA events and the default job event)."""
        cross = {}
        for it in issues:
            if it.last > it.use.k and it.key[0] != 'array':
                cross[it.key] = None
        arrays = []
        for key in cross:
            kind, name, d, _ = key
            if kind == 'idma':
                arr, ekind = f'idma_evts_{name}_{"out" if d else "in"}', 'idma'
            else:
                arr, ekind = f'{name}_evts', name
            cross[key] = arr
            arrays.append(ir.EventsOp.create(attributes={'sym': StringAttr(arr), 'depth': IntAttr(0),
                                                         'kind': StringAttr(ekind)}))
        if not cross:
            return
        for it in issues:
            if it.key in cross:
                for u in [it.use] + it.waits:
                    e = u.op.event.data
                    u.op.attributes['event'] = ir.EvtAttr(ir.EvtRef('array', cross[it.key], e.dir, Sym(var)))
                it.key = ('array', cross[it.key], None, ('rel', 0))
        Rewriter.insert_op(arrays, InsertPoint.before(p))

    def depths(self, module, uses, issues, var, kdefs, remote):
        """Stages touched by each storage indexed `[var + c]`: an op touches its
        buffers in its stage, and keeps those of a transfer/job (and its event)
        until its wait. Slot `var + c` in stage k belongs to iteration var + c,
        for which it is stage k - c. Infers omitted depths, checks the others,
        and checks the single-buffer rule. Returns the schedule table
        [(name, lo, hi, depth)]."""
        last = {id(it.use.op): it.last for it in issues}
        span, single = {}, {}
        allocs = {}
        for u in uses:
            hi = last.get(id(u.op), u.k)
            for a, slot, written in _buffers(u.op, var, kdefs, remote):
                allocs[a.sym.data] = a
                if slot[0] == 'rel' and a.depth.data != 1:
                    c = slot[1]
                    lo0, hi0 = span.get(('buf', a.sym.data), (u.k - c, hi - c))
                    span[('buf', a.sym.data)] = (min(lo0, u.k - c), max(hi0, hi - c))
                else:
                    st, w = single.get(a.sym.data, (set(), False))
                    single[a.sym.data] = (st | {u.k}, w or written)
        for name, (st, written) in single.items():
            if written and len(st) > 1:
                ks = sorted(st)
                raise MgcError(allocs[name], f'`{name}` is written in the loop and used in stages '
                                             f'{", ".join(map(str, ks))}: while stage {ks[-1]} works on iteration i, '
                                             f'stage {ks[0]} already writes it for iteration i + {ks[-1] - ks[0]}. '
                                             f'Make it a multi_buffer indexed [{var}], or use it in one stage only')
        evarrays = {o.sym.data: o for o in module.walk() if isinstance(o, ir.EventsOp)}
        for it in issues:
            if it.key[0] == 'array' and it.key[3][0] == 'rel':
                k, c = ('evt', it.key[1]), it.key[3][1]
                lo0, hi0 = span.get(k, (it.use.k - c, it.last - c))
                span[k] = (min(lo0, it.use.k - c), max(hi0, it.last - c))
        table = []
        for (kind, name), (lo, hi) in span.items():
            op = allocs[name] if kind == 'buf' else evarrays[name]
            need = hi - lo + 1
            what = (f'multi-buffer `{name}`' if kind == 'buf' else f'event array `{name}`')
            if op.depth.data == 0:
                op.attributes['depth'] = IntAttr(max(2, need) if kind == 'buf' else need)
            elif op.depth.data < need:
                fix = 'depth=' if kind == 'buf' else f'{op.kind.data}.events('
                raise MgcError(op, f'{what} is used in stages {lo}..{hi}, so it holds {need} iterations at once: '
                                   f'declare it with {fix}{need}{"" if kind == "buf" else ")"} or more (or omit it)')
            table.append((name, lo, hi, op.depth.data))
        return table

    # -- tiles ---------------------------------------------------------------
    @staticmethod
    def tile_deps(p):
        """(mesh axes -> extent macro, names whose value differs between tiles,
        inlinable definitions of those names)."""
        tiles = p.parent_op()
        while not isinstance(tiles, ir.TilesOp):
            tiles = tiles.parent_op()
        mesh = {tiles.y.data: 'MESH_Y_TILES', tiles.x.data: 'MESH_X_TILES'}
        dep = set(mesh)
        defs = {}
        for op in tiles.walk():
            if isinstance(op, ir.SplitOp):
                dep.add(op.sym.data)
            elif isinstance(op, ir.ScalarOp) and op.value.data.syms() & dep:
                dep.add(op.sym.data)
                if op.decl.data:
                    defs[op.sym.data] = op.value.data
                else:
                    defs.pop(op.sym.data, None)  # reassigned: not inlinable
        return mesh, dep, defs

    @staticmethod
    def inline(e, defs):
        while (names := e.syms() & set(defs)):
            e = e.subst({k: defs[k] for k in names})
        return e

    def mesh_max(self, p, e):
        """Maximum of `e` over all tiles, as an expression valid on every tile.

        Scalars that depend on the tile ids are inlined (they must be declared
        once); the result must be affine in the tile ids with constant
        coefficients, so the maximum is at a mesh corner: e.g. `y_id * 2 + n`
        -> `(MESH_Y_TILES - 1) * 2 + n`."""
        mesh, dep, defs = self.tile_deps(p)
        e = self.inline(e, defs)
        bad = (e.syms() & dep) - set(mesh)
        if bad:
            raise MgcError(p, f'systolic(): cannot compute the time steps of the mesh: skew + n depends on '
                              f'{", ".join(sorted(bad))}, which is not a single assignment of the tile ids')
        poly = e.poly()
        corner = {}
        for v, extent in mesh.items():
            coef = poly.get((v,), 0)
            if any(re.search(rf'\b{v}\b', a) for k in poly if k != (v,) for a in k):
                raise MgcError(p, f'systolic(): skew + n must be affine in {v} with a constant coefficient')
            corner[v] = sub(Sym(extent), Const(1)) if coef > 0 else Const(0)
        return e.subst(corner)

    @staticmethod
    def sync(p):
        """Barrier op for the pipeline's `sync=` scope."""
        return ir.SyncOp.create(attributes={'scope': StringAttr(p.sync.data)})

    # -- schedule --------------------------------------------------------------
    @staticmethod
    def guard(k, S, step, n) -> Optional[Expr]:
        """Stage k works on iteration step + 1 - k: needs 0 <= step + 1 - k < n."""
        conds = []
        if k >= 2:
            conds.append(Cmp('>', step, Const(k - 2)))
        if k < S - 1:
            bound = sub(n, Const(1 - k)) if k < 1 else add(n, Const(k - 1))
            conds.append(Cmp('<', step, bound))
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else Logic('&&', conds)

    @staticmethod
    def outer_pending(p):
        """DMA events issued before the pipeline and not yet waited (e.g. a weight
        load right before the loop); the prologue must wait them if needed."""
        pending = []
        for op in p.parent_block().ops:
            if op is p:
                break
            if isinstance(op, ir.IdmaOp):
                pending.append(op.event.data)
            elif isinstance(op, ir.WaitOp):
                pending = [e for e in pending if e.key() != op.event.data.key()]
        return pending

    def prologue(self, p, blocks, var):
        """The stage-0 blocks of iteration 0, in textual order. DMA transfers
        issued before the pipeline and not yet waited are waited before the
        first commit of a job that needs them, or at the end for the jobs of
        later stages."""
        pending = [e for e in self.outer_pending(p) if e.kind == 'idma']
        out, enq = [], {}
        for b in blocks:
            if b.k != 0:
                continue
            for top in b.ops:
                o = instantiate(top, var, Const(0))
                for x in o.walk():
                    if isinstance(x, ir.HwpeOp) and x.action.data == 'enqueue':
                        enq[x.acc.data] = {alloc_of(v).sym.data for v in x.bufs}
                if isinstance(o, ir.HwpeOp) and o.action.data in ('commit', 'commit_start'):
                    names = enq.get(o.acc.data, set())
                    out += [wait(ir.EvtRef(e.kind, e.name, e.dir)) for e in pending if e.name in names]
                    pending = [e for e in pending if e.name not in names]
                out.append(o)
        # operands of the jobs of later stages still in flight from before the pipeline
        need = {alloc_of(v).sym.data for b in blocks for top in b.ops for o in top.walk()
                if isinstance(o, (ir.JobOp, ir.HwpeOp)) for v in o.bufs}
        out += [wait(ir.EvtRef(e.kind, e.name, e.dir)) for e in pending if e.name in need]
        if p.sync is not None:
            out.append(self.sync(p))
        return out

    def step(self, blocks, var, step, n, S):
        """Loop step: every block in textual order, on its iteration
        step + 1 - k, under its stage guard."""
        seq = [(self.guard(b.k, S, step, n), instantiate(top, var, add(step, Const(1 - b.k))))
               for b in blocks for top in b.ops]
        guards = {g.c() for g, _ in seq if g is not None}
        if len(guards) <= 1:
            # one guard: the shape of the hand-written 2-stage tests
            g = next((g for g, _ in seq if g is not None), None)
            if g is None:
                return [o for _, o in seq]
            return [if_(g, [o for _, o in seq], [o.clone() for gg, o in seq if gg is None])]
        out, cur, grp = [], None, []
        for g, o in seq + [(Const(0), None)]:
            key = g.c() if g is not None else None
            if grp and key != cur:
                out += grp if cur is None else [if_(grp_guard, grp)]
                grp = []
            if o is None:
                break
            if not grp:
                cur, grp_guard = key, g
            grp.append(o)
        return out

    @staticmethod
    def schedule_doc(p, S, table):
        """Block comment before the loop: stage/iteration mapping and the
        stages and depth of each rotating storage."""
        var = p.var.data
        lines = [f'{p.kind.data}: {S} stages; step {var} runs stage k on iteration {var} + 1 - k']
        w = max((len(name) for name, *_ in table), default=0)
        for name, lo, hi, depth in table:
            lines.append(f'  {name.ljust(w)}  stages {lo}..{hi}  depth {depth}')
        return ir.CommentOp.create(attributes={'text': StringAttr('\n'.join(lines)), 'style': StringAttr('block')})
