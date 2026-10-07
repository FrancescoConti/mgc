# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""pipeline: `mg.pipeline` -> explicit, predicated software pipeline.

Body: DMA transfers and jobs (optionally inside tile-invariant `if`s), written
for one iteration `i`. The pass

1. **assigns stages by dependency hops**: an item's stage is 1 + the stage of
   the latest earlier item that writes a slot it reads (same buffer, same
   index), else 0. For WS: loads 0, gemm 1, store 2.
2. **checks multi-buffer depths**: a slot is live from its first to its last
   stage, so depth >= last - first + 1.
3. **generates** a prologue (stage 0 of iteration 0) and a step loop. In step
   `s`, stage k works on iteration s + 1 - k, guarded by 0 <= s + 1 - k < n.

An accelerator job is either a one-shot call (`redmule.gemm(...)`: issued and
waited in its step) or an explicit HWPE protocol, one step per statement:

    redmule.enqueue(x[i], w[i], y, event=ev[i])   # program (addresses only)
    redmule.commit_start()                         # or commit() [+ start()]
    ev[i].wait()                                   # lag=1 by default

The protocol steps are placed as follows (`lag=k` adds k stages):
* commit: in the stage of its latest producer, after it in the step (after
  the DMA's wait, or after the L1-kernel call); one stage later if the
  producer is an asynchronous job or the operand is written by another tile
  (valid only after the barrier);
* enqueue: with its commit, issued together with the transfers;
* start: commit + lag; wait: start (or commit_start) + lag, default lag 1.
  The wait is the writer of the job's outputs for later consumers.
Checked: one protocol per accelerator, complete and in order; at most
`queue_depth` jobs between enqueue and wait, and as many event-array entries.

Within a step: (1) transfers, enqueues and one-shot jobs are issued; (2)
commits/starts follow, each after the waits of the DMAs it needs, so the HWPE
runs while (3) the CV32 L1 kernels execute, each kernel followed by the
commits that consume its output; (4) the remaining waits come last, in issue
order (issues as early as possible, waits as late as possible).

`skew=`/`steps=`/`sync=` make it a systolic pipeline: a global time loop of
`steps` iterations in which the tile is active when 0 <= t - skew <= last
step, with a barrier at the end of every time step.

Code shape matters to GCC (it changes register allocation and the cycle count),
so the 2-stage shape is the one of the hand-written tests:
`if (next) {...; wait} else {wait}`; deeper pipelines use per-stage guards.
"""

from __future__ import annotations

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


# ----------------------------------------------------------------------------
# body analysis
# ----------------------------------------------------------------------------


@dataclass(eq=False)
class Item:
    """One schedulable element of a pipeline body (a DMA, a job, an HWPE
    protocol step, or an `if` wrapping transfers and jobs), with the buffer
    slots it reads/writes, the events it signals and the `stage` assigned to it.

    Protocol steps (`proto` set) belong to the `job` enqueued by an `enqueue`
    item, whose `parts` names its `commit`, `start` and `wait` steps
    (`commit_start` is both commit and start). `live`: slots a step keeps busy
    (all the job's operands, from commit to wait), for the depth check."""
    op: object  # IdmaOp | JobOp | HwpeOp | WaitOp | IfOp (predicated)
    reads: set = field(default_factory=set)
    writes: set = field(default_factory=set)
    events: list = field(default_factory=list)  # (EvtRef, unconditional)
    is_job: bool = False
    stage: int = 0
    proto: Optional[str] = None  # enqueue | commit | start | commit_start | wait
    acc: Optional[str] = None
    lag: int = 0
    job: Optional['Item'] = None
    parts: dict = field(default_factory=dict)  # enqueue only: commit / start / wait -> Item
    live: set = field(default_factory=set)
    prods: list = field(default_factory=list)  # earlier items writing a slot this one reads

    @property
    def is_transfer(self):
        return self.proto is None and not self.is_job


def _offset(idx: Expr, var: str):
    """Slot index -> ('rel', k) for var + k, ('abs', text) otherwise."""
    if var in idx.syms():
        k = const_of(sub(idx, Sym(var), fold=False))
        if k is None:
            raise MgcError(None, f'slot index `{idx.c()}` must be `{var} + constant`')
        return ('rel', k)
    return ('abs', idx.c())


def _accesses(op, var, kdefs):
    """(reads, writes) of buffer slots for a DMA or job."""
    if isinstance(op, ir.IdmaOp):
        key = (alloc_of(op.local).sym.data, _offset(op.slot.data, var))
        return ({key}, set()) if op.dir.data == 1 else (set(), {key})
    roles = _roles(op, kdefs)
    reads, writes = set(), set()
    for v, s, role in zip(op.bufs, op.slots.data, roles):
        key = (alloc_of(v).sym.data, _offset(s.data, var))
        if role in ('in', 'inout'):
            reads.add(key)
        if role in ('out', 'inout'):
            writes.add(key)
    return reads, writes


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


def _item(op, var, kdefs) -> Item:
    """Analyse one pipeline-body op into an `Item`; rejects anything that is not
    a transfer, a job or an `if` independent of the iteration variable."""
    if isinstance(op, (ir.IdmaOp, ir.JobOp)):
        r, w = _accesses(op, var, kdefs)
        ev = [] if op.event is None else [(op.event.data, True)]
        return Item(op, r, w, ev, isinstance(op, ir.JobOp), live=r | w)
    if isinstance(op, ir.HwpeOp):
        # dependencies and live slots are filled in when the steps are bound to their enqueue
        lag = op.lag.data if op.lag is not None else 0
        return Item(op, proto=op.action.data, acc=op.acc.data, lag=lag)
    if isinstance(op, ir.WaitOp):
        if op.event.data.kind == 'idma':
            raise MgcError(op, 'DMA waits inside pipeline() are placed automatically: drop this `.wait()`')
        lag = op.lag.data if op.lag is not None else 1
        return Item(op, proto='wait', lag=lag)
    if isinstance(op, ir.IfOp):
        if var in op.cond.data.syms():
            raise MgcError(op, f'conditions inside pipeline() must not depend on `{var}` (they predicate whole items)')
        it = Item(op)
        branches = []
        for reg in (op.then, op.else_):
            keys = set()
            for o in reg.block.ops:
                if isinstance(o, (ir.CommentOp, ir.RemoteOp)):
                    continue
                if not isinstance(o, (ir.IdmaOp, ir.JobOp)):
                    raise MgcError(o, 'a predicated pipeline item contains only transfers and jobs')
                r, w = _accesses(o, var, kdefs)
                it.reads |= r
                it.writes |= w
                it.is_job |= isinstance(o, ir.JobOp)
                if o.event is not None:
                    keys.add(o.event.data.key())
                    if not any(e.key() == o.event.data.key() for e, _ in it.events):
                        it.events.append((o.event.data, False))
            branches.append(keys)
        both = branches[0] & branches[1]
        it.events = [(e, e.key() in both) for e, _ in it.events]
        it.live = it.reads | it.writes
        return it
    raise MgcError(op, 'pipeline() bodies contain transfers, jobs, HWPE protocol steps and tile-invariant ifs')


def _same_event(a: ir.EvtRef, b: ir.EvtRef):
    """True if `a` and `b` name the same event storage (`ev[i]` and `ev[i]`)."""
    ia = a.index.c() if a.index is not None else None
    ib = b.index.c() if b.index is not None else None
    return (a.kind, a.name, ia) == (b.kind, b.name, ib)


def _bind(items, var, kdefs):
    """Attach every protocol step to its enqueue, check the protocol order
    (enqueue -> commit [-> start] or commit_start -> wait, one job per
    accelerator) and give each step the job's slots."""
    jobs = {}  # acc -> enqueue item
    for it in items:
        if it.proto is None:
            continue
        if it.proto == 'enqueue':
            if it.acc in jobs:
                raise MgcError(it.op, f'one {it.acc} job per pipeline() body: the hardware queue commits '
                               'in order, so two jobs would interleave')
            jobs[it.acc] = it.job = it
            r, w = _accesses(it.op, var, kdefs)
            it.live = r | w
            continue
        if it.proto == 'wait':
            e = it.op.event.data
            j = next((q for q in jobs.values() if _same_event(q.op.event.data, e)), None)
            if j is None:
                raise MgcError(it.op, 'inside pipeline(), `.wait()` is for the event of a job enqueued earlier '
                               'in the same body')
            it.acc = j.acc
        else:
            j = jobs.get(it.acc)
            if j is None:
                raise MgcError(it.op, f'{it.acc}.{it.proto}() needs an earlier {it.acc}.enqueue() in the same '
                               'pipeline() body')
        it.job = j
        p = j.parts
        if it.proto in ('commit', 'commit_start') and 'commit' in p:
            raise MgcError(it.op, f'this {it.acc} job is already committed')
        if it.proto == 'start' and ('commit' not in p or 'start' in p):
            raise MgcError(it.op, f'{it.acc}.start() needs an earlier {it.acc}.commit() of the same job, '
                           'and only one start')
        if it.proto == 'wait' and ('start' not in p or 'wait' in p):
            raise MgcError(it.op, 'the job is waited before it is started, or waited twice')
        if it.proto in ('commit', 'commit_start'):
            p['commit'] = it
        if it.proto in ('start', 'commit_start'):
            p['start'] = it
        if it.proto == 'wait':
            p['wait'] = it
        r, w = _accesses(j.op, var, kdefs)
        it.live = r | w
        if it.proto in ('commit', 'commit_start'):
            it.reads = r  # the operands must be in L1 when the job is committed
        if it.proto == 'wait':
            it.writes = w  # ... and its results are there once it is waited
    for acc, j in jobs.items():
        for part, how in (('commit', 'commit_start() or commit()'), ('start', 'start()'), ('wait', 'a .wait()')):
            if part not in j.parts:
                raise MgcError(j.op, f'this {acc} job needs {how} in the same pipeline() body')
        for it in items:
            if it.is_job and any(isinstance(o, ir.JobOp) and o.kernel.data.startswith(acc + '.')
                                 for o in it.op.walk()):
                raise MgcError(it.op, f'{acc} is driven explicitly (enqueue / commit / wait) in this pipeline(): '
                               f'write this job the same way')


# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pipeline(ModulePass):
    """Expands `for i in pipeline(n)` into an explicit software pipeline.

    Example, output-static GEMM body (`idma.load(x[i], ...)`, `redmule.enqueue(x[i], w, y,
    event=ev[i])`, `redmule.commit_start()`, `ev[i].wait()`): the load, enqueue and
    commit are stage 0 and the wait stage 1, so the result is a prologue that
    loads and commits iteration 0, then a loop where step `s` loads and commits
    iteration `s + 1` while RedMulE computes iteration `s`, then waits for it.
    See the module docstring for the stage/guard rules.
    Errors: empty body, no dependency between items (nothing to pipeline),
    multi-buffer too shallow for the stages it spans, more than one pipeline,
    incomplete or misordered protocol, too many jobs in flight.
    """
    name = 'pipeline'

    def apply(self, ctx, module):
        pipes = [op for op in module.walk() if isinstance(op, ir.PipelineOp)]
        if len(pipes) > 1:
            raise MgcError(pipes[1], 'only one pipeline() per test is supported for now')
        for p in pipes:
            self.expand(module, p)

    def expand(self, module, p):
        """Replace the `mg.pipeline` op `p` with prologue + loop (steps 1-3 of the module docstring)."""
        var = p.var.data
        n = p.n.data
        kdefs = kernels(module)
        docs, items = [], []
        for op in list(p.body.block.ops):
            if isinstance(op, ir.CommentOp):
                if not items:
                    op.detach()
                    docs.append(op)
                continue
            items.append(_item(op, var, kdefs))
        if not items:
            raise MgcError(p, 'empty pipeline() body')
        _bind(items, var, kdefs)

        # 1. stages by dependency hops (HWPE protocol steps: see the module docstring)
        remote_written = {alloc_of(r.res).sym.data for r in module.walk() if isinstance(r, ir.RemoteOp)}
        for k, it in enumerate(items):
            it.prods = [q for q in items[:k] if q.writes & it.reads]
            if it.proto == 'enqueue':
                continue  # placed with its commit, below
            if it.proto in ('commit', 'commit_start'):
                it.stage = it.lag + max((self.commit_stage(q, it, remote_written) for q in it.prods), default=0)
            elif it.proto == 'start':
                it.stage = it.job.parts['commit'].stage + it.lag
            elif it.proto == 'wait':
                it.stage = it.job.parts['start'].stage + it.lag
            else:
                it.stage = 1 + max(q.stage for q in it.prods) if it.prods else 0
        for it in items:
            if it.proto == 'enqueue':
                it.stage = it.parts['commit'].stage
        S = 1 + max(it.stage for it in items)
        if S < 2:
            raise MgcError(p, 'nothing to pipeline: no item consumes what another produces')

        # 2. jobs in flight, multi-buffer depths
        self.check_in_flight(module, items, var)
        allocs = {a.sym.data: a for a in module.walk() if isinstance(a, ir.AllocOp)}
        span = {}
        for it in items:
            for b, (kind, _) in it.live:
                if kind == 'rel':
                    lo, hi = span.get(b, (it.stage, it.stage))
                    span[b] = (min(lo, it.stage), max(hi, it.stage))
        for b, (lo, hi) in span.items():
            need = hi - lo + 1
            if allocs[b].depth.data < need:
                raise MgcError(allocs[b], f'multi-buffer `{b}` is live across {need} pipeline stages: '
                               f'declare it with depth={need} or more')

        skew = p.skew.data if p.skew is not None else None
        step = Sym(var, None, 'int')
        out = self.prologue(p, items, var)
        out += docs
        body = self.step(items, var, step, n, S)

        if skew is None:
            hi = add(n, Const(S - 2)) if S > 2 else n
            loop_body = body
            if p.sync is not None:
                loop_body = body + [self.sync(p)]
            out.append(ir.ForOp.create(attributes={'var': StringAttr(var), 'lo': ir.E(Const(0)),
                                                   'hi': ir.E(hi), 'ctype': StringAttr(step.ctype)},
                                       regions=[_region(loop_body)]))
        else:
            t = p.time.data if p.time is not None else 't'
            steps = p.steps.data
            last = add(n, Const(S - 3)) if S >= 3 else sub(n, Const(1))
            active = Logic('&&', [Cmp('>=', step, Const(0)), Cmp('<=', step, last)])
            loop_body = [ir.ScalarOp.create(attributes={'sym': StringAttr(var), 'ctype': StringAttr('int'),
                                                        'value': ir.E(sub(Sym(t), skew)), 'decl': IntAttr(1)}),
                         if_(active, body)]
            if p.sync is not None:
                loop_body.append(self.sync(p))
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

    @staticmethod
    def sync(p):
        """Barrier op for the pipeline's `sync=` scope."""
        return ir.SyncOp.create(attributes={'scope': StringAttr(p.sync.data)})

    @staticmethod
    def is_sync(it: Item) -> bool:
        """True if the item runs a synchronous software kernel (occupies the core)."""
        ops = [it.op] if not isinstance(it.op, ir.IfOp) else \
            [o for r in (it.op.then, it.op.else_) for o in r.block.ops]
        return any(isinstance(o, ir.JobOp) and '.' not in o.kernel.data for o in ops)

    # -- guards ------------------------------------------------------------
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

    # -- outer pending events (issued before the pipeline, not yet waited) ---
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

    @staticmethod
    def waits_for(it: Item, pred_op=None):
        """Waits for an item's events (predicated ones inside the item's condition)."""
        out = []
        for e, uncond in it.events:
            w = wait(ir.EvtRef(e.kind, e.name, e.dir, e.index))
            out.append(w if uncond or not isinstance(it.op, ir.IfOp) else if_(it.op.cond.data, [w]))
        return out

    # -- HWPE protocol: placement and checks -------------------------------------
    def commit_stage(self, q: Item, c: Item, remote_written) -> int:
        """Earliest stage at which commit `c` may follow its producer `q`: the
        same stage (after the DMA's wait, or after the L1-kernel call), or the
        next one if `q` is an asynchronous job / job wait, or if the operand is
        written by another tile (valid only after the barrier)."""
        if any(b in remote_written for b, _ in q.writes & c.reads):
            return q.stage + 1
        if q.is_transfer or (q.is_job and self.is_sync(q)):
            return q.stage
        return q.stage + 1

    @staticmethod
    def check_in_flight(module, items, var):
        """At most `queue_depth` jobs between enqueue and wait, and an event
        storage for each of them. Step `s` names the events `ev[s + 1 - stage]`,
        so the array must also be deeper than the largest such offset."""
        evarrays = {o.sym.data: o.depth.data for o in module.walk() if isinstance(o, ir.EventsOp)}
        for j in items:
            if j.proto != 'enqueue':
                continue
            w = j.parts['wait']
            n = w.stage - j.stage + 1
            qd = devices.HWPES[j.acc].queue_depth
            if n > qd:
                raise MgcError(w.op, f'{j.acc} holds {qd} jobs, but {n} would be in flight (enqueue at stage '
                               f'{j.stage}, wait at stage {w.stage}): lower `lag=`')
            e = j.op.event.data
            need = max(n, abs(1 - j.stage) + 1, abs(1 - w.stage) + 1)
            if e.kind == 'array' and evarrays.get(e.name, need) < need:
                raise MgcError(j.op, f'{n} jobs are in flight (enqueue at stage {j.stage}, wait at stage '
                               f'{w.stage}): declare `{e.name}` with {j.acc}.events({need}) or more')
            if e.kind != 'array' and n > 1:
                raise MgcError(j.op, f'{n} jobs are in flight, each needs its own event: pass '
                               f'`event=ev[{var}]` with ev = {j.acc}.events({n})')

    # -- schedule --------------------------------------------------------------
    def order(self, sel):
        """Issue order of the items `sel` running in one step, as a list of
        ('issue', item) / ('wait', item) (the item's DMA or one-shot job waits):

        1. transfers, enqueues and one-shot jobs;
        2. commits/starts, each after the waits of the same-stage DMAs it needs;
        3. L1-kernel calls, each followed by the commits/starts that need it;
        4. the remaining waits, in issue order.
        """
        def stage(it):
            return it.stage

        transfers = sorted([it for it in sel if it.is_transfer], key=stage)
        enqueues = [it for it in sel if it.proto == 'enqueue']
        # asynchronous jobs before synchronous (core) calls: a call occupies the
        # control core, so everything that can run meanwhile must already be issued
        jobs = sorted([it for it in sel if it.is_job], key=lambda it: (self.is_sync(it), it.stage))
        asyncs = [it for it in jobs if not self.is_sync(it)]
        calls = [it for it in jobs if self.is_sync(it)]
        steps = [it for it in sel if it.proto in ('commit', 'start', 'commit_start')]
        waited = set()

        def commit_of(c):
            return c.job.parts['commit']

        def call_before(c):
            """The same-stage L1-kernel call a commit/start must follow, if any."""
            cm = commit_of(c)
            if cm.stage != c.stage:
                return None
            ks = [q for q in cm.prods if q in sel and q.stage == c.stage and q.is_job and self.is_sync(q)]
            return ks[-1] if ks else None

        def ready(c):
            out = []
            for q in commit_of(c).prods:
                if q in sel and q.stage == c.stage and q.is_transfer and q not in waited:
                    waited.add(q)
                    out.append(('wait', q))
            return out + [('issue', c)]

        out = [('issue', it) for it in transfers + enqueues + asyncs]
        for c in steps:
            if call_before(c) is None:
                out += ready(c)
        for k in calls:
            out.append(('issue', k))
            for c in steps:
                if call_before(c) is k:
                    out += ready(c)
        out += [('wait', it) for it in transfers + asyncs if it not in waited]
        out += [('issue', it) for it in sel if it.proto == 'wait']
        return out

    def emit(self, tokens, var, at, guard):
        """(guard, op) list for `order()` tokens: each item instantiated for its
        iteration `at(item)`, under `guard(item)`."""
        seq = []
        for kind, it in tokens:
            ops = [it.op] if kind == 'issue' else self.waits_for(it)
            for o in ops:
                c = instantiate(o, var, at(it))
                if isinstance(c, (ir.HwpeOp, ir.WaitOp)):
                    c.attributes.pop('lag', None)
                seq.append((guard(it), c))
            if kind == 'wait':
                for o in ops:
                    o.erase()
        return seq

    def prologue(self, p, items, var):
        """Stage 0 of iteration 0, in step order. DMA transfers issued before the
        pipeline and not yet waited are waited before the first commit that
        needs them, or at the end for jobs of later stages."""
        sel = [it for it in items if it.stage == 0]
        pending = [e for e in self.outer_pending(p) if e.kind == 'idma']
        out = []
        for g, o in self.emit(self.order(sel), var, lambda it: Const(0), lambda it: None):
            if isinstance(o, ir.HwpeOp) and o.action.data in ('commit', 'commit_start'):
                job = next(it for it in sel if it.proto == 'enqueue' and it.acc == o.acc.data)
                names = {alloc_of(v).sym.data for v in job.op.bufs}
                out += [wait(ir.EvtRef(e.kind, e.name, e.dir)) for e in pending if e.name in names]
                pending = [e for e in pending if e.name not in names]
            out.append(o)
        # operands of the jobs of later stages still in flight from before the pipeline
        need = {alloc_of(v).sym.data for it in items
                if (it.is_job and isinstance(it.op, ir.JobOp)) or it.proto == 'enqueue' for v in it.op.bufs}
        out += [wait(ir.EvtRef(e.kind, e.name, e.dir)) for e in pending if e.name in need]
        if p.sync is not None:
            out.append(self.sync(p))
        return out

    def step(self, items, var, step, n, S):
        """Loop step: every stage's items for their iterations (stage k works on
        step + 1 - k), in `order()`, each under its stage guard."""
        seq = self.emit(self.order(items), var, lambda it: add(step, Const(1 - it.stage)),
                        lambda it: self.guard(it.stage, S, step, n))
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

