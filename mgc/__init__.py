# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""mgc ("magic"): a Python-syntax DSL for MAGIA mesh tests.

.mgc files are parsed by mgcc.py and never executed. The names below are
placeholders so that `from mgc import *` resolves in editors and linters; the
language itself is documented in mgc/README.md. (The compiler lives in the
sub-modules: frontend, ir, passes, emit_c, driver.)
"""


def _stub(*_a, **_k):
    """Placeholder for every DSL construct: calling it outside the compiler is an error."""
    raise RuntimeError('mgc programs are compiled with mgcc.py, not executed')


# module level
sizes = l2 = define = test = top = l1_kernel = _stub
fp16 = 'fp16'
u8, u16, u32, i32 = 'u8', 'u16', 'u32', 'i32'
WFE = 'WFE'
POLLING = 'POLLING'
CORE = 'CORE'  # L1-kernel executor: the tile's CV32 control core
GLOBAL, ROW, COL = 'GLOBAL', 'ROW', 'COL'  # sync() / pipeline(sync=...) scopes
GEMM, MATMUL = 'GEMM', 'MATMUL'  # redmule.enqueue(..., op=...): y += x @ w / z = x @ w

# loop levels: spatial (tiles), time (range / pipeline), reserved (cores)
tiles = pipeline = cores = check = comment = sync = rotate = _stub
LAST = 'LAST'  # last iteration of the preceding pipeline() / systolic(), e.g. `b[LAST]`


class _Ns:
    """Namespace stub (`L1`, `idma`, `redmule`): any attribute is a placeholder function."""

    def __getattr__(self, _):
        return _stub


L1 = _Ns()  # L1.single_buffer(view), L1.multi_buffer(view, depth=N)
idma = _Ns()  # idma.load(dst, src), idma.store(dst, src)
redmule = _Ns()  # gemm / matmul / enqueue / commit / start / commit_start / events

__all__ = ['sizes', 'l2', 'define', 'test', 'top', 'l1_kernel', 'fp16', 'u8', 'u16', 'u32', 'i32', 'WFE', 'POLLING',
           'CORE', 'GLOBAL', 'ROW', 'COL', 'GEMM', 'MATMUL', 'tiles', 'pipeline', 'cores', 'check', 'comment', 'sync', 'rotate',
           'LAST', 'L1', 'idma', 'redmule']
