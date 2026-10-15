# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""Hardware descriptions: the only place passes learn what the hardware can do.

* Adding an iDMA mode (3-D, 4-D, strided L1, hardware multicast, ...) means
  editing `IDMA`.
* Adding an HWPE accelerator means adding an `Hwpe` entry (and its mglib
  wrapper); no pass changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# ----------------------------------------------------------------------------
# iDMA
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class IdmaCaps:
    """What one iDMA transfer can do, as exposed by mglib.

    `legalize-idma` uses it to decide whether a tensor view maps to a transfer.
    With the default `IDMA` below (rank 2, dense L1), a 2-D tile of a larger
    matrix is one `mg_idma_memcpy_2d` (rows x contiguous), a full-width block of
    rows collapses to one `mg_idma_memcpy_1d`, and anything needing a third
    dimension is rejected.
    """
    max_rank: int  # dimensions of one transfer (1 = contiguous, 2 = rows x contiguous, ...)
    l1_strided: bool  # can the L1 (OBI) side be strided? (mglib: no)
    functions: Dict[int, str]  # rank -> mglib function
    # multicast scope ('ROW' | 'COL') -> mglib function for a native L1 -> L1
    # multicast (same arguments as the 1-D unicast). Scopes not listed are
    # lowered by `idma-mcast` to one unicast transfer per destination tile.
    multicast: Dict[str, str] = field(default_factory=dict)


IDMA = IdmaCaps(max_rank=2, l1_strided=False,
                functions={1: 'mg_idma_memcpy_1d', 2: 'mg_idma_memcpy_2d'})

# ----------------------------------------------------------------------------
# HWPE accelerators (acquire/enqueue -> commit -> start -> event)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class HwpeJob:
    """One kind of job an accelerator can run, e.g. RedMulE `gemm` (`y += x @ w`).

    `operands`: L1 buffers it takes, as `(name, role, rank)`, e.g.
    `('x', 'in', 2)` is a read-only 2-D buffer. `params`: scalar arguments of the
    mglib call as `(C type, expression)` over operand shapes, e.g.
    `('uint16_t', 'x.shape[0]')`. `constraints`: pairs of shape expressions that
    must be equal, e.g. `('x.shape[1]', 'w.shape[0]')` (inner GEMM dimension).
    `op`: the symbol that selects the job in `enqueue(..., op=GEMM)`; when the
    jobs of an accelerator have one, `op=` is required. `if_uninit`: a job that
    gives the same result when the `inout` operands hold nothing yet (the
    uninit-acc pass switches to it), e.g. `matmul` for `gemm`.
    """
    operands: Tuple[Tuple[str, str, int], ...]  # (name, role in|out|inout, rank)
    params: Tuple[Tuple[str, str], ...]  # (C type, expression over operand shapes)
    constraints: Tuple[Tuple[str, str], ...]  # pairs of shape expressions that must be equal
    op: str = ''  # `op=` symbol in enqueue
    if_uninit: str = ''  # equivalent job when the inout operands are not initialized


@dataclass(frozen=True)
class Hwpe:
    """Description of a memory-mapped accelerator (HWPE) on the tile.

    `name` is the name used in `.mgc` (`redmule.gemm(...)`), `queue_depth` how
    many jobs the hardware can hold at once (2 for RedMulE: one running, one queued), `jobs` the
    supported jobs by name, and `fn` the mglib function used for each protocol
    step (`enqueue`, `oneshot`, `commit`, `start`, `commit_start`, `wait`),
    where `{job}` expands to the job name (`mg_redmule_{job}_enqueue` ->
    `mg_redmule_gemm_enqueue`). Commit and start act on the whole queue,
    whatever its jobs: they take the name of the first job.
    """
    name: str
    queue_depth: int  # hardware job queue
    jobs: Dict[str, HwpeJob]
    fn: Dict[str, str] = field(default_factory=dict)  # action -> mglib function ({job} expands)


HWPES: Dict[str, Hwpe] = {
    'redmule':
    Hwpe(name='redmule',
         queue_depth=2,
         jobs={
             # y += x @ w
             'gemm':
             HwpeJob(operands=(('x', 'in', 2), ('w', 'in', 2), ('y', 'inout', 2)),
                     params=(('uint16_t', 'x.shape[0]'), ('uint16_t', 'x.shape[1]'),
                             ('uint16_t', 'w.shape[1]')),
                     constraints=(('x.shape[1]', 'w.shape[0]'), ('x.shape[0]', 'y.shape[0]'),
                                  ('w.shape[1]', 'y.shape[1]')),
                     op='GEMM', if_uninit='matmul'),
             # z = x @ w (no accumulation: z is not read)
             'matmul':
             HwpeJob(operands=(('x', 'in', 2), ('w', 'in', 2), ('z', 'out', 2)),
                     params=(('uint16_t', 'x.shape[0]'), ('uint16_t', 'x.shape[1]'),
                             ('uint16_t', 'w.shape[1]')),
                     constraints=(('x.shape[1]', 'w.shape[0]'), ('x.shape[0]', 'z.shape[0]'),
                                  ('w.shape[1]', 'z.shape[1]')),
                     op='MATMUL'),
         },
         fn={
             'enqueue': 'mg_redmule_{job}_enqueue',
             'oneshot': 'mg_redmule_{job}',
             'commit': 'mg_redmule_{job}_commit',
             'start': 'mg_redmule_{job}_start',
             'commit_start': 'mg_redmule_{job}_commit_start',
             'wait': 'mg_redmule_wait',
         }),
}

# ----------------------------------------------------------------------------
# Software executors for L1 kernels
# ----------------------------------------------------------------------------

# CORE: the tile's CV32 control core, called synchronously. (Spatz / PULP
# offload will be added as asynchronous executors once their mglib layer exists.)
EXECUTORS = ('CORE',)

# Synchronization scopes -> fsync calls
SYNC = {'GLOBAL': 'fsync_sync_global', 'ROW': 'fsync_sync_row', 'COL': 'fsync_sync_col'}
