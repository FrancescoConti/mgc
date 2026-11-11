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

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple

from .cexpr import object_macros
from .errors import MgcError

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
    # transfers each direction (L2 -> L1, L1 -> out) holds in its hardware job queue, issued and not yet waited:
    # a pipeline may issue a transfer in an earlier stage than its wait, as an HWPE job, as long as they fit
    queue_depth: int = 1


IDMA = IdmaCaps(max_rank=2, l1_strided=False,
                functions={1: 'mg_idma_memcpy_1d', 2: 'mg_idma_memcpy_2d'},
                # one 2D job (source stride 0) per contiguous run of destination tiles, 1-D blocks only
                multicast={'ROW': 'mg_idma_mcast_row', 'COL': 'mg_idma_mcast_col'},
                # iDMA_JobFifoDepth of each channel (magia_tile_pkg.sv)
                queue_depth=16)

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

# CORE: a function on the tile's CV32 control core, called synchronously.
# SPATZ: the same, for a function that runs tasks on the tile's Spatz through mglib
#   (mg_spatz_start/mg_spatz_run, or the SDK kernels' L1 APIs): the generated code
#   sets Spatz up once, in its prologue (mg_spatz_init), before the first call.
# (PULP offload will be added as an executor once its mglib layer exists.)
EXECUTORS = ('CORE', 'SPATZ')

# Synchronization scopes -> mglib barriers
SYNC = {'GLOBAL': 'mg_fsync_global', 'ROW': 'mg_fsync_row', 'COL': 'mg_fsync_col'}

# ----------------------------------------------------------------------------
# L1
# ----------------------------------------------------------------------------

# Tile address map of the SDK: magia-sdk is $MAGIA_SDK, or the checkout mgc is a submodule of
TILE_ADDR_MAP = os.path.join('targets', 'magia', 'include', 'addr_map', 'tile_addr_map.h')


def _tile_addr_map() -> str:
    sdk = os.environ.get('MAGIA_SDK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    return os.path.abspath(os.path.join(sdk, TILE_ADDR_MAP))


def sdk_macros() -> dict:
    """The object-like macros of the SDK's tile address map (e.g. `L1_SIZE -> '(0x000DFFFF)'`),
    which the `define()`s of a test may use; {} if the map is not found."""
    try:
        with open(_tile_addr_map()) as f:
            return object_macros(f.read())
    except OSError:
        return {}


# bytes of L1 that the mgc buffers leave free at the end of a tile's L1: an access just past the last buffer
# (e.g. an engine overfetching) must not leave the L1, whose end is unmapped. The L1 fit check uses
# sdk_l1_bytes() - L1_MARGIN; the generated C defines it too, for the `define()`s that size buffers over L1_SIZE.
# A .mgc may override it with `L1_MARGIN = define(N)`.
L1_MARGIN = 64

# rows of the M tiles of RedMulE (its array height: 8 on 8 x 8, 16 on 16 x 16; the largest used). The store of the
# last M tile of a job also writes the rows past M up to the end of the tile, with zeros, at the Z row stride: past
# the end of Z. l1-layout leaves that room after every buffer RedMulE writes, rounding its rows up to this.
REDMULE_M_TILE = 16


def sdk_l1_bytes() -> int:
    """Bytes of a tile's L1 for the mgc buffers, L1_SIZE + 1 in the SDK's tile address map (e.g. 0xDFFFF + 1 =
    0xE0000, all the L1 from L1_BASE up to the next tile's), the one source of the L1 size: lower it there to use
    less L1. Raises `MgcError` if the map or the macro is not found (then give mgcc --l1-bytes)."""
    path = _tile_addr_map()
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        raise MgcError(None, f'cannot read the L1 size: {path} not found (set MAGIA_SDK or give --l1-bytes)')
    m = re.search(r'^\s*#define\s+L1_SIZE\s+\(?\s*(0x[0-9a-fA-F]+|\d+)\s*\)?', text, re.M)
    if not m:
        raise MgcError(None, f'cannot read the L1 size: no numeric L1_SIZE in {path} (give --l1-bytes)')
    return int(m.group(1), 0) + 1
