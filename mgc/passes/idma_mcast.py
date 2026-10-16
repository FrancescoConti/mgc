# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""idma-mcast: row/column multicasts -> unicast transfers.

`idma.store(q.row, q)` copies `q` into the same slot of `q` on every other
tile of this tile's row. A 1-D block with a `devices.IDMA.multicast` entry
for the scope stays a single native call; otherwise it becomes one unicast per
destination tile:

    for (uint32_t mcast_q = 0; mcast_q < MESH_X_TILES; mcast_q++) {
        if (mcast_q != x_id) {
            mg_idma_memcpy_1d(..., get_l1_base(GET_ID(y_id, mcast_q)) + (...), ...);
        }
    }

All the unicasts stamp the same event. Transfers in one direction complete in
issue order, so waiting on the event (the last transfer) waits for all of
them. The pass runs after event-alloc so the event check sees one issue.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr
from xdsl.rewriter import Rewriter
from xdsl.passes import ModulePass

from .. import devices
from .. import ir
from ..expr import Cmp, Const, Sym
from .common import alloc_of, new_block_region, tiles

MESH = {'row': 'MESH_X_TILES', 'col': 'MESH_Y_TILES'}


@dataclass(frozen=True)
class IdmaMcast(ModulePass):
    """Expands each `mg.idma` to a `row`/`col` `mg.remote` that the iDMA cannot
    multicast natively into an `mg.for` over the row/column, skipping this
    tile, with an absolute-tile (`abs`) unicast in the body."""
    name = 'idma-mcast'

    def apply(self, ctx, module):
        t = tiles(module)
        y, x = Sym(t.y.data), Sym(t.x.data)
        for op in list(module.walk()):
            if not isinstance(op, ir.IdmaOp) or op.remote is None:
                continue
            r = op.remote.owner
            mode = r.mode.data
            if mode not in MESH:
                continue
            if mode.upper() in devices.IDMA.multicast and op.xfer.data.rank == 1:
                continue  # native multicast (1-D blocks only)
            var = Sym(f'mcast_{alloc_of(op.local).sym.data}', None, 'uint32_t')
            ty, tx, me = (y, var, x) if mode == 'row' else (var, x, y)
            rattrs = {'mode': StringAttr('abs'), 'ty': ir.E(ty), 'tx': ir.E(tx)}
            if r.view is not None:
                rattrs['view'] = r.view
            rop = ir.RemoteOp.create(operands=[r.buf], attributes=rattrs, result_types=[ir.BufType()])
            uni = ir.IdmaOp.create(operands=[op.local, rop.res], attributes=dict(op.attributes))
            guard = ir.IfOp.create(attributes={'cond': ir.E(Cmp('!=', var, me))},
                                   regions=[new_block_region([rop, uni]), new_block_region([])])
            loop = ir.ForOp.create(attributes={'var': StringAttr(var.name), 'lo': ir.E(Const(0)),
                                               'hi': ir.E(Sym(MESH[mode])), 'ctype': StringAttr('uint32_t')},
                                   regions=[new_block_region([guard])])
            for o in (rop, uni, guard, loop):
                o.lineno = getattr(op, 'lineno', None)
            Rewriter.replace_op(op, loop, [])
            if not r.res.uses:
                Rewriter.erase_op(r)
