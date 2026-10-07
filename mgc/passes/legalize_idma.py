# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""legalize-idma: tensor views -> iDMA transfers.

The only pass that knows the iDMA's capabilities (devices.IDMA). A view is
turned into a contiguous run of `len` bytes repeated over outer dimensions
(innermost first), after folding dimensions that are contiguous with their
inner neighbour. The result must fit `IDMA.max_rank`; the L1 side is dense.

Each mg.alloc gets the geometry of its declaration view (`xfer`) and its
base L2 address (`axi`). Each mg.idma is checked against its buffer's geometry
and gets its L2 address as the buffer's base plus an offset.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.passes import ModulePass

from .. import devices
from .. import ir
from ..errors import MgcError
from ..expr import BinOp, Const, Paren, Raw, Sym, const_of, equal, is_zero, product, sub
from ..frontend import Tensor
from .common import alloc_of, esize, tensors


def geometry(node, view: ir.View, t, caps=None) -> ir.Xfer:
    """Describe a tensor view as an iDMA transfer (`ir.Xfer`).

    `view` is the slice of L2 tensor `t`; `node` is used for error locations;
    `caps` defaults to `devices.IDMA`. Dimensions of size 1 are dropped and a
    dimension is folded into its inner neighbour when the inner one is a full row.

    Examples for fp16 `X` of shape (M, N):
    - `X[0:8, :]` (full rows)   -> 1-D: `8*N*2` contiguous bytes;
    - `X[0:8, 0:16]` (a block)  -> 2-D: `32` bytes repeated 8 times, L2 stride `N*2`;
    - `X[3, 0:16]`              -> 1-D: 32 bytes.
    Raises `MgcError` if the innermost dimension is strided or more dimensions
    than `caps.max_rank` remain.
    """
    caps = caps or devices.IDMA  # looked up at call time: the table is the extension point
    dims = [[s, t.stride(d)] for d, (s, k) in enumerate(zip(view.sizes, view.kept))
            if k and const_of(s) != 1]
    merged = []
    for s, st in dims:
        # an outer dim folds into the inner one if the inner one is a full row
        if merged and equal(merged[-1][1], product(s, st)):
            merged[-1] = [product(merged[-1][0], s), st]
        else:
            merged.append([s, st])
    if not merged:
        merged = [[Const(1), Const(1)]]
    if const_of(merged[-1][1]) != 1:
        raise MgcError(node, 'the innermost transferred dimension must be contiguous')
    if len(merged) > caps.max_rank:
        raise MgcError(node, f'mglib iDMA transfers are at most {caps.max_rank}-D; this slice needs '
                       f'{len(merged)} non-contiguous dimensions')
    es = Const(t.esize)
    inner = merged[-1][0]
    outer = [(r, product(st, es)) for r, st in reversed(merged[:-1])]
    return ir.Xfer(product(inner, es), outer)


def same_xfer(a: ir.Xfer, b: ir.Xfer) -> bool:
    """True if two transfers have identical rank, length, repetitions and strides."""
    return (a.rank == b.rank and equal(a.len, b.len)
            and all(equal(r1, r2) and equal(s1, s2) for (r1, s1), (r2, s2) in zip(a.outer, b.outer)))


def base_addr(view: ir.View, t) -> BinOp:
    """L2 byte address of the first element of `view`, as a C expression:
    `(uint32_t)x_inp + (row_start * N * 2) + (col_start * 2)`."""
    e = Raw(f'(uint32_t){t.cname}')
    for d, s in enumerate(view.starts):
        if not is_zero(s):
            e = BinOp('+', e, Paren(product(s, t.stride(d), Const(t.esize))))
    return e


def block_offset(view: ir.View, t):
    """Byte offset of the first element of `view` in a dense buffer shaped like
    `t`: `(x_id * tile_h * 2)` for row `x_id` of a (MESH_X_TILES, tile_h) fp16 buffer."""
    off = None
    for k, s in enumerate(view.starts):
        if not is_zero(s):
            o = product(s, t.stride(k), Const(t.esize))
            off = o if off is None else BinOp('+', off, o)
    return off if off is not None else Const(0)


@dataclass(frozen=True)
class LegalizeIdma(ModulePass):
    """Gives every `mg.alloc` and `mg.idma` its transfer geometry and L2 address.

    Example: `x = L1.single_buffer(X[0:8, 0:16])` gets a 2-D `xfer` and base address
    `l2_addr_x`; a later `idma.load(x, X[8:16, 0:16])` is checked to have the same
    geometry and gets address `l2_addr_x + (8 * N * 2)`. A load with a different
    geometry or shape than the buffer's declaration is an `MgcError`.
    """
    name = 'legalize-idma'

    def apply(self, ctx, module):
        ts = tensors(module)
        for op in module.walk():
            if isinstance(op, ir.AllocOp):
                v = op.view.data
                if v.tensor is None:
                    # L1-only buffer: dense, no L2 address
                    op.attributes['xfer'] = ir.XferAttr(ir.Xfer(product(*v.shape, Const(esize(op, ts)))))
                    continue
                t = ts[v.tensor]
                op.attributes['xfer'] = ir.XferAttr(geometry(op, v, t))
                op.attributes['axi'] = ir.E(base_addr(v, t))
            elif isinstance(op, ir.IdmaOp):
                self.idma(op, ts)

    def idma(self, op, ts):
        """Check and annotate one `mg.idma`: neighbour stores are dense copies of
        the whole buffer, or of the buffer (or of a contiguous block of it, at byte
        offset `obi`) into a block of another one (`axi` is then the byte offset of
        the block, computed with this tile's sizes); L2 transfers must match the buffer's declared shape and
        geometry, and get the L2 address as base plus per-dimension offset."""
        a = alloc_of(op.local)
        decl = a.view.data
        if op.remote is not None and op.remote.owner.view is not None:
            # L1 -> a block of another buffer on other tile(s): the block is laid out
            # in the destination buffer like a view of an L2 tensor of the buffer's shape
            d = alloc_of(op.remote)
            view = op.remote.owner.view.data
            src = op.src.data if op.src is not None else None
            shp, bshp = view.shape, (src or decl).shape
            sname = f'`{a.sym.data}` block' if src else f'`{a.sym.data}`'
            if len(shp) != len(bshp) or not all(equal(x, y) for x, y in zip(shp, bshp)):
                raise MgcError(op, f'shape mismatch: the `{d.sym.data}` block is {shp}, {sname} is {bshp}')
            if d.dtype.data != a.dtype.data:
                raise MgcError(op, f'`{a.sym.data}` and `{d.sym.data}` have different dtypes')
            t = Tensor(d.sym.data, None, d.view.data.shape, d.dtype.data)
            op.attributes['xfer'] = ir.XferAttr(geometry(op, view, t))
            op.attributes['axi'] = ir.E(block_offset(view, t))
            if src is not None:
                # the L1 side is dense: the local block must be contiguous
                ts = Tensor(a.sym.data, None, decl.shape, a.dtype.data)
                if geometry(op, src, ts).rank != 1:
                    raise MgcError(op, f'the `{a.sym.data}` block must be contiguous')
                op.attributes['obi'] = ir.E(block_offset(src, ts))
            return
        if op.remote is not None:
            # L1 -> same slot on a neighbour tile: both sides dense
            op.attributes['xfer'] = ir.XferAttr(ir.Xfer(product(*decl.shape, Const(esize(a, ts)))))
            return
        view = op.l2.data
        t = ts[view.tensor]
        shp, bshp = view.shape, decl.shape
        if len(shp) != len(bshp) or not all(equal(x, y) for x, y in zip(shp, bshp)):
            raise MgcError(op, f'shape mismatch: {view.tensor} slice is {shp}, `{a.sym.data}` is {bshp}')
        g = geometry(op, view, t)
        if not same_xfer(g, a.xfer.data):
            raise MgcError(op, f'transfer geometry differs from the one `{a.sym.data}` was declared with')
        if view.tensor == decl.tensor:
            axi = Sym(f'l2_addr_{a.sym.data}')
            for d, (s0, s1) in enumerate(zip(decl.starts, view.starts)):
                diff = sub(s1, s0, fold=False)
                if is_zero(diff):
                    continue
                if is_zero(s0):
                    diff = s1
                axi = BinOp('+', axi, Paren(product(diff, t.stride(d), Const(t.esize))))
        else:
            axi = base_addr(view, t)
        op.attributes['xfer'] = ir.XferAttr(a.xfer.data)
        op.attributes['axi'] = ir.E(axi)
