# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""Compilation driver: front-end -> passes -> C."""

from __future__ import annotations

import io
import warnings

from xdsl.context import Context
from xdsl.printer import Printer

from . import ir
from .cexpr import evaluate, object_macros, undefined
from .devices import L1_MARGIN, sdk_l1_bytes, sdk_macros
from .emit_c import emit
from .errors import MgcError, MgcWarning
from .expr import Const
from .frontend import parse, parse_header
from .ir import MG
from .passes import PASSES

PASS_NAMES = ['frontend'] + [p.name for p in PASSES]


def print_ir(module) -> str:
    """Return the textual form of the IR module (what `--print-ir-after` shows)."""
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue() + '\n'


def check_l1(module, header, tiles, l1_bytes=None):
    """Raise `MgcError` if the L1 buffers of a tile do not fit `l1_bytes` minus `L1_MARGIN` on a
    `tiles` x `tiles` mesh, for each of the mesh sides `tiles`. `l1_bytes` None: read from the SDK
    (`devices.sdk_l1_bytes`, an error if it cannot be found). `L1_MARGIN`: the .mgc's
    `define()` if it has one, else `devices.L1_MARGIN` (64 bytes).

    The footprint (`l1_ends` from l1-layout) is evaluated with the `define()`s, the macros of
    the test header and of the SDK's tile address map (C expressions too, e.g.
    `define("(L1_SIZE + 1) / 2")`; see `cexpr`), `MESH_X_TILES = MESH_Y_TILES = tiles` and the
    tile-split maxima `h_max = ceil(E / tiles)`. Buffers whose size cannot be computed (run-time
    values, unknown macros) are left out with an `MgcWarning` naming them and the missing
    values; the others are still checked, a lower bound of the footprint.

    Example: `S_BLOCK_SIZE_Q = define(1024)` on a 2 x 2 mesh splits each Q block in 512 rows per
    tile; a 512 x 512 fp16 buffer with 3 slots (1.5 MiB) does not fit the 896 KiB of L1.
    """
    if not tiles or 'l1_ends' not in module.attributes:
        return
    if l1_bytes is None:
        l1_bytes = sdk_l1_bytes()
    macros = {**sdk_macros(), **object_macros(parse_header(header)[1]), 'L1_MARGIN': '(%d)' % L1_MARGIN}
    for d in module.walk():
        if isinstance(d, ir.DefineOp):
            macros[d.sym.data] = d.text.data
    splits = [s for s in module.walk() if isinstance(s, ir.SplitOp)]
    ends = [e.data for e in module.attributes['l1_ends'].data]
    bufs = [b.data for b in module.attributes['l1_end_bufs'].data]
    names = set().union(*(e.syms() for e in ends), *(s.extent.data.syms() for s in splits))
    margin = evaluate('L1_MARGIN', macros, {})
    if margin is None:
        raise MgcError(None, f'L1_MARGIN ({macros["L1_MARGIN"]}) is not a compile-time integer')
    budget = l1_bytes - margin
    for t in tiles:
        mesh = {'MESH_X_TILES': t, 'MESH_Y_TILES': t}
        m = {k: Const(v) for k, v in ((k, evaluate(k, macros, mesh)) for k in names | set(mesh)) if v is not None}
        split_of = {}  # tile-split maximum -> the names its extent needs
        for s in splits:
            ext = s.extent.data.subst(m).value()
            if ext is not None:
                m[s.sym.data + '_max'] = Const((ext + t - 1) // t)
            split_of[s.sym.data + '_max'] = s.extent.data.syms()
        vals = [e.subst(m).value() for e in ends]
        unknown = sorted({b for b, v in zip(bufs, vals) if v is None})
        used = max((v for v in vals if v is not None), default=0)
        if used > budget:
            raise MgcError(None, f'the L1 buffers of a tile take {"at least " if unknown else ""}{used} bytes on '
                                 f'a {t}x{t} mesh, more than the {budget} bytes of L1 ({l1_bytes} minus the '
                                 f'L1_MARGIN of {margin})')
        if unknown:
            need = set().union(*(e.syms() for e, v in zip(ends, vals) if v is None)) - set(m)
            need = set().union(*(split_of.get(x, {x}) for x in need)) - set(m)
            missing = sorted(set().union(*(undefined(x, macros, mesh) or {x} for x in need)))
            warnings.warn(MgcWarning(f'L1 fit on a {t}x{t} mesh checked without {", ".join(f"`{b}`" for b in unknown)}, '
                                     f'whose size depends on {", ".join(missing)}: not known at compile time'))


def compile_module(src, filename, header=None, until=None, dump=None, tiles=(), l1_bytes=None):
    """Run the front-end and the passes (up to and including `until`).
    `dump(name, text)` is called with the IR after each stage.

    Parameters: `src` is the .mgc text, `filename` is used in error messages,
    `header` is the test header path (optional), `until` is a stage name from
    `PASS_NAMES` at which to stop (default: run everything), `dump` an optional
    callback receiving `(stage_name, ir_text)`, `tiles` the mesh sides on which the
    L1 footprint is checked (see `check_l1`; none by default) and `l1_bytes` the L1 size
    (None: read from the SDK).

    Returns the IR module after the last executed stage. The module is verified
    after each stage. Raises `MgcError` for invalid programs.

    Example: `compile_module(src, 'a.mgc', until='pipeline')` returns the IR
    with the software pipelines already expanded into explicit guarded steps.
    """
    ctx = Context()
    ctx.load_dialect(MG)
    module = parse(src, filename, header)
    module.verify()
    if dump:
        dump('frontend', print_ir(module))
    if until == 'frontend':
        return module
    for p in PASSES:
        p.apply(ctx, module)
        module.verify()
        if dump:
            dump(p.name, print_ir(module))
        if until == p.name:
            break
    check_l1(module, header, tiles, l1_bytes)
    return module


def compile_source(src, filename, header=None, tiles=(), l1_bytes=None) -> str:
    """Full compilation: `.mgc` text in, C source text out (unformatted).

    Same arguments as `compile_module`. This is what the tests use; `mgcc.py`
    additionally runs clang-format on the result.
    """
    return emit(compile_module(src, filename, header, tiles=tiles, l1_bytes=l1_bytes), filename)
