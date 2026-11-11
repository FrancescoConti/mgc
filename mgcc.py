#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""mgcc - compile an mgc ("magic") test description into mglib C.

    mgcc.py kernels/mgc_kernels/mm_os/mm_os.mgc --header tests/magia/mgc/mm_os_mgc/include/test.h \
        -o kernels/mgc_kernels/mm_os/src/mgc_mm_os.c --test tests/magia/mgc/mm_os_mgc/src/test.c   # a @kernel
    mgcc.py tests/data/mm_os_l1k/mm_os_l1k.mgc -o test.c                                 # a @test
    mgcc.py mm_os.mgc --print-ir-after=pipeline     # inspect the IR after a pass
    mgcc.py mm_os.mgc --print-ir-after-all

Needs xDSL: python3 -m venv mgc/.venv && mgc/.venv/bin/pip install -r mgc/requirements.txt
"""

import argparse
import os
import shutil
import subprocess
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mgc.driver import PASS_NAMES, compile_module  # noqa: E402
from mgc import ir  # noqa: E402
from mgc.emit_c import emit, emit_kernel  # noqa: E402
from mgc.errors import MgcError, MgcWarning  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _func(module):
    return next(o for o in module.ops if isinstance(o, ir.FuncOp))


def is_kernel(module):
    return _func(module).kind.data == 'kernel'


def kernel_name(module):
    return _func(module).fname.data


def find_clang_format():
    """Path of an executable `clang-format` (from `PATH`, else the repo's
    `llvm/install/bin`), or None if not found (output is then left unformatted)."""
    for p in (shutil.which('clang-format'), os.path.join(REPO, 'llvm', 'install', 'bin', 'clang-format')):
        if p and os.access(p, os.X_OK):
            return p
    return None


def main():
    """Command-line entry point: compile `source.mgc` to a C test.

    Reads the `.mgc` file, compiles it (see `mgc.driver`), formats it with
    clang-format when available and writes it to `-o FILE` or stdout. With
    `--print-ir-after=STAGE` it prints the IR at that stage and stops instead.
    Returns the process exit code: 0 on success, 1 after printing a compile error.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='.mgc file')
    ap.add_argument('-o', '--output', help='output C file (default: stdout)')
    ap.add_argument('--header', help='test header with sizes and L2 data '
                    '(default: include/test.h next to the source)')
    ap.add_argument('--tiles', nargs='+', type=int, default=[], metavar='N',
                    help='mesh sides (make build tiles=N) on which to check that the L1 buffers of a tile fit')
    ap.add_argument('--l1-bytes', type=lambda x: int(x, 0), metavar='BYTES',
                    help='L1 size of a tile for --tiles (default: L1_SIZE + 1 from the SDK\'s '
                         'tile_addr_map.h; an error if it cannot be found)')
    ap.add_argument('--print-ir-after', choices=PASS_NAMES, help='print the IR after this stage and stop')
    ap.add_argument('--print-ir-after-all', action='store_true', help='print the IR after every stage to stderr')
    ap.add_argument('--no-format', action='store_true', help='do not run clang-format on the output')
    ap.add_argument('--kernel-header', metavar='FILE', help='@kernel: output header (default: include/<name>.h '
                    'next to the src/ of -o)')
    ap.add_argument('--test', metavar='FILE', help='@kernel: output C test calling the kernel')
    args = ap.parse_args()
    # compile warnings as `file.mgc: warning: msg`, each time they are issued
    warnings.simplefilter('always', MgcWarning)
    show = warnings.showwarning
    warnings.showwarning = lambda msg, cat, *a, **k: (print(msg, file=sys.stderr) if issubclass(cat, MgcWarning)
                                                      else show(msg, cat, *a, **k))

    with open(args.source) as f:
        src = f.read()
    header = args.header or os.path.join(os.path.dirname(os.path.abspath(args.source)), 'include', 'test.h')
    try:
        if args.print_ir_after:
            def dump(name, text):
                if name == args.print_ir_after:
                    sys.stdout.write(text)
            compile_module(src, args.source, header, until=args.print_ir_after, dump=dump, tiles=args.tiles,
                           l1_bytes=args.l1_bytes)
            return 0
        dump = (lambda name, text: sys.stderr.write(f'// ----- IR after {name}\n{text}')) \
            if args.print_ir_after_all else None
        module = compile_module(src, args.source, header, dump=dump, tiles=args.tiles, l1_bytes=args.l1_bytes)
        if is_kernel(module):
            if not args.output:
                raise MgcError(None, 'a @kernel needs -o FILE (the kernel C file)')
            c, h, t = emit_kernel(module, args.source)
        else:
            c, h, t = emit(module, args.source), None, None
    except MgcError as e:
        print(e, file=sys.stderr)
        return 1

    cf = None if args.no_format else find_clang_format()

    def fmt(text, name):
        if not cf:
            return text
        # always the repo style, wherever the output goes
        style = os.path.join(REPO, '.clang-format')
        return subprocess.run([cf, f'--style=file:{style}', f'--assume-filename={name}'], input=text,
                              capture_output=True, text=True, check=True).stdout

    def write(path, text):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            f.write(text)

    if h is not None:
        out_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.output)))
        write(args.kernel_header or os.path.join(out_dir, 'include', kernel_name(module) + '.h'), fmt(h, 'k.h'))
        if args.test:
            write(args.test, fmt(t, 'test.c'))
    c = fmt(c, 'test.c')
    if args.output:
        write(args.output, c)
    else:
        sys.stdout.write(c)
    return 0


if __name__ == '__main__':
    sys.exit(main())
