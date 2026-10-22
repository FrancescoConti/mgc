#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""mgc_sweep - run the tests in tests/magia/mgc over a sweep of configurations on gvsoc and
compare cycles, MAC/cycle and the utilization of RedMulE, iDMA and Spatz.

The runs are ordered by mesh side; gvsoc is rebuilt for each mesh side with
    make gvsoc tiles=<N> pulp_cluster=0
and each configuration is built, one at a time, and run with
    make build tiles=<N> test=<test> pulp_cluster=0
    make run_profiling platform=gvsoc tiles=<N> test=<test> pulp_cluster=0
and measured on gvsoc_work/trace.perfetto-trace. With --jobs J, up to J runs of the same mesh
side simulate in parallel, each with its own CMAKE_BUILDDIR and GVSOC_WORK_DIR (job 0: build/
and gvsoc_work/, the others in <out>/jobs/<k>/); builds stay serial, as they share the test
sources. Metrics:
  - cycles: from tile 0's entry in main to the exit of the last tile other than tile 0, i.e.
    without crt0 init and without tile 0's final check(); for mm tests, where each tile checks
    its own output tile, to the end of the last iDMA store (L1->L2)
  - MAC/cycle: matmul MACs only / cycles: M*N*K*N_ITERATIONS for mm, 2*S*S*P for attention
    (Q @ Kt and Attn @ V of one head, softmax counts 0), 2*S*S*P*H + 4*S*D*P*H for mhsa (all heads,
    plus the Q, Kt, V and output projections), 2*S*S*P*H + 2*S*D*P*H + 2*S*D*P*HKV for gqa
  - per-tile utilization in that window, averaged over all tiles: RedMulE busy, iDMA backend
    busy (L2->L1, L1->L2, either), Spatz running a task or its VFU active, and empty cycles
    (none of them busy: CV32 alone, or CV32 asleep waiting for an event)

Parameters (S, D, P, H, HKV, M, N, K from include/test.h, S_BLOCK_SIZE_Q/KV, T_SIZE from the .mgc)
default to the test's own (mm tests: M, N, K and T_SIZE, settable for those with a gen_golden.py); any other value regenerates test.h (golden generators) and src/test.c (mgcc) in
place. The test sources are restored at the end (also on Ctrl-C, or on the next start if the
script was killed). D and H are written to test.h but not used by the current attention kernels.

Sweep: a JSON file with a list of groups, each the cartesian product of its lists (missing
parameters take each test's default), e.g.
    {"groups": [
      {"tests": ["unfused_attn", "flat_attn_pipeopt3"], "tiles": [2, 4]},
      {"tests": ["flat_attn_pipeopt3"], "tiles": [4], "S_BLOCK_SIZE_KV": [256, 512]}
    ]}
or a single group from the command line:
    mgc/scripts/mgc_sweep.py --tests unfused_attn flat_attn_pipeopt3 --tiles 2 4 [--S 512 1024 ...]
    mgc/scripts/mgc_sweep.py --sweep sweep.json [--out DIR] [--jobs J] [--dry-run] [--keep-traces]
                             [--no-profile] [--timeout SEC] [--l2-size BYTES]
Hardware: spatz_vlen (bits) and redmule ('HxW') sweep the gvsoc Spatz VLEN and RedMulE array
(make spatz_vlen=... redmule_h=... redmule_w=...; default: gvsoc's, 256 and 8x8), e.g.
    mgc/scripts/mgc_sweep.py --tests flat_attn_pipeopt3 --spatz-vlen 256 512 --redmule 8x8 16x16
gvsoc is rebuilt for each (mesh side, hardware).
--no-profile still runs make run_profiling, but gvsoc2perfetto reduces the trace to 5 bits per tile
(CV32, Spatz, RedMulE, iDMA in and out busy) plus tile 0's entry in main: VFU% is N/A, and each run's
(small) perfetto trace is kept in runs/<id>/. --timeout kills a simulation that does not end
(status TIMEOUT). --l2-size builds and runs gvsoc with a larger L2 (make l2_size=...), needed by tests whose
golden generator places the L2 arrays in .l2ext (see targets/magia/link.ld).
    mgc/scripts/mgc_sweep.py --list

Output in --out: summary.log (human-readable table), results.csv (one row per run, plain
comma-separated numbers, empty for N/A: Excel, or numpy.genfromtxt(f, delimiter=',',
names=True, dtype=None, encoding='utf-8')), gvsoc_tiles<N>[_<hw>].log (gvsoc builds), runs/<id>/ (make
log, per-tile metrics).

Needs the `perfetto` Python package (pip install perfetto; on hosts with glibc < 2.34 its
trace_processor_shell does not run, and the sweep fetches perfetto v47.0's instead, or give one
with --trace-processor); the golden generators need torch
(--gen-python), mgcc needs xDSL (mgc/.venv, see mgc/README.md).
Run it from anywhere: the magia-sdk checkout is $MAGIA_SDK, or the parent of this mgc checkout.
"""

import argparse
import csv
import datetime
import glob
import itertools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue

import numpy as np

MGC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# magia-sdk checkout: $MAGIA_SDK, or the parent of mgc (a submodule of magia-sdk)
REPO = os.path.abspath(os.environ.get('MAGIA_SDK', os.path.join(MGC, '..')))
MGC_TESTS = os.path.join(REPO, 'tests', 'magia', 'mgc')
FLAT_GEN = os.path.join(REPO, 'tests', 'magia', 'mesh', 'mm_generator', 'flat_attn.py')
SWEEP_DIR = os.path.join(REPO, 'mgc_sweep')
BACKUP = os.path.join(SWEEP_DIR, '.backup')
CYCLE_NS = 5  # gvsoc clock: 200 MHz, perfetto timestamps in ns

PARAMS = ['S', 'D', 'P', 'H', 'HKV', 'S_BLOCK_SIZE_Q', 'S_BLOCK_SIZE_KV', 'M', 'N', 'K', 'T_SIZE']
HEADER_DEFINES = {'S': 'SEQ_LENGTH', 'D': 'EMBED_SIZE', 'P': 'PROJ_SIZE', 'H': 'NUM_HEADS', 'HKV': 'NUM_KV_HEADS',
                  'M': 'M_SIZE', 'N': 'N_SIZE', 'K': 'K_SIZE'}
MGC_DEFINES = ['S_BLOCK_SIZE_Q', 'S_BLOCK_SIZE_KV', 'T_SIZE']
# gvsoc hardware knobs (None: the gvsoc default): Spatz VLEN in bits, RedMulE array as 'HxW'
HW_PARAMS = ['spatz_vlen', 'redmule']
# Only the gvsoc events analyze() reads (make's defaults dump ~20x more: asm, NoC, L2, ...),
# as POSIX basic regexes for gvsoc and as a regex of the gvsoc2perfetto tracks they become
# (CV32 PC of tile 0 only, for its entry in main: one change per cycle of a spinning core)
PROFILE_EVENTS = ["'/tile-0-cv32-core/active_pc$$'"] + ["'/tile-[0-9]*-%s$$'" % e for e in (
    'cv32-core/busy', 'redmule/busy', 'idma-[a-z0-9]*/be/busy', 'snitch-spatz/active_pc',
    'snitch-spatz/ara/active')]
PERFETTO_INCLUDE = (r'(?x) tile-\d+-cv32-core\.(busy|active_pc)$$ | tile-\d+-redmule\.busy$$'
                    r' | tile-\d+-idma-(axi2obi|obi2axi)\.be\.busy$$'
                    r' | tile-\d+-snitch-spatz\.(active_pc|ara\.active)$$')
# --no-profile: the same events, reduced by gvsoc2perfetto to tile 0's main marker (cv32-core.main)
# and one busy bit per unit; Spatz busy (spatz_busy) = Snitch PC in a task | VFU active
LIGHT_KEEP = (r'(?x) cv32-core\.(busy|main)$$ | redmule\.busy$$ | idma-(axi2obi|obi2axi)\.be\.busy$$'
              r' | snitch-spatz\.spatz_busy$$')
# perfetto's prebuilt trace_processor_shell needs glibc >= 2.34 since v48: on older hosts (e.g.
# RHEL 8, glibc 2.28) the sweep fetches v47.0, the last release built against glibc 2.23
TP_MIN_GLIBC = (2, 34)
TP_LEGACY_URL = ('https://commondatastorage.googleapis.com/perfetto-luci-artifacts/v47.0/linux-amd64/'
                 'trace_processor_shell')
TP_LEGACY_BIN = os.path.join(os.path.expanduser('~'), '.local', 'share', 'perfetto', 'prebuilts',
                             'trace_processor_shell-v47.0')
# Spatz runtime outside of any task (boot, dispatcher, interrupt handling)
SPATZ_IDLE_FUNCS = {'_start', 'dispatcher_loop', '_trap_handler', '_handle_exception',
                    '_handle_interrupt', '_trap_ret'}

CSV_COLUMNS = [
    'run_id', 'test', 'tiles', 'n_tiles', *HW_PARAMS, *PARAMS, 'status', 'errors', 'cycles', 'macs',
    'mac_per_cycle', 'mac_per_cycle_per_tile', 'redmule_pct', 'idma_pct', 'idma_in_pct',
    'idma_out_pct', 'spatz_pct', 'spatz_vfu_pct', 'empty_pct', 'empty_cv32_active_pct',
    'empty_cv32_sleep_pct', 'main_start_cycle', 'end_cycle', 'sim_seconds']

# (header, csv column, format) of the human-readable table
TABLE = [
    ('id', 'run_id', '{}'), ('test', 'test', '{}'), ('tiles', 'tiles', '{}'), ('VLEN', 'spatz_vlen', '{}'),
    ('RedMulE', 'redmule', '{}'), ('S', 'S', '{}'),
    ('D', 'D', '{}'), ('P', 'P', '{}'), ('H', 'H', '{}'), ('HKV', 'HKV', '{}'), ('BQ', 'S_BLOCK_SIZE_Q', '{}'),
    ('BKV', 'S_BLOCK_SIZE_KV', '{}'), ('M', 'M', '{}'), ('N', 'N', '{}'), ('K', 'K', '{}'),
    ('T', 'T_SIZE', '{}'), ('status', 'status', '{}'), ('cycles', 'cycles', '{:,}'),
    ('MAC/cyc', 'mac_per_cycle', '{:.1f}'), ('MAC/cyc/tile', 'mac_per_cycle_per_tile', '{:.2f}'),
    ('RedMulE%', 'redmule_pct', '{:.1f}'), ('iDMA%', 'idma_pct', '{:.1f}'),
    ('in%', 'idma_in_pct', '{:.1f}'), ('out%', 'idma_out_pct', '{:.1f}'),
    ('Spatz%', 'spatz_pct', '{:.1f}'), ('VFU%', 'spatz_vfu_pct', '{:.1f}'),
    ('empty%', 'empty_pct', '{:.1f}'), ('CV32 run%', 'empty_cv32_active_pct', '{:.1f}'),
    ('CV32 sleep%', 'empty_cv32_sleep_pct', '{:.1f}')]


# --------------------------------------------------------------------------------------------
# Tests and their parameters
# --------------------------------------------------------------------------------------------

def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    with open(path, 'w') as f:
        f.write(text)


class Test:
    """One test directory of tests/magia/mgc (built by tests/magia/mgc/CMakeLists.txt)."""

    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(MGC_TESTS, name)
        cmake = read(os.path.join(self.dir, 'CMakeLists.txt'))
        self.target = re.search(r'set\(TEST_NAME\s+(\S+)\)', cmake).group(1)
        m = re.search(r'TEST_NAME\s+(\w+_spatz)\b', cmake)
        self.spatz_name = m.group(1) if m else None
        self.header = os.path.join(self.dir, 'include', 'test.h')
        self.test_c = os.path.join(self.dir, 'src', 'test.c')
        self.gen = os.path.join(self.dir, 'gen_golden.py')
        self.mgc = os.path.join(self.dir, name + '.mgc')
        if not os.path.exists(self.mgc):
            self.mgc = sorted(glob.glob(os.path.join(self.dir, '*.mgc')))[0]
        if name.startswith('flat_attn'):
            self.kind = 'flat_attn'
        elif name.startswith('mhsa'):
            self.kind = 'mhsa'
        elif name.startswith('gqa'):
            self.kind = 'gqa'
        elif name == 'unfused_attn':
            self.kind = 'unfused_attn'
        else:
            self.kind = 'mm'
        self.defaults = self._defaults()
        self.goldens = [int(t) for t in re.findall(r'^#(?:el)?if MESH_X_TILES == (\d+)', read(self.header), re.M)]

    def _defaults(self):
        h, src = read(self.header), read(self.mgc)
        d = {}
        for p, macro in HEADER_DEFINES.items():
            m = re.search(r'#define\s+%s\s+\(?(\d+)\)?' % macro, h)
            d[p] = int(m.group(1)) if m else None
        for p in MGC_DEFINES:
            m = re.search(r'^%s\s*=\s*define\((\d+)\)' % p, src, re.M)
            d[p] = int(m.group(1)) if m else None
        if self.kind == 'mm':
            m = re.search(r'^N_ITERATIONS\s*=\s*define\((\d+)\)', src, re.M)
            self.n_iterations = int(m.group(1)) if m else 1
        return d

    def applicable(self):
        """Parameters this test has (the others are N/A)."""
        return [p for p in PARAMS if self.defaults[p] is not None]

    def macs(self, cfg):
        if self.kind == 'mm':
            return cfg['M'] * cfg['N'] * cfg['K'] * self.n_iterations
        # Q @ Kt and Attn @ V, S*S*P each; flat_attn computes a single head (H is not used)
        if self.kind not in ('mhsa', 'gqa'):
            return 2 * cfg['S'] * cfg['S'] * cfg['P']
        # mhsa: + Q, Kt, V and output projections, S*D*(H*P) each; gqa: Kt and V on HKV heads only
        S, D, P, H = cfg['S'], cfg['D'], cfg['P'], cfg['H']
        HKV = cfg['HKV'] if self.kind == 'gqa' else H
        return 2 * S * S * P * H + 2 * S * D * P * H + 2 * S * D * P * HKV

    def check(self, cfg, tiles):
        """Reason why (cfg, tiles) cannot run, or None."""
        if self.kind == 'mm' and not os.path.exists(self.gen) and any(
                cfg[p] != self.defaults[p] for p in self.applicable()):
            return 'mm tests without a gen_golden.py run with their default sizes only'
        if self.kind == 'mm':
            if cfg['M'] % tiles or cfg['K'] % tiles:
                return 'M and K must be multiples of tiles'
            if cfg['T_SIZE'] and (cfg['N'] % cfg['T_SIZE'] or cfg['N'] // cfg['T_SIZE'] < 2):
                return 'N must be a multiple of T_SIZE, at least 2 timeslots'
        if self.kind == 'gqa' and cfg['H'] % cfg['HKV']:
            return 'H must be a multiple of HKV'
        if self.kind in ('flat_attn', 'mhsa', 'gqa'):
            S, bq, bkv = cfg['S'], cfg['S_BLOCK_SIZE_Q'], cfg['S_BLOCK_SIZE_KV']
            if tiles & (tiles - 1):
                return 'tiles must be a power of 2'
            if S % bq or S % bkv:
                return 'S must be a multiple of S_BLOCK_SIZE_Q and S_BLOCK_SIZE_KV'
            if bq % tiles or bkv % tiles:
                return 'S_BLOCK_SIZE_Q and S_BLOCK_SIZE_KV must be multiples of tiles'
            if S // bq > 255 or S // bkv > 255:
                return 'more than 255 blocks'
        if self.kind == 'unfused_attn':
            # softmax: S / tiles^2 rows per tile, in chunks of SM_CHUNK = min(32, rows)
            S, P, n = cfg['S'], cfg['P'], tiles * tiles
            if S % tiles or P % tiles:
                return 'S and P must be multiples of tiles'
            if S % n or (S // n > 32 and S // n % 32):
                return 'S / tiles^2 (softmax rows per tile) must be an integer, <= 32 or a multiple of 32'
        return None

    def needs_regen(self, cfg, tiles):
        changed = any(cfg[p] != self.defaults[p] for p in self.applicable())
        return changed or (self.goldens and tiles not in self.goldens)


def list_tests():
    cm = read(os.path.join(MGC_TESTS, 'CMakeLists.txt'))
    return re.findall(r'^add_subdirectory\((\w+)\)', cm, re.M)


# --------------------------------------------------------------------------------------------
# In-place regeneration of test.h / test.c, with backup and restore
# --------------------------------------------------------------------------------------------

class Sources:
    """Modifies a test's sources in place; restore() puts back the originals."""

    def __init__(self, log):
        self.log = log
        self.saved = {}  # original path -> backup path

    def save(self, path):
        if path in self.saved:
            return
        os.makedirs(BACKUP, exist_ok=True)
        bak = os.path.join(BACKUP, '%d_%s' % (len(self.saved), os.path.basename(path)))
        shutil.copy2(path, bak)
        self.saved[path] = bak
        write(os.path.join(BACKUP, 'manifest.json'), json.dumps(self.saved, indent=1))

    def restore(self):
        for path, bak in self.saved.items():
            shutil.copy2(bak, path)
        if self.saved:
            self.log('restored %d test source file(s)' % len(self.saved))
        self.saved = {}
        shutil.rmtree(BACKUP, ignore_errors=True)

    @staticmethod
    def recover(log):
        """Restore the sources left modified by a killed sweep."""
        man = os.path.join(BACKUP, 'manifest.json')
        if os.path.exists(man):
            for path, bak in json.loads(read(man)).items():
                shutil.copy2(bak, path)
                log('recovered %s from a previous interrupted sweep' % os.path.relpath(path, REPO))
        shutil.rmtree(BACKUP, ignore_errors=True)


def generate(test, cfg, tiles, args, srcs, log):
    """Write test.h, the .mgc defines and test.c for cfg; return None or an error string."""
    for p in (test.header, test.test_c, test.mgc):
        srcs.save(p)
    # restart from the originals: a previous configuration may have changed them
    for p in (test.header, test.test_c, test.mgc):
        shutil.copy2(srcs.saved[p], p)

    if test.kind == 'mm':
        key = '%s_M%s_N%s_K%s' % (test.name, cfg['M'], cfg['N'], cfg['K'])
    else:
        # the flat_attn tests share one generator; mhsa/gqa tests have their own (e.g. the _masked ones)
        key = '%s_S%s_D%s_P%s_H%s%s_KV%s_T%d' % (test.kind if test.kind == 'flat_attn' else test.name,
                                                cfg['S'], cfg['D'], cfg['P'], cfg['H'],
                                                '_HKV%s' % cfg['HKV'] if test.kind == 'gqa' else '',
                                                cfg['S_BLOCK_SIZE_KV'], tiles)
    cache = os.path.join(args.cache, key)
    cached = os.path.join(cache, 'test.h')
    if not os.path.exists(cached):
        log('  generating test.h (%s)...' % key)
        os.makedirs(cache, exist_ok=True)
        env = dict(os.environ)
        if test.kind in ('flat_attn', 'mhsa', 'gqa'):
            gen = FLAT_GEN if test.kind == 'flat_attn' else os.path.join(test.dir, 'gen_golden.py')
            cmd = [args.gen_python, gen, '--s_size', cfg['S'], '--e_size', cfg['D'],
                   '--p_size', cfg['P'], '--n_heads', cfg['H'], '--block_kv', cfg['S_BLOCK_SIZE_KV'],
                   '--tiles', tiles, '--inc_dir', cache]
            if test.kind == 'gqa':
                cmd += ['--n_kv_heads', cfg['HKV']]
        elif test.kind == 'mm':
            cmd = [args.gen_python, test.gen, '--m', cfg['M'], '--n', cfg['N'], '--k', cfg['K'], '--inc_dir', cache]
        else:
            # writes the test's own include/test.h (restored at the end)
            cmd = [args.gen_python, os.path.join(test.dir, 'gen_golden.py'),
                   '--s_size', cfg['S'], '--p_size', cfg['P']]
            env['PYTHONPATH'] = os.path.dirname(FLAT_GEN)
        r = subprocess.run([str(c) for c in cmd], cwd=REPO, env=env, capture_output=True, text=True)
        write(os.path.join(cache, 'gen.log'), r.stdout + r.stderr)
        if r.returncode:
            shutil.rmtree(cache, ignore_errors=True)
            return 'golden generation failed: ' + (r.stderr.strip().splitlines() or ['?'])[-1]
        if test.kind == 'unfused_attn':
            shutil.copy2(test.header, cached)
    shutil.copy2(cached, test.header)

    src = read(test.mgc)
    for p in MGC_DEFINES:
        if cfg[p] is not None:
            src = re.sub(r'^(%s\s*=\s*define\()\d+(\))' % p, r'\g<1>%d\2' % cfg[p], src, flags=re.M)
    write(test.mgc, src)
    r = subprocess.run([args.mgcc_python, os.path.join(MGC, 'mgcc.py'), test.mgc, '-o', test.test_c],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        return 'mgcc failed: ' + ' '.join(r.stderr.split())[:200]
    return None


# --------------------------------------------------------------------------------------------
# Build + run
# --------------------------------------------------------------------------------------------

PROCS = set()  # running make process groups, killed on interrupt
STOP = threading.Event()


def make(cmd, timeout=None, env=None):
    """Run cmd in its own process group, with env added to the environment; return (return
    code, output), return code None if killed after timeout seconds."""
    env_setup = 'source sourceme.sh && ' if os.path.exists(os.path.join(REPO, 'sourceme.sh')) else ''
    p = subprocess.Popen(['bash', '-c', env_setup + cmd], cwd=REPO, env={**os.environ, **(env or {})},
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors='replace', start_new_session=True)
    PROCS.add(p)
    try:
        out = p.communicate(timeout=timeout)[0]
        return p.returncode, out
    except subprocess.TimeoutExpired:
        kill(p)
        return None, p.communicate()[0]
    except BaseException:
        kill(p)
        raise
    finally:
        PROCS.discard(p)


def kill(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def slot_dirs(out, slot):
    """(CMAKE_BUILDDIR, GVSOC_WORK_DIR) of a parallel job slot."""
    if slot == 0:
        return os.path.join(REPO, 'build'), os.path.join(REPO, 'gvsoc_work')
    d = os.path.join(out, 'jobs', str(slot))
    return os.path.join(d, 'build'), os.path.join(d, 'gvsoc_work')


def gvsoc_opts(args, hw):
    """make options for the gvsoc target: L2 size (--l2-size) and hardware knobs, or ''."""
    opts = ' l2_size=%s' % args.l2_size if args.l2_size else ''
    if hw['spatz_vlen']:
        opts += ' spatz_vlen=%d' % hw['spatz_vlen']
    if hw['redmule']:
        opts += ' redmule_h=%d redmule_w=%d' % redmule_hw(hw['redmule'])
    return opts


def redmule_hw(s):
    """(H, W) of a RedMulE array given as 'HxW'."""
    h, w = s.lower().split('x')
    return int(h), int(w)


def hw_desc(hw):
    return ''.join(' %s=%s' % (p, hw[p]) for p in HW_PARAMS if hw[p])


def build_gvsoc(tiles, hw, logf, args):
    """gvsoc is built for one mesh side and hardware: rebuild it whenever they change."""
    # gvsoc2perfetto too, so that parallel run_profiling do not race to build it
    cmd = 'make gvsoc tiles=%d pulp_cluster=0%s && make gvsoc2perfetto' % (tiles, gvsoc_opts(args, hw))
    rc, out = make(cmd)
    write(logf, '$ ' + cmd + '\n' + out)
    return rc


def build_test(test, tiles, build, logf):
    shutil.rmtree(build, ignore_errors=True)
    cmd = 'make build tiles=%d test=%s pulp_cluster=0 CMAKE_BUILDDIR=%s' % (tiles, test.target, build)
    rc, out = make(cmd)
    write(logf, '$ ' + cmd + '\n' + out)
    return rc


def run_test(test, tiles, hw, build, work, logf, args):
    cmd = ('make run_profiling platform=gvsoc tiles=%d test=%s pulp_cluster=0 CMAKE_BUILDDIR=%s GVSOC_WORK_DIR=%s%s'
           % (tiles, test.target, build, work, gvsoc_opts(args, hw)))
    env = {}
    if args.no_profile:  # through the environment: the Makefile's ?= defaults give way
        main = symbols(test, build)['main']
        extra = ['--pc-in', r'tile-0-cv32-core\.active_pc$$=main:%x-%x' % (main, main + 1)]
        if test.spatz_name:
            extra += ['--pc-in', r'snitch-spatz\.active_pc$$=task:' + ','.join(
                '%x-%x' % r for r in spatz_task_pcs(test, build))]
        extra += ['--or', 'spatz_busy=task,ara.active', '--keep', LIGHT_KEEP]
        env = {'profile_events': ' '.join(PROFILE_EVENTS), 'GVSOC2PERFETTO_INCLUDE': PERFETTO_INCLUDE,
               'GVSOC2PERFETTO_EXTRA': ' '.join(map(shlex.quote, extra))}
    elif not args.keep_traces:
        cmd += ' ' + shlex.quote('profile_events=' + ' '.join(PROFILE_EVENTS))
        cmd += ' ' + shlex.quote('GVSOC2PERFETTO_INCLUDE=' + PERFETTO_INCLUDE)
    rc, out = make(cmd, args.timeout, env)
    with open(logf, 'a') as f:
        f.write(''.join('$ export %s=%s\n' % (k, shlex.quote(v)) for k, v in env.items()))
        f.write('$ ' + cmd + '\n' + out)
    return rc, out


def parse_run(out):
    kills = {int(m.group(2)): int(m.group(1)) for m in re.finditer(
        r'^\d+: (\d+): .*kill-module.*Received kill request from port (\d+)', out, re.M)}
    errs = [int(e) for e in re.findall(r'TILE \d+: Number of errors: (\d+)', out)]
    return kills, errs


# --------------------------------------------------------------------------------------------
# Trace analysis
# --------------------------------------------------------------------------------------------

def merge(iv):
    """Union of [start, end) intervals, as a sorted disjoint (n, 2) array."""
    iv = np.asarray(iv, dtype=np.int64).reshape(-1, 2)
    if not len(iv):
        return iv
    iv = iv[np.argsort(iv[:, 0])]
    ends = np.maximum.accumulate(iv[:, 1])
    new = np.r_[True, iv[1:, 0] > ends[:-1]]
    starts = iv[new, 0]
    stop = np.r_[np.flatnonzero(new)[1:] - 1, len(iv) - 1]
    return np.stack([starts, ends[stop]], axis=1)


def clip(iv, lo, hi):
    iv = np.clip(np.asarray(iv, dtype=np.int64).reshape(-1, 2), lo, hi)
    return iv[iv[:, 1] > iv[:, 0]]


def length(iv):
    return int((iv[:, 1] - iv[:, 0]).sum()) if len(iv) else 0


def overlap(a, b):
    """Length of the intersection of two merged interval sets."""
    i = j = t = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        t += max(0, hi - lo)
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return int(t)


def symbols(test, build):
    """Symbol -> address of the test's binary."""
    m = re.search(r'^CMAKE_NM:FILEPATH=(.+)$', read(os.path.join(build, 'CMakeCache.txt')), re.M)
    nm = m.group(1) if m else 'riscv64-unknown-elf-nm'
    syms = subprocess.run([nm, os.path.join(build, 'bin', test.target)], capture_output=True,
                          text=True).stdout
    return {m.group(2): int(m.group(1), 16) for m in re.finditer(r'^([0-9a-f]+) \w (\w+)$', syms, re.M)}


def spatz_idle_pcs(test, build):
    """Absolute [lo, hi) PC ranges of the Spatz runtime outside tasks, and the binary's range."""
    addr = symbols(test, build)
    base, end = addr['_spatz_binary_start'], addr['_spatz_binary_end']
    dump = glob.glob(os.path.join(build, 'bin', 'magia', test.spatz_name, 'spatz_task', '*.dump'))[0]
    funcs, in_text = [], False
    for line in open(dump):
        if line.startswith('Disassembly of section'):
            in_text = line.strip().endswith('.text:')
        m = re.match(r'^([0-9a-f]+) <([^>]+)>:', line)
        if in_text and m and not m.group(2).startswith('.L'):
            funcs.append((int(m.group(1), 16), m.group(2)))
    funcs.append((end - base, None))
    idle = [(base + a, base + b) for (a, f), (b, _) in zip(funcs, funcs[1:]) if f in SPATZ_IDLE_FUNCS]
    return idle, (base, end)


def spatz_task_pcs(test, build):
    """[lo, hi) PC ranges of the Spatz binary outside the runtime's idle code."""
    idle, (lo, end) = spatz_idle_pcs(test, build)
    busy = []
    for a, b in sorted(idle):
        if a > lo:
            busy.append((lo, a))
        lo = max(lo, b)
    return busy + [(lo, end)] if lo < end else busy


def trace_processor_bin(path, log):
    """trace_processor_shell for the perfetto package: path if given, None (perfetto's own
    prebuilt) on hosts with a recent enough glibc, else TP_LEGACY_BIN (downloaded once)."""
    if path:
        return path
    try:
        glibc = tuple(int(v) for v in os.confstr('CS_GNU_LIBC_VERSION').split()[1].split('.')[:2])
    except (ValueError, OSError, IndexError, AttributeError):
        return None
    if glibc >= TP_MIN_GLIBC:
        return None
    if not os.path.exists(TP_LEGACY_BIN):
        log('glibc %d.%d is too old for perfetto\'s trace_processor_shell, fetching %s' % (*glibc, TP_LEGACY_URL))
        os.makedirs(os.path.dirname(TP_LEGACY_BIN), exist_ok=True)
        tmp = TP_LEGACY_BIN + '.part'
        urllib.request.urlretrieve(TP_LEGACY_URL, tmp)
        os.chmod(tmp, 0o755)
        os.replace(tmp, TP_LEGACY_BIN)
    return TP_LEGACY_BIN


def analyze(test, trace, kills, n_tiles, build, light, tp_bin=None):
    from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
    tp = TraceProcessor(trace=trace, config=TraceProcessorConfig(bin_path=tp_bin, load_timeout=1800))
    try:
        tracks = {}  # (tile, track name) -> track id
        for r in tp.query("select t.id, t.name, p.name pname from track t join track p on t.parent_id = p.id "
                          "where p.name like '%magia-tile %'"):  # 'magia-v3-soc.magia-tile N' if only tile events
            tracks[(int(r.pname.split()[-1]), r.name)] = r.id

        def slices(tile, name):
            tid = tracks.get((tile, name))
            if tid is None:
                return np.zeros((0, 2), np.int64)
            q = tp.query('select ts, dur from slice where track_id = %d' % tid).as_pandas_dataframe()
            ts, dur = q.ts.to_numpy(np.int64), q.dur.to_numpy(np.int64)
            dur = np.where(dur < 0, np.iinfo(np.int64).max // 4, dur)  # unterminated: until the end
            return np.stack([ts, ts + dur], axis=1)

        main0 = (tp.query('select min(ts) t from slice where track_id = %d' % tracks[(0, 'cv32-core.main')])
                 if light else
                 tp.query("select min(ts) t from slice where track_id = %d and name = 'main'"
                          % tracks[(0, 'cv32-core.function')])).as_pandas_dataframe().t[0]
        others = [c for t, c in kills.items() if t != 0] or list(kills.values())
        lo, hi = int(main0), max(others) * CYCLE_NS
        if test.kind == 'mm':
            # every tile checks its own output tile after storing it: end at the last store
            stores = [slices(t, 'idma-obi2axi.be.busy') for t in range(n_tiles)]
            hi = min(hi, max(int(s[:, 1].max()) for s in stores if len(s)))

        # Spatz busy: Snitch PC outside the runtime's idle code (dispatcher, traps, boot)
        idle, (base, end) = spatz_idle_pcs(test, build) if test.spatz_name else ([], (0, 0))
        spatz_busy = {}
        pc_tracks = {t: tid for (t, n), tid in tracks.items() if n == 'snitch-spatz.pc' and not light}
        if pc_tracks:
            names = tp.query('select distinct name from slice where track_id in (%s)'
                             % ','.join(map(str, pc_tracks.values()))).as_pandas_dataframe().name
            idle_names = [n for n in names if not (base <= int(n, 16) < end)
                          or any(a <= int(n, 16) < b for a, b in idle)]
            idle_sql = ','.join("'%s'" % n for n in idle_names) or "''"
            # islands of consecutive busy PCs, one interval each
            q = tp.query(f"""
                with p as (select track_id, ts, dur, name not in ({idle_sql}) as b from slice
                           where track_id in ({','.join(map(str, pc_tracks.values()))})),
                     q as (select *, (b != lag(b, 1, -1) over (partition by track_id order by ts)) as chg from p),
                     r as (select *, sum(chg) over (partition by track_id order by ts rows unbounded preceding) as g from q)
                select track_id, min(ts) as a, max(ts + dur) as e from r where b group by track_id, g
            """).as_pandas_dataframe()
            inv = {tid: t for t, tid in pc_tracks.items()}
            for tid, grp in q.groupby('track_id'):
                spatz_busy[inv[tid]] = np.stack([grp.a.to_numpy(np.int64), grp.e.to_numpy(np.int64)], axis=1)

        per_tile = []
        for t in range(n_tiles):
            def get(name):
                return merge(clip(slices(t, name), lo, hi))
            rm = get('redmule.busy')
            din, dout = get('idma-axi2obi.be.busy'), get('idma-obi2axi.be.busy')
            vfu = get('snitch-spatz.vfu.active')
            sp = get('snitch-spatz.spatz_busy') if light else merge(np.r_[clip(spatz_busy.get(t, np.zeros((0, 2))), lo, hi), vfu])
            cv = get('cv32-core.busy')
            acc = merge(np.r_[rm, din, dout, sp])
            w = hi - lo
            empty = w - length(acc)
            cv_alone = length(cv) - overlap(cv, acc)
            per_tile.append(dict(redmule=length(rm) / w, idma=length(merge(np.r_[din, dout])) / w,
                                 idma_in=length(din) / w, idma_out=length(dout) / w, spatz=length(sp) / w,
                                 vfu=length(vfu) / w, empty=empty / w, cv32_active=cv_alone / w,
                                 cv32_sleep=(empty - cv_alone) / w))
        return lo // CYCLE_NS, hi // CYCLE_NS, per_tile
    finally:
        tp.close()


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------

def fmt_cell(fmt, v):
    if v is None or v == '' or (isinstance(v, float) and np.isnan(v)):
        return '-'
    return fmt.format(v)


def table(rows):
    cells = [[h for h, _, _ in TABLE]] + [[fmt_cell(f, r.get(c)) for _, c, f in TABLE] for r in rows]
    w = [max(len(row[i]) for row in cells) for i in range(len(TABLE))]
    left = {'test', 'status'}
    line = lambda row: '  '.join(c.ljust(w[i]) if TABLE[i][0] in left else c.rjust(w[i])
                                 for i, c in enumerate(row))
    return '\n'.join([line(cells[0]), '  '.join('-' * x for x in w)] + [line(r) for r in cells[1:]])


def csv_row(row):
    return ['' if row.get(c) is None else round(row[c], 4) if isinstance(row[c], float) else row[c]
            for c in CSV_COLUMNS]


class Report:
    """summary.log and results.csv, shared by the parallel jobs."""

    def __init__(self, out):
        self.out = out
        self.lock = threading.Lock()
        self.rows = []
        self.logf = open(os.path.join(out, 'summary.log'), 'a')
        self.csvf = os.path.join(out, 'results.csv')
        with open(self.csvf, 'w', newline='') as f:
            csv.writer(f, lineterminator='\n').writerow(CSV_COLUMNS)

    def log(self, msg=''):
        with self.lock:
            print(msg, flush=True)
            self.logf.write(msg + '\n')
            self.logf.flush()

    def add(self, row):
        with self.lock:
            self.rows.append(row)
            with open(self.csvf, 'a', newline='') as f:
                csv.writer(f, lineterminator='\n').writerow(csv_row(row))

    def sorted_rows(self):
        """Rows in run order (parallel jobs complete out of order); rewrites results.csv so."""
        with self.lock:
            rows = sorted(self.rows, key=lambda r: r['run_id'])
            with open(self.csvf, 'w', newline='') as f:
                w = csv.writer(f, lineterminator='\n')
                w.writerow(CSV_COLUMNS)
                w.writerows(csv_row(r) for r in rows)
        return rows


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def expand(groups, tests):
    """List of (test, cfg, tiles, hw) of all groups, without duplicates."""
    runs, seen = [], set()
    for g in groups:
        # a scalar is a one-element list
        hw_values = [g.get(p) or [None] for p in HW_PARAMS]
        hw_values = [v if isinstance(v, list) else [v] for v in hw_values]
        for name in g['tests']:
            if name not in tests:
                sys.exit('unknown test %r (see --list)' % name)
            t = tests[name]
            app = t.applicable()
            values = [g.get(p) or [t.defaults[p]] for p in app]
            values = [v if isinstance(v, list) else [v] for v in values]
            for tiles in g['tiles']:
                for hw_combo in itertools.product(*hw_values):
                    hw = dict(zip(HW_PARAMS, hw_combo))
                    for combo in itertools.product(*values):
                        cfg = {p: None for p in PARAMS}
                        cfg.update(zip(app, combo))
                        key = (name, tiles, tuple(hw_combo), tuple(cfg[p] for p in PARAMS))
                        if key not in seen:
                            seen.add(key)
                            runs.append((t, cfg, tiles, hw))
    # mesh side and hardware outermost, so gvsoc is rebuilt once per (side, hardware); then tests
    # and parameters together, so the sources change as little as possible
    runs.sort(key=lambda r: (r[2], r[3]['spatz_vlen'] or 0, redmule_hw(r[3]['redmule']) if r[3]['redmule'] else (0, 0),
                             r[0].name, [r[1][p] or 0 for p in PARAMS]))
    return runs


def simulate(t, cfg, tiles, hw, row, rdir, slot, build, work, t0, tag, args, rep, free):
    """Run a built configuration in job slot `slot`, measure it and add its row."""
    try:
        rc, outp = run_test(t, tiles, hw, build, work, os.path.join(rdir, 'make.log'), args)
        if STOP.is_set():
            return
        kills, errs = parse_run(outp)
        row['sim_seconds'] = round(time.time() - t0, 1)
        row['errors'] = sum(errs) if errs else None
        if rc is None:
            row['status'] = 'TIMEOUT'
            rep.log('%s TIMEOUT after %ds (%d/%d tiles exited), see %s' % (tag, args.timeout, len(kills),
                                                                          tiles * tiles, rdir))
        elif rc or len(kills) != tiles * tiles:
            row['status'] = 'BUILD/RUN FAIL' if rc else 'NO EXIT'
            rep.log('%s %s (rc=%d, %d/%d tiles exited), see %s' % (tag, row['status'], rc, len(kills),
                                                                 tiles * tiles, rdir))
        else:
            row['status'] = 'PASS' if errs and sum(errs) == 0 else 'FAIL'
            trace = os.path.join(work, 'trace.perfetto-trace')
            lo, hi, per_tile = analyze(t, trace, kills, tiles * tiles, build, args.no_profile, args.trace_processor)
            cyc = hi - lo
            macs = t.macs(cfg)
            if args.no_profile:  # Spatz busy only, no VFU
                for pt in per_tile:
                    del pt['vfu']
            mean = {k: 100 * np.mean([pt[k] for pt in per_tile]) for k in per_tile[0]}
            mean.setdefault('vfu', None)
            row.update(cycles=cyc, macs=macs, mac_per_cycle=macs / cyc,
                       mac_per_cycle_per_tile=macs / cyc / (tiles * tiles),
                       redmule_pct=mean['redmule'], idma_pct=mean['idma'], idma_in_pct=mean['idma_in'],
                       idma_out_pct=mean['idma_out'], spatz_pct=mean['spatz'], spatz_vfu_pct=mean['vfu'],
                       empty_pct=mean['empty'], empty_cv32_active_pct=mean['cv32_active'],
                       empty_cv32_sleep_pct=mean['cv32_sleep'], main_start_cycle=lo, end_cycle=hi)
            with open(os.path.join(rdir, 'per_tile.txt'), 'w') as f:
                f.write('tile ' + ' '.join('%11s' % k for k in per_tile[0]) + '   (%)\n')
                for k, pt in enumerate(per_tile):
                    f.write('%4d ' % k + ' '.join('%11.1f' % (100 * v) for v in pt.values()) + '\n')
            rep.log('%s %s: %s cycles, %.1f MAC/cycle, RedMulE %.1f%%, iDMA %.1f%%, Spatz %.1f%%, '
                    'empty %.1f%% (%.1fs)' % (tag, row['status'], format(cyc, ','), macs / cyc,
                                              mean['redmule'], mean['idma'], mean['spatz'],
                                              mean['empty'], time.time() - t0))
    except Exception as e:
        if STOP.is_set():
            return
        row['status'] = 'ANALYSIS FAIL'
        rep.log('%s analysis failed: %r' % (tag, e))
    finally:
        # traces are GBs: keep them only if asked or light (the VCD is left behind by a failed run)
        trace = os.path.join(work, 'trace.perfetto-trace')
        if (args.keep_traces or args.no_profile) and os.path.exists(trace) and not STOP.is_set():
            shutil.move(trace, rdir)
        for f in (trace, os.path.join(work, 'all.vcd')):
            if os.path.exists(f):
                os.remove(f)
        free.put(slot)
    rep.add(row)


def on_sigterm(*_):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep', help='JSON sweep file (list of groups)')
    ap.add_argument('--tests', nargs='+', help='tests (directory names in tests/magia/mgc)')
    ap.add_argument('--tiles', nargs='+', type=int, help='mesh sides (default: 2)')
    for p in PARAMS:
        ap.add_argument('--' + p, nargs='+', type=int, help='%s values (default: the test\'s own)' % p)
    ap.add_argument('--spatz-vlen', dest='spatz_vlen', nargs='+', type=int,
                    help='gvsoc Spatz VLEN values in bits (default: gvsoc\'s, 256)')
    ap.add_argument('--redmule', nargs='+', metavar='HxW',
                    help='gvsoc RedMulE array sizes, e.g. 8x8 16x16 (default: gvsoc\'s, 8x8)')
    ap.add_argument('--out', help='output directory (default: mgc_sweep/<date>_<time>)')
    ap.add_argument('--cache', default=os.path.join(SWEEP_DIR, 'cache'),
                    help='cache of generated test.h (default: %(default)s)')
    ap.add_argument('--gen-python', default='python3', help='Python with torch for the golden generators')
    ap.add_argument('--mgcc-python', default=os.path.join(MGC, '.venv', 'bin', 'python')
                    if os.path.exists(os.path.join(MGC, '.venv', 'bin', 'python')) else 'python3',
                    help='Python with xDSL for mgcc (default: %(default)s)')
    ap.add_argument('--jobs', '-j', type=int, default=1,
                    help='gvsoc simulations of the same mesh side to run in parallel (default: 1)')
    ap.add_argument('--keep-traces', action='store_true',
                    help='keep each run\'s perfetto trace, with all of make\'s profile events (default: '
                         'only the events the sweep measures, trace deleted after the analysis)')
    ap.add_argument('--no-profile', action='store_true',
                    help='make run_profiling with the trace reduced (through the environment) to 5 busy bits '
                         'per tile and tile 0\'s entry in main: VFU%% N/A, the small perfetto trace is kept '
                         'in runs/<id>/')
    ap.add_argument('--timeout', type=int, help='kill a simulation after this many seconds (status TIMEOUT)')
    ap.add_argument('--l2-size', help='gvsoc L2 size (make l2_size=..., e.g. 0x10000000), for tests with data in '
                    '.l2ext above the default L2 (mhsa/gqa_pipeopt3 with more than 12 MiB of L2 arrays)')
    ap.add_argument('--trace-processor', metavar='BIN',
                    help='perfetto trace_processor_shell to use (default: the perfetto package\'s own, or '
                         'v47.0 on hosts with glibc < 2.34)')
    ap.add_argument('--dry-run', action='store_true', help='only list the configurations')
    ap.add_argument('--list', action='store_true', help='list the tests and their default parameters')
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error('--jobs must be at least 1')

    tests = {n: Test(n) for n in list_tests()}
    if args.list:
        for t in tests.values():
            d = ' '.join('%s=%s' % (p, t.defaults[p]) for p in t.applicable())
            print('%-22s %-24s %s%s' % (t.name, t.target, d, '  goldens for tiles=%s' % t.goldens if t.goldens else ''))
        return 0

    if args.sweep:
        groups = json.loads(read(args.sweep))['groups']
    elif args.tests:
        groups = [{'tests': args.tests, **{p: getattr(args, p) for p in PARAMS + HW_PARAMS if getattr(args, p)}}]
    else:
        ap.error('give --sweep FILE or --tests')
    for g in groups:
        g.setdefault('tiles', args.tiles or [2])
    runs = expand(groups, tests)

    if args.dry_run:
        for i, (t, cfg, tiles, hw) in enumerate(runs):
            why = t.check(cfg, tiles)
            print('%3d  %-20s tiles=%-2d%s %s%s%s' % (i, t.name, tiles, hw_desc(hw),
                                                    ' '.join('%s=%s' % (p, cfg[p]) for p in t.applicable()),
                                                  '  [regenerate]' if t.needs_regen(cfg, tiles) else '',
                                                  '  SKIP: ' + why if why else ''))
        return 0

    # absolute: the job paths under it are passed to gvrun, which runs in its work directory
    out = os.path.abspath(args.out or os.path.join(SWEEP_DIR, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    args.cache = os.path.abspath(args.cache)
    os.makedirs(os.path.join(out, 'runs'), exist_ok=True)
    rep = Report(out)
    Sources.recover(rep.log)
    srcs = Sources(rep.log)
    signal.signal(signal.SIGTERM, on_sigterm)
    rep.log('mgc sweep: %d configurations, output in %s' % (len(runs), out))
    if args.no_profile and args.keep_traces:
        ap.error('--no-profile and --keep-traces are exclusive')
    args.trace_processor = trace_processor_bin(args.trace_processor, rep.log)

    pool = ThreadPoolExecutor(args.jobs)
    free = Queue()  # job slots not running a simulation
    for k in range(args.jobs):
        free.put(k)
    futures = []
    gvsoc_hw, gvsoc_rc = None, 0  # (mesh side, hardware) gvsoc is built for, and its build's return code
    try:
        for i, (t, cfg, tiles, hw) in enumerate(runs):
            rid = '%03d' % i
            rdir = os.path.join(out, 'runs', rid)
            os.makedirs(rdir, exist_ok=True)
            row = dict(run_id=rid, test=t.name, tiles=tiles, n_tiles=tiles * tiles, **hw, **cfg)
            desc = ' '.join('%s=%s' % (p, cfg[p]) for p in t.applicable())
            tag = '[%d/%d]' % (i + 1, len(runs))
            rep.log('%s %s tiles=%d%s %s' % (tag, t.name, tiles, hw_desc(hw), desc))
            gvsoc_log = os.path.join(out, 'gvsoc_tiles%d%s.log' % (tiles, hw_desc(hw).replace(' ', '_')))
            t0 = time.time()
            why = t.check(cfg, tiles)
            if why is None and t.needs_regen(cfg, tiles):
                why = generate(t, cfg, tiles, args, srcs, rep.log)
            elif why is None:
                for p in list(srcs.saved):  # back to the committed sources
                    shutil.copy2(srcs.saved[p], p)
            if why is None and (tiles, hw) != gvsoc_hw:
                wait(futures)  # the running jobs use the current gvsoc
                rep.log('  building gvsoc for tiles=%d%s...' % (tiles, hw_desc(hw)))
                gvsoc_hw = (tiles, hw)
                gvsoc_rc = build_gvsoc(tiles, hw, gvsoc_log, args)
            if why:
                row['status'] = 'SKIP'
                rep.log('  skipped: ' + why)
            elif gvsoc_rc:
                row['status'] = 'GVSOC FAIL'
                rep.log('  gvsoc build failed (rc=%d), see %s' % (gvsoc_rc, gvsoc_log))
            else:
                slot = free.get()
                build, work = slot_dirs(out, slot)
                rc = build_test(t, tiles, build, os.path.join(rdir, 'make.log'))
                if rc:
                    free.put(slot)
                    row['status'] = 'BUILD/RUN FAIL'
                    rep.log('  build failed (rc=%d), see %s' % (rc, rdir))
                else:
                    if args.jobs > 1:
                        rep.log('  simulating in job %d' % slot)
                    # the sources may change from now on: the simulation only needs the build
                    futures.append(pool.submit(simulate, t, cfg, tiles, hw, row, rdir, slot, build, work, t0,
                                               '  ' + tag if args.jobs > 1 else ' ', args, rep, free))
                    continue
            rep.add(row)
        wait(futures)
    except KeyboardInterrupt:
        STOP.set()
        for p in list(PROCS):
            kill(p)
        rep.log('interrupted')
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        srcs.restore()
    if not STOP.is_set():
        shutil.rmtree(os.path.join(out, 'jobs'), ignore_errors=True)

    rows = rep.sorted_rows()
    rep.log('')
    rep.log('Utilization: % of the cycles from tile 0\'s main to the last exit of tiles != 0, mean over tiles.')
    rep.log('MAC: matmuls only (softmax = 0). empty = no RedMulE/iDMA/Spatz activity: CV32 running '
            'alone (CV32 run) or asleep (CV32 sleep).')
    rep.log('')
    rep.log(table(rows))
    rep.log('')
    rep.log('CSV: ' + rep.csvf)
    return 0


if __name__ == '__main__':
    sys.exit(main())
