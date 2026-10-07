# mgc

mgc ("magic") is a Python-syntax domain-specific language for MAGIA mesh tests. `mgcc.py` compiles a `.mgc`
file to a C test that uses mglib (`mg_idma_*`, `mg_redmule_*`, `mg_event_*`) and fsync.

The compiler handles tile partitioning, L1 layout, DMA address/stride arithmetic, multi-buffer
rotation, event storage, the RedMulE enqueue/commit/start sequence and neighbour-tile addressing.
`.mgc` files are parsed with Python's `ast` module and never executed. The compiler is built on
[xDSL](https://xdsl.dev).

mgc was developed using Claude Opus 5.5 for coding, debugging, and (partially) documentation under
human guidance.

## Setup and usage

```sh
python3 -m venv mgc/.venv
mgc/.venv/bin/pip install -r mgc/requirements.txt

# compile (the header defaults to include/test.h next to the source)
mgc/.venv/bin/python mgc/mgcc.py tests/magia/mesh/mm_os_mgc/mm_os.mgc \
    -o tests/magia/mesh/mm_os_mgc/src/test.c

# build and run as any other test
make build tiles=2 test=test_mm_os_mgc
make run tiles=2 test=test_mm_os_mgc platform=gvsoc

# compiler tests
mgc/.venv/bin/python -m unittest discover -s mgc/tests
```

| Option | Effect |
|---|---|
| `-o FILE` | output file (default: stdout) |
| `--header FILE` | test header with sizes and L2 arrays |
| `--print-ir-after=STAGE` | print the IR after `frontend` or a pass, then stop |
| `--print-ir-after-all` | print the IR after every stage to stderr |
| `--no-format` | skip clang-format (otherwise taken from `PATH` or `llvm/install/bin`) |

Generated files are committed next to their source. Regenerate them after editing a `.mgc` file
or the compiler; the unit tests fail if they are out of date.

## Example tests

| Directory | Source | Content |
|---|---|---|
| `tests/magia/mesh/mm_os_mgc` | `mm_os.mgc`, `mm_os_explicit.mgc` | output-static GEMM, 2-stage pipeline, RedMulE job overlapped with the next loads |
| `tests/magia/mesh/mm_ws_mgc` | `mm_ws.mgc`, `mm_ws_explicit.mgc` | weight-static systolic GEMM, partial sums move down columns |
| `tests/magia/mesh/mm_is_mgc` | `mm_is.mgc` | input-static systolic GEMM, partial sums move east along rows |
| `tests/magia/mesh/mm_os_l1k` | `mm_os_l1k.mgc`, `src/kernels.c` | output-static GEMM with a CV32 kernel in the pipeline |

`*_explicit.mgc` files write the schedule by hand; the others use `pipeline()`. The hand-written
references are `mm_os_mglib`, `mm_ws_mglib` and `mm_is_mglib`; the last two are mglib ports of
`mm_ws_2` and `mm_is_2` with the same schedule.

Weight-static, complete per-tile code:

```python
for y_id, x_id in tiles():
    tile_h = y_id.split(N)
    tile_w = x_id.split(K)
    timeslots: u8 = 4
    t_size: u8 = M // timeslots
    t_start: u8 = y_id * 2
    total_timeslots: u8 = (MESH_Y_TILES - 1) * 2 + timeslots + 1

    w = L1.single_buffer(W[tile_h, tile_w])
    x = L1.multi_buffer(X[0:t_size, tile_h], depth=2)
    y = L1.multi_buffer(Y[0:t_size, tile_w], depth=3)

    for z in range(N_ITERATIONS):
        idma.load(w, W[tile_h, tile_w]).wait()
        for pt in pipeline(timeslots, skew=t_start, steps=total_timeslots, sync=GLOBAL):
            if y_id == 0:
                idma.load(y[pt], Y[pt * t_size:(pt + 1) * t_size, tile_w])
            idma.load(x[pt], X[pt * t_size:(pt + 1) * t_size, tile_h])
            j = redmule.enqueue(x[pt], w, y[pt])          # y += x @ w
            redmule.commit_start()
            j.wait(lag=0)                                 # done within the step
            if y_id == MESH_Y_TILES - 1:
                idma.store(Y[pt * t_size:(pt + 1) * t_size, tile_w], y[pt])
            else:
                idma.store(y.at_rel(1, 0)[pt], y[pt])          # same slot, tile below

    sync(COL)
    if y_id == MESH_Y_TILES - 1:
        check(Y[:, tile_w], Z[:, tile_w], tol=0x11)
```

## Language

### Module level

| Statement | Meaning |
|---|---|
| `A, B = sizes("A_SIZE", "B_SIZE")` | Imports dimensions that **already exist** as `#define`s in the test header (`--header`, e.g. `include/test.h`). Each Python name (`A`, `B`) stands for the C macro whose name is given as a string; the generated C uses the macro by name and nothing is emitted. The compiler errors if a macro is missing from the header. Use it for anything the header's data depends on: tensor and tile dimensions, shared with the L2 arrays and the golden data. |
| `X = l2("x_inp", (A, B), fp16)` | Declares a tensor in L2 memory, row-major, with the given shape (built from `sizes` names or constants) and element type (`fp16`, `u8`, ...). `"x_inp"` is the name of the C array holding its data, which must exist in the header; the compiler errors otherwise. `X` is how the program refers to it. |
| `NAME = define(value)` | Creates a **new** preprocessor constant owned by the `.mgc` file: the generated C gets `#define NAME value`, and `NAME` can be used in expressions (e.g. loop bounds). Unlike `sizes`, the value is not in the header, so the header and its golden-data generator cannot see it. Use it for schedule tunables such as `N_ITERATIONS` or `N_TIMESLOTS`. `WAIT_MODE` (`WFE` or `POLLING`) selects how the test waits for completion and defaults to `WFE` if not defined. |
| `k = l1_kernel("c_fn", operands=("dst:out", "src:in"), params=("src.size",), executor=CORE)` | Declares a software kernel: a C function `c_fn`, written by you in the test (e.g. `src/kernels.c`), that runs on the tile's core and works on data in L1. `operands` lists the L1 buffers it takes, each with a role (`in`, `out`, `inout`) that the compiler uses to order operations. `params` are extra integer arguments computed from operand shapes (`src.size`, `src.bytes`, `src.shape[i]`). `executor` says who runs it (`CORE` is the only one at present). Calling `k(...)` in `main` invokes the function. |
| `@test("name", authors=[...])` + `def main():` | Marks `main` as the test being compiled: `name` is the test name and `authors` are listed in the generated file header. The docstring of `main` becomes the descriptive comment at the top of the generated C. `main` is the only function allowed in the file. |
| `@top` / `@top(authors=[...])` + `def name():` | Alternative to `@test`, for code meant to be called from a larger program: the generated C is a `void name(void)` function (named after the Python function) with no `main`, no `#include "test.h"` and no `return` value. It still initialises the idma/redmule/eu/fsync controllers itself, so it is a drop-in call. `check()` is not available (nothing to return), and a file holds either one `@test` `main` or one `@top` function. The `--header` is optional. |

### Loops

In `mgc`, "loops" encapsulate a diversity of meanings and can be either *spatial* or *temporal*.
All `mgc` programs include a **spatial tile loop**, which encapsulates tile-level parallelism.
The `main` contains exactly one such loop in the form `for y_id, x_id in tiles():`.
The body of the tile loop runs equally in each tile, in a variant of the SPMD (Single Program, Multiple Data Stream) paradigm.
`y_id` and `x_id` are the tile IDs, which can be used in expressions and conditions.

| Loop | Kind | Emitted |
|---|---|---|
| `for y_id, x_id in tiles():` | spatial | tile coordinates from `hartid`, `l1_tile_base` |
| `for v in pipeline(n, ...)` | temporal | software pipeline (see below) |
| `for v in range(n)`, `range(a, b)` | temporal | regular `for` loop |
| `for c in cores():` | spatial | reserved for future usage with the PULP cluster in each tile, compile error (for now) |

### Scalars, tile splits and other statements

- Besides the loops above, the body of `main` accepts assignment, `if`/`elif`/`else`, `continue`
  and `pass`.
- `name: u8 = expr` declares a variable of type `u8`, `u16`, `u32` or `i32`. Values known at
  compile time are range-checked. An unannotated first assignment declares a `uint32_t`; later
  assignments (`pt = pt + 1`, `pt += 1`) update it. `//` is treated as integer division.
- `h = y_id.split(E)` splits extent `E` over the mesh rows (`x_id.split` over columns) in blocks
  of `h_max = ceil(E / MESH_Y_TILES)`. Edge tiles are clipped, and tiles with an empty block
  return 0. `h` can index a tensor dimension (start `h_max * y_id`, size `h`). `h.start` and
  `h.size` are expressions.

### Views and L1 buffers

`T[i, j, ...]` on an L2 tensor is a view. Each index is a slice `a:b` or `:` with unit step, a
tile split, or an integer expression (which removes the dimension).

| Expression | Meaning |
|---|---|
| `b = L1.single_buffer(view)` | dense L1 buffer with the view's shape; the view also fixes the transfer geometry |
| `b = L1.single_buffer((d0, d1, ...), dtype)` | L1-only buffer: each `d` is a tile split (its per-tile size) or an integer expression, e.g. `L1.single_buffer((tile_h, tile_w), fp16)`. No L2 tensor is involved; transfers to/from L2 must be dense. `L1.multi_buffer((d0, ...), dtype, depth=N)` works the same way |
| `b = L1.multi_buffer(view, depth=N)` | `N` L1 buffers (*slots*, `N` ≥ 2 and constant), each shaped like `view`, used for double/triple buffering: e.g. load the next tile into one slot while an accelerator reads the current one. `b` itself is not an operand; it must be indexed to pick a slot. `b[c]` with a constant `c` (e.g. `b[0]`) is always the same, fixed slot. `b[v + k]`, where `v` is a loop variable and `k` a constant offset with \|k\| < N, is a *rotating* slot: the slot used changes with `v` (it is slot `(v + k) % N`), so `b[i]` and `b[i + 1]` mean "current" and "next" in iteration `i`. |
| `b.at_rel(dy, dx)[k]` | The copy of buffer `b` that lives in the L1 of a **neighbour tile**, i.e. the tile at `(y_id + dy, x_id + dx)`; `dy` and `dx` must be constants (e.g. `b.at_rel(0, 1)` is the tile to the right, `b.at_rel(1, 0)` the one below). Every tile allocates `b` at the same L1 offset, so the neighbour's copy is found at the neighbour's L1 base plus the local offset. `[k]` picks a slot as for a local multi-buffer (omit it if `b` has a single slot). It can only be used as the destination of `idma.store`, to push a local buffer to a neighbour: `idma.store(b.at_rel(0, 1)[k], b[k])` copies local slot `k` of `b` into the same slot of `b` on the tile to the right. Source and destination must be the same buffer and the same slot. |
| `b.at(y, x)[k]` | As `b.at_rel`, but on the tile with **absolute** coordinates `(y, x)`, which can be any integer expressions, including run-time ones (e.g. `b.at(y_id, 0)`, the first tile of this row). The tile's L1 base is computed where the transfer is issued (`get_l1_base(GET_ID(y, x))`). |
| `b.at(y, x)[i, j, ...]` | A **block** of single-slot buffer `b` on another tile (works with `at_rel`, `at`, `row`, `col`). Indices are as for L2 views (slices, or integers that drop the dimension), over `b`'s shape; there must be more than one index or a slice, otherwise it is a slot. Only the destination of `idma.store`, whose source can then be **any** local buffer or slot of the same shape and dtype: `idma.store(rs.at(y_id, 0)[x_id, :], r)` gathers `r` into row `x_id` of `rs` on the first tile of the row. The block's offset is computed with this tile's sizes, so tile splits in `b`'s shape must be the same on the receiving tile (e.g. a split along the axis shared by the two tiles). |
| `b[i, j, ...]` | A **block** of local single-slot buffer `b`, with indices as for `b.at(y, x)[i, j, ...]` (more than one index or a slice). Only the source of a store into a block of another buffer, and it must be contiguous: `idma.store(rs.at(y_id, 0)[x_id, :], rs[0, :])` sends row 0 of the local `rs` to row `x_id` of `rs` on the first tile of the row. |
| `b.row[k]`, `b.col[k]` | **Multicast**: the copies of `b` on every *other* tile of this tile's row (`row`) or column (`col`). Same rules as `b.at_rel`: `idma.store(b.row[k], b[k])` copies local slot `k` into slot `k` of `b` on all the other tiles of the row. |

### Transfers and events

| Statement | Meaning |
|---|---|
| `e = idma.load(slot, view)` | asynchronous L2 → L1 |
| `e = idma.store(view, slot)` | asynchronous L1 → L2 |
| `e = idma.store(b.at_rel(dy, dx)[k], b[k])` | asynchronous L1 → neighbour's L1, same slot |
| `e = idma.store(b.at(y, x)[k], b[k])` | asynchronous L1 → L1 of tile `(y, x)`, same slot |
| `e = idma.store(b.row[k], b[k])`, `b.col[k]` | asynchronous L1 → L1 of every other tile of the row / column, same slot |
| `e = idma.store(c.at(y, x)[i, :], b[k])` | asynchronous L1 → a block of buffer `c` on tile `(y, x)`; also `at_rel`, `row`, `col`. The block may be 2-D strided (`c.at(y, x)[:, 4:8]`) |
| `e = idma.store(c.at(y, x)[i, :], b[j, :])` | as above, from a contiguous block of local buffer `b` |
| `e.wait()` | wait for the transfer |

Allowed transfers follow `devices.IDMA`. After merging contiguous dimensions, a transfer is at
most `max_rank`-dimensional (currently 2: `mg_idma_memcpy_1d/2d`), with unit steps and a dense
L1 side. L1→L1 within a tile and L2→L2 are rejected.

Row/column multicasts are lowered by `idma-mcast` to one unicast per destination tile (a `for` over
`MESH_X_TILES`/`MESH_Y_TILES` that skips the issuing tile), all on the same event: same-direction
iDMA transfers complete in issue order, so one `e.wait()` waits for all of them. Once the iDMA
supports multicast in hardware, an `IDMA.multicast` entry makes it a single call.

Each L1 buffer has one event per direction (`idma_evt_<buf>_in/_out`). The two are merged into
`idma_evt_<buf>` when they are never pending at the same time. The following are compile errors:
issuing a transfer while the previous one on the same event is certainly still pending, and
waiting on an event that was never issued.

### Accelerators, kernels, synchronization

| Statement | Emitted |
|---|---|
| `j = redmule.gemm(x, w, y)` then `j.wait()` | `mg_redmule_gemm` (y += x @ w), `mg_redmule_wait` |
| `ev = redmule.events(n)` | event array; index with `ev[c]` or `ev[v + k]` |
| `redmule.enqueue(x, w, y, event=ev[k])` | `mg_redmule_gemm_enqueue` |
| `redmule.commit()`, `start()`, `commit_start()` | `mg_redmule_gemm_commit`, `_start`, `_commit_start` |
| `ev[k].wait()` | `mg_redmule_wait` |
| `commit…(lag=k)`, `.wait(lag=k)` | inside `pipeline()` only: place the step `k` stages later (see below) |
| `k(a, b, ...)` (an `l1_kernel`) | `c_fn(a, b, ..., params)`, a synchronous call on the CV32 core |
| `sync(GLOBAL)`, `sync(ROW)`, `sync(COL)` | `fsync_sync_global/row/col` + `eu_fsync_wait` |
| `rotate(v)` | places the multi-buffer slot selection for `v` here (optional) |
| `check(result, golden, tol=0x11)` | per-tile comparison loop, `return errors`; allowed inside `if` |

Accelerator operand ranks, parameters (`m, n, k`) and shape constraints come from
`devices.HWPES`.

L1 kernel operands must be L1 buffers or slots, each declared with a role (`in`, `out`,
`inout`); the compiler uses the roles to order operations. `params` are expressions over operand
shapes: `name.shape[i]`, `name.size` (elements) or `name.bytes`. The C function receives the
operand addresses, then the parameters, all as `uint32_t`, and must be provided by the test
(e.g. `src/kernels.c`).

### `pipeline(n, skew=None, steps=None, sync=None)`

The body describes one iteration `i`: transfers, jobs, CV32 kernel calls and HWPE protocol steps,
with transfers and jobs optionally inside `if` statements that do not depend on `i`. The compiler:

1. **Assigns stages.** An item is placed one stage after the latest earlier item that writes a
   slot it reads; otherwise it is in stage 0. Examples: load (0) → kernel (1) → store (2).
   HWPE protocol steps follow their own rules (below).
2. **Checks depths.** Each multi-buffer must have at least as many slots as the number of stages
   its slots are live across. Each HWPE can hold `queue_depth` jobs (2 for RedMulE), and each job
   in flight needs its own event.
3. **Generates the schedule.** A prologue runs stage 0 of iteration 0. In loop step `s`,
   stage `k` works on iteration `s + 1 - k`, guarded by `0 <= s + 1 - k < n`. A step runs in four
   phases:
   1. transfers, enqueues and `redmule.gemm` one-shots are issued;
   2. commits/starts follow, each after the waits of the DMAs it needs, so the HWPE computes
      while the core is busy with the next phase;
   3. CV32 kernel calls, each followed by the commits that consume its output;
   4. the remaining waits, in issue order.

With `skew` and `steps`, the loop runs over `steps` global time steps. The tile is active when
`0 <= t - skew <= last step`. `sync=SCOPE` adds a barrier after the prologue and at the end of
every time step, active or not.

#### HWPE jobs

A job is written as the steps of the HWPE protocol:

```python
ev = redmule.events(2)
for i in pipeline(timeslots):
    idma.load(x[i], X[...])
    idma.load(w[i], W[...])
    redmule.enqueue(x[i], w[i], y, event=ev[i])   # program the job: addresses only
    redmule.commit_start()                         # or commit() and start()
    ev[i].wait()
```

| Step | Stage |
|---|---|
| `commit_start()` / `commit()` | the stage of its latest producer, placed after it in the step (after the DMA's wait, or after the CV32 call); one stage later if the producer is an asynchronous job, or if the operand is written by another tile (data from a neighbour is valid only after the barrier) |
| `enqueue(...)` | same stage as its commit, issued together with the transfers so that programming overlaps the DMA |
| `start()` | commit + `lag` |
| `.wait()` | start (or `commit_start`) + `lag`, **default `lag=1`**: job `i` runs while iteration `i + 1` loads; the wait is the writer of the job's outputs for later items |

`lag=k` on `commit`/`commit_start`/`start`/`.wait()` moves the step `k` stages later. In the
example, the step of iteration `i + 1` loads, enqueues and commits job `i + 1` behind job `i`,
then waits for job `i`. `.wait(lag=0)` keeps the job inside its step, as the systolic WS/IS
tests do. With load → job → store, the store follows the wait one stage later, so `y` needs
depth 3. The compiler checks that each protocol is complete and in order. It also checks that
there is at most one explicit job per accelerator per `pipeline()` body and that the job does
not exceed the queue or the event array. `event=` may be omitted when only one job is in flight.
`redmule.gemm(...)` remains available as a one-shot job, issued and waited within its step.

The control-flow shape of the generated C follows the hand-written tests, because GCC's output
depends on it:

- 2-stage pipelines use `if (next) {...; wait} else {wait}`;
- deeper pipelines use one guard per stage.

### Comments

Docstrings become `/** */` comments; standalone `#` lines and `comment("...")` become `//`
comments in the generated C.

## Compiler

In `mgc`, the `.mgc` file is parsed into a typed IR (the `mg` dialect, built on xDSL) that keeps the *intent* of the program:
tensors in L2, per-tile L1 buffers, views, DMA transfers, accelerator jobs, events and
pipelines. A fixed sequence of passes then lowers that intent to low-level details,
and the last stage prints C against mglib. Each pass can be inspected on its own with `--print-ir-after`.

```
.mgc → frontend → legalize-idma → pipeline → job-lower → multi-buffer → l1-layout → event-alloc → idma-mcast → emit-c → clang-format
```

`mgc` also assists the MAGIA programmer by taking some decisions autonomously and rejecting some
common mistakes. In particular:
- it **decides** tile sizes and borders, L1 addresses (identical on all tiles), L2 offsets,
  iDMA length/stride/repetition, 1-D vs 2-D transfer selection, buffer rotation, event variables,
  HWPE job sequence, and the guarded stage schedule of `pipeline()`.
- it **rejects** views that the iDMA cannot express, accelerator operands with the wrong shape,
  pipelines deeper than their buffers, and events that are waited on without being issued
  (or reused before the previous transfer/job on it was waited).

### Tiny example

One GEMM per tile, no pipelining (`Y += X @ W`, each tile owns an `M/Y × K/X` block of `Y`):

```python
from mgc import *

M, N, K = sizes("M_SIZE", "N_SIZE", "K_SIZE")   # from include/test.h
X = l2("x_inp", (M, N), fp16)
W = l2("w_inp", (N, K), fp16)
Y = l2("y_inp", (M, K), fp16)

@test("test_tiny")
def main():
    for y_id, x_id in tiles():
        tile_h = y_id.split(M)                    # rows owned by this tile
        tile_w = x_id.split(K)                    # columns owned by this tile
        x = L1.single_buffer(X[tile_h, :])
        w = L1.single_buffer(W[:, tile_w])
        y = L1.single_buffer(Y[tile_h, tile_w])
        idma.load(x, X[tile_h, :])
        idma.load(w, W[:, tile_w])
        idma.load(y, Y[tile_h, tile_w]).wait()
        redmule.gemm(x, w, y)
        idma.store(Y[tile_h, tile_w], y).wait()
```

The 14 source lines become about 130 lines of C (excerpt):

```c
uint32_t len_w     = tile_w * 2;                 // one row of the W block, in bytes
uint32_t std_w     = K_SIZE * 2;                 // row stride in L2
uint32_t reps_w    = (uint32_t)N_SIZE;           // number of rows
uint32_t l1_addr_w = l1_addr_x + (tile_h * N_SIZE * 2);   // L1 layout: x, then w, then y
uint32_t l2_addr_w = (uint32_t)w_inp + (tile_w_max * x_id * 2);
...
mg_idma_memcpy_2d(&idma_ctrl, &eu_ctrl, WAIT_MODE, 0,
                  l2_addr_w, l1_addr_w, len_w, std_w, reps_w, &idma_evt_w, NULL);
...
mg_redmule_gemm(&redmule_ctrl, &eu_ctrl, WAIT_MODE, l1_addr_x, l1_addr_w, l1_addr_y,
                (uint16_t)tile_h, (uint16_t)N_SIZE, (uint16_t)tile_w, &redmule_evt, NULL);
```

The `W[:, tile_w]` column block is not contiguous in L2, so `legalize-idma` turns it into a 2-D
transfer, while `X[tile_h, :]` is contiguous and stays 1-D. The GEMM dimensions come from the
buffer shapes, and the `tile_h`/`tile_w` borders handle meshes that do not divide the problem.
To see the intermediate steps, run `mgcc.py tiny.mgc --print-ir-after=frontend` (or any pass name).
Larger schedules replace the straight-line body with `pipeline()` and multi-buffered
buffers, as in the weight-static example above.

### Source map

| File | Content |
|---|---|
| `mgcc.py` | command line |
| `mgc/frontend.py` | Python AST → `mg` IR, name resolution |
| `mgc/ir.py` | `mg` dialect (ops, attributes, `!mg.buf` type) |
| `mgc/expr.py` | symbolic integer expressions and conditions |
| `mgc/devices.py` | hardware description: iDMA capabilities, HWPE descriptors, executors, sync scopes |
| `mgc/passes/*.py` | one file per pass |
| `mgc/emit_c.py` | C printer |
| `mgc/driver.py` | runs the stages |
| `tests/test_mgc.py` | compiler tests |

IR conventions:

- Control flow, operations and L1 buffers (`!mg.buf` SSA values) are IR.
- Index arithmetic is a symbolic expression (`#mg.expr`) over C names, so the emitted C keeps
  the source expressions.
- Events are referenced by storage name (`#mg.evt`), not as SSA values, because explicit
  schedules reuse and rotate event variables across iterations and branches.

| Pass | Job |
|---|---|
| `legalize-idma` | views → N-dimensional transfers, checked against `devices.IDMA`; base and per-transfer L2 addresses |
| `pipeline` | `mg.pipeline` → explicit guarded schedule: stages, HWPE protocol placement, depth/queue checks |
| `job-lower` | `mg.job` → RedMulE one-shot or CV32 call; parameters and shape checks from the descriptors |
| `multi-buffer` | slot indices → static addresses or rotating pointers with an `if`/`switch` selection |
| `l1-layout` | identical L1 layout on all tiles; uses `*_max` tile sizes when buffers are accessed remotely; neighbour L1 bases |
| `event-alloc` | event variable names; dataflow check of issue/wait |
| `idma-mcast` | row/column multicasts → loops of unicast transfers, unless `devices.IDMA.multicast` lists the scope |
| `emit-c` | prints C; each buffer's `len_/std_/reps_/l1_addr_/l2_addr_` is declared once and referenced by name |

Extending:

| To add | Change |
|---|---|
| an iDMA mode (3-D, ...) | `IDMA.max_rank` and `IDMA.functions` in `devices.py` |
| hardware row/column multicast | `IDMA.multicast` in `devices.py` (scope → mglib function) |
| an HWPE accelerator | an `Hwpe` entry in `devices.py` (operands, parameters, constraints, queue depth, function names) and its mglib wrapper |
| a software kernel | `l1_kernel(...)` in the `.mgc` file and the C function |
| Spatz / PULP offload | an asynchronous executor (needs an mglib layer first) |

## Results

2×2 mesh, gvsoc. Exit cycle of each tile, sorted. Every test reports 0 errors.

> The `mm_os_mgc`, `mm_ws_mgc` (`mm_ws.mgc`), `mm_is_mgc` and `mm_os_l1k` rows are measured with
> the explicit HWPE protocol in `pipeline()` (2026-10-06). On the same gvsoc build, the code
> generated by the earlier early-enqueue template gives 355888 / 355903 / 391194 / 391209 (OS),
> 247466 / 247500 / 528967 / 594869 (WS), 245423 / 245431 / 583806 / 583814 (IS) and
> 466622 / 466635 / 506299 / 506303 (L1 kernel). The WS/IS rows are therefore not comparable with
> the reference rows, which were measured on an earlier setup and have not been re-run.

| Test | Cycles |
|---|---|
| `mm_os_mglib` (reference) | 355929 / 355970 / 391240 / 391241 |
| `mm_os_mgc` | 355517 / 355529 / 390745 / 390757 |
| `mm_ws_2` (raw HAL) | 42954 / 43002 / 153104 / 177774 |
| `mm_ws_mglib` (reference) | 45735 / 45769 / 156153 / 180870 |
| `mm_ws_mgc` (`mm_ws.mgc`) | 247499 / 247526 / 528999 / 594873 |
| `mm_ws_mgc` (`mm_ws_explicit.mgc`) | 45398 / 45432 / 152476 / 177424 |
| `mm_is_2` (raw HAL) | 43292 / 43297 / 171430 / 171435 |
| `mm_is_mglib` (reference) | 46064 / 46064 / 174466 / 174466 |
| `mm_is_mgc` | 245316 / 245317 / 583684 / 583685 |
| `mm_os_l1k` | 466796 / 466827 / 506287 / 506297 |

- **OS:** `mm_os_mglib` clips `tile_w` against `N_SIZE` instead of `K_SIZE`; mgc uses `K_SIZE`.
  With that line changed back, `main` disassembles identically and the cycles are equal.
  `mm_os.mgc` and `mm_os_explicit.mgc` produce the same C, comments aside.
- **WS/IS:** the mglib references are about 3k cycles slower than the raw-HAL tests (mglib
  event bookkeeping). The mgc versions are 0.8–1.9% faster than the references on the slowest
  tile. The differences:
  - neighbour addresses are computed as `l1_base_s + (slot - l1_tile_base)` instead of a
    per-slot `switch`;
  - the pipelined version prefetches the top row's bias block one step early;
  - L1 offsets use `*_max` tile sizes. `mm_ws_2` and `mm_is_2` use the local tile size, which
    gives wrong neighbour addresses when edge tiles are clipped.
- **L1 kernel:** issuing RedMulE before the CV32 call takes 29k cycles (5.7%) less than calling
  the kernel first. With the explicit protocol, job `i - 1` runs on RedMulE during the CV32
  call of iteration `i`, and job `i` is committed right after it.

## Limitations

- One `tiles()` loop and one `pipeline()` per test; `cores()` left for future implementation to enable support for PULP cluster parallelism.
- One explicit HWPE job (enqueue / commit / wait) per accelerator per `pipeline()` body; HWPE
  protocol steps cannot be inside `if`. DMA waits inside `pipeline()` are always placed by the
  compiler.
- fp16 only; RedMulE GEMM is the only HWPE accelerator described.
- L1 kernels run only on the CV32 control core.
- Comments inside a `pipeline()` body are dropped, except the leading docstring.
- No L1 capacity check (tile sizes are run-time values).
- The event check accepts code that is valid on at least one path, so it can miss errors that
  depend on correlated conditions.
- Multicasts are lowered to unicast transfers. On a 1-wide row/column nothing is transferred, so
  waiting on the multicast's event is invalid there.
