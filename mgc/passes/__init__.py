# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""The mgc pass pipeline, in order (see ../../README.md).

`PASSES` is the list of passes the driver applies after the front-end, each an
xDSL `ModulePass` with a `name` (usable with `--print-ir-after=NAME`) and an
`apply(ctx, module)` that rewrites the IR in place:
legalize-idma, uninit-acc, pin-check, pipeline, job-lower, multi-buffer, l1-layout, event-alloc, idma-mcast.
"""

from .events import EventAlloc
from .idma_mcast import IdmaMcast
from .job_lower import JobLower
from .l1_layout import L1Layout
from .legalize_idma import LegalizeIdma
from .multibuffer import MultiBuffer
from .pin import PinCheck
from .pipeline import Pipeline
from .uninit_acc import UninitAcc

PASSES = [LegalizeIdma(), UninitAcc(), PinCheck(), Pipeline(), JobLower(), MultiBuffer(), L1Layout(), EventAlloc(), IdmaMcast()]
