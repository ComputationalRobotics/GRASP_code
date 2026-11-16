"""Restart a GRASP-MPC run with larger MuJoCo Warp buffers when, and only when, it stopped on a buffer overflow.

MuJoCo Warp allocates contacts (``nconmax_per_env`` per world) and constraint rows (``njmax_per_env``) up front.  A run
that exceeds them stops with ``MJWarp capacity exceeded: ...`` (``grasp.sim.mjwp.check_capacity``) or, already when
the initial state is copied to the GPU, with ``ValueError: nconmax overflow (nconmax must be >= N)`` /
``njmax overflow``.  ``next_capacity`` reads that message and returns the enlarged buffer; the run restarts from
scratch with the same seed, so no state computed with truncated physics is ever reused.  Anything else is not retried.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from collections import deque

LIMITS = {"nconmax_per_env": 1024, "njmax_per_env": 8192}


def _grow(old: int, demand: int, key: str) -> int:
    # geometric growth: at least double, at least 1.25 x the demand, rounded up to a power of two
    target = 1 << (max(old * 2, math.ceil(demand * 1.25)) - 1).bit_length()
    if target > LIMITS[key]:
        raise RuntimeError(f"{key} needs {target}, above the retry limit {LIMITS[key]}")
    return target


def next_capacity(message: str, worlds: int, overrides: dict) -> dict:
    """Buffer overrides justified by the last capacity error in ``message``; ``{}`` if there is none."""
    errors = re.findall(r"^RuntimeError: MJWarp capacity exceeded: (.+)$", message, re.M)
    if errors:
        changes = {}
        for pattern, key, divisor in [
            (r"(?:broadphase pairs|contacts) (\d+) > naconmax (\d+)", "nconmax_per_env", worlds),
            (r"constraints (\d+) > njmax (\d+)", "njmax_per_env", 1),
        ]:
            matches = re.findall(pattern, errors[-1])
            if not matches:
                continue
            demand = max(math.ceil(int(a) / divisor) for a, _ in matches)
            old = max(math.ceil(int(b) / divisor) for _, b in matches)
            changes[key] = _grow(max(old, int(overrides.get(key, old))), demand, key)
        return changes
    init = re.findall(r"^ValueError: (nconmax|njmax) overflow \((?:nconmax|njmax) must be >= (\d+)\)$", message, re.M)
    if init:
        name, demand = init[-1]
        key = f"{name}_per_env"
        return {key: _grow(int(overrides.get(key, 0)), int(demand), key)}
    return {}


def run(command: list[str], worlds: int, overrides: dict | None = None, max_attempts: int = 6) -> int:
    """Run ``command`` (a ``grasp.mpc.run`` invocation; ``--override key=value`` is appended for the buffers), restarting
    on capacity errors up to ``max_attempts`` times.  Returns the last exit code."""
    capacities = dict(overrides or {})
    rc = 1
    for attempt in range(1, max_attempts + 1):
        argv = list(command) + [x for k, v in sorted(capacities.items()) for x in ("--override", f"{k}={v}")]
        print(f"[capacity-retry] attempt {attempt}: {capacities or 'default buffers'}", flush=True)
        tail: deque[str] = deque(maxlen=400)
        with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                sys.stdout.write(line)
                tail.append(line)
            rc = proc.wait()
        if rc == 0:
            return 0
        changes = next_capacity("".join(tail), worlds, capacities)
        if not changes:
            return rc
        capacities.update(changes)
    print(f"[capacity-retry] still failing after {max_attempts} attempts", flush=True)
    return rc
