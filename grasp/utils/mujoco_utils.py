# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Source: SPIDER, https://github.com/facebookresearch/spider, spider/mujoco_utils.py
# at commit 44717007de41cbef7565dff7ff9f4453557a2d3d, adapted for GRASP.
# Licensed under CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.

"""Utils for mujoco.

Author: Chaoyi Pan
Date: 2025-11-01
"""

from contextlib import contextmanager

import mujoco
import mujoco.viewer


def get_viewer(show_viewer: bool, model: mujoco.MjModel, data: mujoco.MjData):
    if show_viewer:
        run_viewer = lambda: mujoco.viewer.launch_passive(model, data)
    else:
        cam = mujoco.MjvCamera()
        cam.type = 2
        cam.fixedcamid = 0

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {"is_running": lambda: True, "sync": lambda: None, "cam": 0},
            )

    return run_viewer
