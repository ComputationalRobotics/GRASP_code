# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Source: SPIDER, https://github.com/facebookresearch/spider, spider/preprocess/_decompose_common.py
# at commit 44717007de41cbef7565dff7ff9f4453557a2d3d, adapted for GRASP.
# Licensed under CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.

from collections.abc import Iterable

import numpy as np

MeshPart = tuple[np.ndarray, np.ndarray]


def flatten_base(
    hulls: Iterable[MeshPart],
    thickness: float = 0.01,
    R_world_local: np.ndarray | None = None,
    obj_world_pos: np.ndarray | None = None,
    floor_z: float = 0.0,
    plate_xy: np.ndarray | tuple[float, float] | None = None,
    plate_size: float = 0.03,
) -> list[MeshPart]:
    """Append a thin support plate body-fixed to the object.

    The plate is a small (``plate_size`` x ``plate_size``) square whose BOTTOM
    sits at the object's lowest world-frame z (``floor_z``) and TOP at
    ``floor_z + thickness``. The plate lives *inside* the object's lower bbox
    so it adds no height below the object — the object's effective lowest z
    is unchanged.

    If ``R_world_local`` and ``obj_world_pos`` are given, the plate is built
    in world coordinates centered at ``plate_xy`` (default: object's lowest
    contact xy in world frame), then transformed into the object's local
    frame so it follows the body.

    Otherwise (legacy fallback), the plate is added in the convex hull's
    local frame at the hull's ``min_z`` with the same small footprint.
    """
    hull_list = list(hulls)
    if not hull_list:
        return hull_list

    all_vertices = np.vstack([vertices for vertices, _ in hull_list])
    half = plate_size / 2.0

    if R_world_local is not None and obj_world_pos is not None:
        if plate_xy is None:
            v_world = (R_world_local @ all_vertices.T).T + obj_world_pos
            cx = float(0.5 * (v_world[:, 0].min() + v_world[:, 0].max()))
            cy = float(0.5 * (v_world[:, 1].min() + v_world[:, 1].max()))
        else:
            cx = float(plate_xy[0])
            cy = float(plate_xy[1])
        wx_min, wx_max = cx - half, cx + half
        wy_min, wy_max = cy - half, cy + half
        z_bot = floor_z
        z_top = floor_z + thickness
        corners_world_top = np.array(
            [
                [wx_min, wy_min, z_top],
                [wx_max, wy_min, z_top],
                [wx_max, wy_max, z_top],
                [wx_min, wy_max, z_top],
            ]
        )
        corners_world_bot = corners_world_top.copy()
        corners_world_bot[:, 2] = z_bot
        corners_world = np.vstack([corners_world_bot, corners_world_top])
        plate_vertices = (R_world_local.T @ (corners_world - obj_world_pos).T).T
    else:
        cx = float(0.5 * (all_vertices[:, 0].min() + all_vertices[:, 0].max()))
        cy = float(0.5 * (all_vertices[:, 1].min() + all_vertices[:, 1].max()))
        min_z = float(all_vertices[:, 2].min())
        z0 = min_z
        z1 = min_z + thickness
        plate_vertices = np.array(
            [
                [cx - half, cy - half, z0],
                [cx + half, cy - half, z0],
                [cx + half, cy + half, z0],
                [cx - half, cy + half, z0],
                [cx - half, cy - half, z1],
                [cx + half, cy - half, z1],
                [cx + half, cy + half, z1],
                [cx - half, cy + half, z1],
            ]
        )

    plate_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=int,
    )

    hull_list.append((plate_vertices, plate_faces))
    return hull_list
