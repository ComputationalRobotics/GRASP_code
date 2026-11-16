# Source: Do As I Do, https://github.com/malik-group/do-as-i-do, retargeting/retargeting/pipeline/decompose_mesh.py
# (_local_thickness, thicken_mesh, coacd_convex_decomp, dilate_hulls) at commit 824591b808c342b20079c3b4198a8c2bdf88c74e.
# MIT License; see THIRD_PARTY_NOTICES.md.

"""Object collision hulls: thicken thin walls, CoACD convex decomposition, dilate every hull.

``hulls(mesh)`` is the OakInk2 decomposition used by ``grasp.pipeline.decompose_fast`` (``method="coacd"``): regions
thinner than 2 mm are thickened to 2 mm, CoACD splits the result (concavity threshold 0.05, at most 32 hulls of at most
256 vertices), and each hull is expanded outward by 2 mm.  Thickening needs ``rtree`` (trimesh ray casting).
"""

from __future__ import annotations

from collections.abc import Iterable

import coacd
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

MeshPart = tuple[np.ndarray, np.ndarray]


def coacd_convex_decomp(
    mesh: trimesh.Trimesh,
    threshold: float = 0.05,
    max_convex_hull: int = 40,
    max_ch_vertex: int = 256,
    preprocess_resolution: int = 100,
) -> list[MeshPart]:
    coacd_mesh = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        max_ch_vertex=max_ch_vertex,
        preprocess_resolution=preprocess_resolution,
        decimate=True,
    )
    hulls: list[MeshPart] = []
    for vertices, faces in parts:
        hulls.append((np.asarray(vertices), np.asarray(faces, dtype=int)))
    return hulls


def _local_thickness(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex thickness via inward raycasting; returns nearest hit distance."""
    origins = mesh.vertices
    directions = -mesh.vertex_normals
    hits, ray_ids, _ = mesh.ray.intersects_location(origins, directions)
    # closest hit that isn't the origin itself
    thickness = np.full(len(mesh.vertices), np.inf)
    for hit_pos, ray_id in zip(hits, ray_ids):
        dist = np.linalg.norm(hit_pos - origins[ray_id])
        if dist > 1e-6 and dist < thickness[ray_id]:
            thickness[ray_id] = dist
    return thickness


def thicken_mesh(mesh: trimesh.Trimesh, min_thickness: float) -> trimesh.Trimesh:
    """Thicken only the regions thinner than min_thickness, leaving thick parts alone."""
    local_t = _local_thickness(mesh)
    deficit = np.maximum(min_thickness - local_t, 0.0)
    half_offset = (deficit / 2)[:, None]
    normals = mesh.vertex_normals
    outer_verts = mesh.vertices + normals * half_offset
    inner_verts = mesh.vertices - normals * half_offset
    all_verts = np.concatenate([outer_verts, inner_verts], axis=0)
    n_verts = len(mesh.vertices)
    outer_faces = mesh.faces
    inner_faces = mesh.faces[:, ::-1] + n_verts
    all_faces = np.concatenate([outer_faces, inner_faces], axis=0)
    # Outer/inner copies coincide at vertices where deficit==0; leaving them
    # duplicated produces a non-manifold input that crashes CoACD on certain
    # meshes. process=True merges duplicate verts and degenerate faces.
    return trimesh.Trimesh(all_verts, all_faces, process=True)


def dilate_hulls(
    hulls: Iterable[MeshPart], margin: float = 0
) -> list[MeshPart]:
    """Expand each convex hull outward by margin (Minkowski sum with a sphere).

    Offsets each face's vertices along that face's normal rather than the vertex
    normal: vertex-normal dilation under-expands at sharp edges (normals average
    out), which would leave seams between thin-walled pieces unsealed.
    """
    dilated: list[MeshPart] = []
    for vertices, faces in hulls:
        mesh = trimesh.Trimesh(
            np.asarray(vertices), np.asarray(faces, dtype=int), process=False
        )
        face_normals = mesh.face_normals
        new_points = []
        for fi, face in enumerate(mesh.faces):
            fn = face_normals[fi]
            for vi in face:
                new_points.append(mesh.vertices[vi] + fn * margin)
        new_points = np.array(new_points)
        hull = ConvexHull(new_points)
        hull_mesh = trimesh.Trimesh(
            new_points[hull.vertices], process=False
        ).convex_hull
        dilated.append(
            (np.asarray(hull_mesh.vertices), np.asarray(hull_mesh.faces, dtype=int))
        )
    return dilated


def hulls(mesh: trimesh.Trimesh, thickness: float = 0.002, dilate: float = 0.002) -> list[MeshPart]:
    """Object-frame collision hulls of ``mesh``: thicken -> CoACD -> dilate."""
    mesh = thicken_mesh(mesh, thickness)
    parts = coacd_convex_decomp(mesh, threshold=0.05, max_convex_hull=32, max_ch_vertex=256, preprocess_resolution=100)
    return dilate_hulls(parts, margin=dilate) if parts else parts
