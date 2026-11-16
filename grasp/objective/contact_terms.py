"""Hand-object contact terms of the OakInk2 objective, subtracted from every step reward.

* penetration: ``weight * max(d - margin, 0)^2`` with ``d`` the deepest active hand-object contact of the world
  (``max(-dist, 0)`` over contacts between a ``collision_hand_*`` geom and a ``{right,left}_object_*`` collision geom);
* contact mismatch: ``mismatch_weight`` per object side and step when the reference gate is on and the world has no
  touching (``dist <= 0``) contact of the required kind -- ``gate_inhand``: no hand contact on that side's object;
  ``gate_rest``: no floor contact on it.  The gates come from the reference (``grasp.pipeline.oakink2_contact_gates``),
  one row per kinematic-reference row.

The per-world quantities are computed by Warp kernels captured in the same CUDA graph as ``mujoco_warp.step``, so they
cost no host synchronisation.  Every ``check_every``-th evaluated step they are recomputed in torch from the same contact
buffer and must agree bit for bit, otherwise the run stops.
"""

from __future__ import annotations

import mujoco
import mujoco_warp
import numpy as np
import torch
import warp as wp


@wp.kernel
def _pen_zero(deepest: wp.array(dtype=float), pen: wp.array(dtype=float)):
    i = wp.tid()
    deepest[i] = 0.0
    pen[i] = 0.0


@wp.kernel
def _pen_depth(nacon: wp.array(dtype=int), dist: wp.array(dtype=float), geom: wp.array(dtype=wp.vec2i),
               worldid: wp.array(dtype=int), is_hand: wp.array(dtype=int), is_obj: wp.array(dtype=int), nworld: int,
               deepest: wp.array(dtype=float)):
    c = wp.tid()
    if c >= nacon[0]:
        return
    d = dist[c]
    if wp.isnan(d):
        return
    g = geom[c]
    w = worldid[c]
    if w < 0 or w >= nworld or g[0] < 0 or g[1] < 0:
        return
    ng = is_hand.shape[0]
    g1 = wp.min(g[0], ng - 1)
    g2 = wp.min(g[1], ng - 1)
    if (is_hand[g1] != 0 and is_obj[g2] != 0) or (is_obj[g1] != 0 and is_hand[g2] != 0):
        wp.atomic_max(deepest, w, wp.max(-d, 0.0))


@wp.kernel
def _pen_value(deepest: wp.array(dtype=float), weight: float, margin: float, pen: wp.array(dtype=float)):
    i = wp.tid()
    x = wp.max(deepest[i] - margin, 0.0)
    pen[i] = weight * (x * x)


# geom class: 1 hand, 2 right object, 3 left object, 4 floor, 0 other.  flags[w * 4 + j] = 1 when world w has a touching
# contact of kind j: (hand, right object), (hand, left object), (floor, right object), (floor, left object).
@wp.kernel
def _touch_zero(flags: wp.array(dtype=int)):
    flags[wp.tid()] = 0


@wp.kernel
def _touch(nacon: wp.array(dtype=int), dist: wp.array(dtype=float), geom: wp.array(dtype=wp.vec2i),
           worldid: wp.array(dtype=int), cls: wp.array(dtype=int), nworld: int, flags: wp.array(dtype=int)):
    c = wp.tid()
    if c >= nacon[0]:
        return
    d = dist[c]
    if wp.isnan(d) or d > 0.0:
        return
    g = geom[c]
    w = worldid[c]
    if w < 0 or w >= nworld or g[0] < 0 or g[1] < 0:
        return
    ng = cls.shape[0]
    a = cls[wp.min(g[0], ng - 1)]
    b = cls[wp.min(g[1], ng - 1)]
    lo = wp.min(a, b)
    hi = wp.max(a, b)
    if lo == 1 and hi == 2:
        wp.atomic_max(flags, w * 4 + 0, 1)
    if lo == 1 and hi == 3:
        wp.atomic_max(flags, w * 4 + 1, 1)
    if hi == 4 and lo == 2:
        wp.atomic_max(flags, w * 4 + 2, 1)
    if hi == 4 and lo == 3:
        wp.atomic_max(flags, w * 4 + 3, 1)


def load_gates(path, num_reference_rows: int, horizon_steps: int, sim_dt: float, ref_dt: float,
               start_frame: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-simulation-step gates ``(T, 2)`` (right, left) from a gate file with one row per kinematic-reference row:
    step ``s`` uses row ``start_frame + round(s * sim_dt / ref_dt)`` (clipped to the last row)."""
    with np.load(path) as f:
        gin, grest = np.asarray(f["gate_inhand"], bool), np.asarray(f["gate_rest"], bool)
    if gin.shape != (num_reference_rows, 2) or grest.shape != (num_reference_rows, 2):
        raise ValueError(f"{path}: gates must have one row per reference row ({num_reference_rows}, 2), "
                         f"got {gin.shape} and {grest.shape}")
    if (gin & grest).any():
        raise ValueError(f"{path}: gate_inhand and gate_rest overlap")
    rows = np.minimum(int(start_frame) + np.round(np.arange(int(horizon_steps)) * float(sim_dt) / float(ref_dt)).astype(int),
                      num_reference_rows - 1)
    return gin[rows], grest[rows]


class ContactTerms:
    """Penetration penalty and contact-mismatch term per world and step (see module docstring).

    Construction re-captures ``env.graph`` as ``mujoco_warp.step`` followed by the kernels, so every later
    ``step_env`` leaves this step's per-world values in the buffers read by ``__call__``.
    """

    def __init__(self, env, num_worlds: int, device, penetration_weight: float, penetration_margin: float,
                 mismatch_weight: float = 0.0, gate_inhand=None, gate_rest=None, check_every: int = 100):
        if penetration_weight <= 0:
            raise ValueError("the contact terms need a positive penetration weight")
        m = env.model_cpu
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        is_obj = lambda n: n.startswith(("right_object_", "left_object_")) and not n.endswith("visual")
        self.env, self.N, self.device = env, int(num_worlds), device
        self.hand = torch.tensor([n.startswith("collision_hand_") for n in names], dtype=torch.bool, device=device)
        self.obj = torch.tensor([is_obj(n) for n in names], dtype=torch.bool, device=device)
        if not bool(self.hand.any()) or not bool(self.obj.any()):
            raise RuntimeError("contact terms: no collision_hand_* / *_object_* collision geoms in the scene")
        self.weight, self.margin = float(penetration_weight), float(penetration_margin)
        self.mismatch_weight = float(mismatch_weight)
        self.check_every = int(check_every)
        self.steps = 0
        self.checked = 0
        self.gin = self.grest = None
        if self.mismatch_weight > 0:
            gin, grest = np.asarray(gate_inhand, bool), np.asarray(gate_rest, bool)
            if gin.ndim != 2 or gin.shape[1] != 2 or gin.shape != grest.shape:
                raise ValueError(f"gates must be (T, 2), got {gin.shape} and {grest.shape}")
            self.gin = torch.as_tensor(gin, dtype=torch.float32, device=device)
            self.grest = torch.as_tensor(grest, dtype=torch.float32, device=device)
            cls = np.zeros(len(names), np.int32)
            for i, n in enumerate(names):
                if n.startswith("collision_hand_"):
                    cls[i] = 1
                elif n.startswith("right_object_") and not n.endswith("visual"):
                    cls[i] = 2
                elif n.startswith("left_object_") and not n.endswith("visual"):
                    cls[i] = 3
                elif n == "floor":
                    cls[i] = 4
            if (cls == 4).sum() != 1:
                raise RuntimeError("contact mismatch: the scene needs exactly one geom named 'floor'")
            self.cls = torch.as_tensor(cls, dtype=torch.long, device=device)
        self._idx = None
        self._capture()

    def _capture(self) -> None:
        env, d, N = self.env, self.env.data_wp, self.N
        with wp.ScopedDevice(env.device):
            self._hand_wp = wp.array(self.hand.to(torch.int32).cpu().numpy(), dtype=int)
            self._obj_wp = wp.array(self.obj.to(torch.int32).cpu().numpy(), dtype=int)
            self._deep_wp = wp.zeros(N, dtype=float)
            self._pen_wp = wp.zeros(N, dtype=float)
            if self.mismatch_weight > 0:
                self._cls_wp = wp.array(self.cls.to(torch.int32).cpu().numpy(), dtype=int)
                self._flag_wp = wp.zeros(N * 4, dtype=int)
            naconmax = int(d.contact.dist.shape[0])

            def tail():
                wp.launch(_pen_zero, dim=N, inputs=[self._deep_wp, self._pen_wp])
                wp.launch(_pen_depth, dim=naconmax, inputs=[d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid,
                                                            self._hand_wp, self._obj_wp, N, self._deep_wp])
                wp.launch(_pen_value, dim=N, inputs=[self._deep_wp, self.weight, self.margin, self._pen_wp])
                if self.mismatch_weight > 0:
                    wp.launch(_touch_zero, dim=N * 4, inputs=[self._flag_wp])
                    wp.launch(_touch, dim=naconmax, inputs=[d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid,
                                                            self._cls_wp, N, self._flag_wp])

            tail()                      # load the kernels outside the capture (writes only our buffers)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                mujoco_warp.step(env.model_wp, d)
                tail()
            wp.synchronize()
            env.graph = capture.graph

    # -- torch references of the kernels (same contact buffer) -------------------------------------------------------
    def _contacts(self):
        d = self.env.data_wp
        dist = wp.to_torch(d.contact.dist).reshape(-1)
        geom = wp.to_torch(d.contact.geom).reshape(-1, 2)
        wid = wp.to_torch(d.contact.worldid).reshape(-1)
        if self._idx is None or self._idx.shape[0] != dist.shape[0]:
            self._idx = torch.arange(dist.shape[0], device=dist.device)
        ok = ((self._idx < wp.to_torch(d.nacon).reshape(-1)[0]) & ~torch.isnan(dist) & (wid >= 0) & (wid < self.N)
              & (geom[:, 0] >= 0) & (geom[:, 1] >= 0))
        return dist, geom, wid, ok

    def _penetration_torch(self):
        dist, geom, wid, ok = self._contacts()
        ng = self.hand.shape[0]
        g1 = geom[:, 0].long().clamp(0, ng - 1)
        g2 = geom[:, 1].long().clamp(0, ng - 1)
        sel = ok & ((self.hand[g1] & self.obj[g2]) | (self.obj[g1] & self.hand[g2]))
        depth = torch.where(sel, torch.clamp(-dist, min=0.0), torch.zeros_like(dist))
        deepest = torch.zeros(self.N, dtype=depth.dtype, device=depth.device)
        deepest.scatter_reduce_(0, wid.long().clamp(0, self.N - 1), depth, reduce="amax", include_self=True)
        return self.weight * torch.clamp(deepest - self.margin, min=0.0) ** 2, deepest

    def _touch_torch(self):
        dist, geom, wid, ok = self._contacts()
        ok = ok & (dist <= 0)
        ng = self.cls.shape[0]
        a = self.cls[geom[:, 0].long().clamp(0, ng - 1)]
        b = self.cls[geom[:, 1].long().clamp(0, ng - 1)]
        lo, hi = torch.minimum(a, b), torch.maximum(a, b)
        out = torch.zeros(self.N * 4, device=dist.device)
        for j, (l_, h_) in enumerate(((1, 2), (1, 3), (2, 4), (3, 4))):
            sel = ok & (lo == l_) & (hi == h_)
            out.scatter_reduce_(0, wid.long().clamp(0, self.N - 1) * 4 + j, sel.float(), reduce="amax", include_self=True)
        return out.view(self.N, 4)

    # -- per-step values -------------------------------------------------------------------------------------------------
    def penetration(self) -> torch.Tensor:
        """(N,) penetration penalty of the last simulated step."""
        pen = wp.to_torch(self._pen_wp)
        deep = wp.to_torch(self._deep_wp)
        self.steps += 1
        if self.check_every > 0 and self.steps % self.check_every == 0:
            pen_t, deep_t = self._penetration_torch()
            if not (torch.equal(pen, pen_t) and torch.equal(deep, deep_t)):
                bad = int(((pen != pen_t) | (deep != deep_t)).sum())
                raise RuntimeError(f"penetration penalty: graph kernel and torch reference differ in {bad} of {self.N} "
                                   f"worlds at checked step {self.steps}")
            self.checked += 1
        self.last_depth = deep
        return pen

    def mismatch(self, s: int) -> torch.Tensor:
        """(N,) contact-mismatch term of the last simulated step, gates of reference step ``s``."""
        f = wp.to_torch(self._flag_wp).view(self.N, 4).float()
        if self.check_every > 0 and self.steps % self.check_every == 0 and not torch.equal(f, self._touch_torch()):
            raise RuntimeError("contact mismatch: graph kernel and torch reference differ")
        k = min(int(s), self.gin.shape[0] - 1)
        gi, gr = self.gin[k], self.grest[k]
        miss = gi[0] * (1 - f[:, 0]) + gi[1] * (1 - f[:, 1]) + gr[0] * (1 - f[:, 2]) + gr[1] * (1 - f[:, 3])
        return self.mismatch_weight * miss

    def __call__(self, s: int) -> torch.Tensor:
        """(N,) total contact cost of the step just simulated (reference step ``s``)."""
        cost = self.penetration()
        if self.mismatch_weight > 0:
            cost = cost + self.mismatch(s)
        return cost

    def summary(self) -> dict:
        return dict(penetration_weight=self.weight, penetration_margin=self.margin, mismatch_weight=self.mismatch_weight,
                    hand_geoms=int(self.hand.sum()), object_geoms=int(self.obj.sum()), check_every=self.check_every,
                    gate_inhand_frac=None if self.gin is None else [float(x) for x in self.gin.mean(0)],
                    gate_rest_frac=None if self.grest is None else [float(x) for x in self.grest.mean(0)])
