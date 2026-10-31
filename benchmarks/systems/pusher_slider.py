"""Quasi-static (QS) planar pusher-slider with explicit contact "mode" controls (JAX-only).

This variant removes the soft-contact detection used in `pusher_slider_qs_box.py`.
Instead, each control step *explicitly specifies* a contact:

Control (4D) per step:
  u = [edge_mode, s_along_edge, v_local_x, v_local_y]

Where:
  - edge_mode in {1,2,3,4} selects which box edge is contacted:
      1: +x (right), 2: -x (left), 3: +y (top), 4: -y (bottom) in the BOX frame.
    For gradient-based methods, we optionally use a smooth soft-assignment around these modes.
  - s_along_edge in [-1,1] selects the contact point along the chosen edge.
  - v_local is the pusher velocity in the BOX frame (m/s), clamped to u_max.

State (3D):
  x = [b_x, b_y, theta]  (box pose in world frame)

Visualization renders the pusher center as exactly tangent to the box surface at the chosen
contact point (no gaps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import jax
import jax.numpy as jnp
import numpy as np

from .base import DynamicalSystem, SystemParams, CostWeights


def _wrap_angle(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def _rot2(theta: jnp.ndarray) -> jnp.ndarray:
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.float32)


@dataclass
class PusherSliderQSContactModeParams(SystemParams):
    dt: float = 0.02

    # Box geometry
    box_w: float = 0.3
    box_h: float = 0.2

    # Pusher
    pusher_radius: float = 0.05
    u_max: float = 1.0  # max speed (m/s)

    # Contact location constraint: s in [-s_max_frac, s_max_frac]
    # where s parameterizes the half-edge length (s=1 reaches the corner).
    s_max_frac: float = 0.9

    # QS "dissipation metric": penalize angular speed relative to translation
    use_geom_rg: bool = True
    rg_override: float = 0.15  # meters (used if use_geom_rg=False)
    max_omega: float = 30.0

    # How to interpret edge_mode
    # Edge selection mode:
    # - "mixture": use a convex combination of the 4 edges (differentiable, but not physical).
    # - "soft_gumbel": fixed-noise Gumbel-Softmax weights in both the forward
    #   and backward passes (smooth and deterministic for a given timestep).
    # - "st_argmax": choose a single edge via argmax(soft weights) with straight-through gradients.
    # - "st_gumbel": choose a single edge via Gumbel-Softmax sampling (fixed per-timestep noise) with straight-through gradients.
    edge_selection: str = "st_argmax"
    # RBF temperature used to turn the continuous selector into logits/weights over {1,2,3,4}.
    edge_temp: float = 0.15  # smaller => sharper preference for the nearest integer edge
    # Gumbel-Softmax temperature (st_gumbel only): smaller => more one-hot.
    gumbel_tau: float = 0.3
    # Seed for fixed per-timestep Gumbel noise (st_gumbel only).
    gumbel_seed: int = 0

    # Push direction constraint (in box frame):
    # velocity must be within +/- max_push_angle_deg of the inward edge normal direction.
    # Example: 60deg allows significant tangential pushing but disallows near-tangential "scraping".
    max_push_angle_deg: float = 60.0


@dataclass
class PusherSliderQSContactModeCostWeights(CostWeights):
    # Goal (box pose)
    goal_x: float = 0.6
    goal_y: float = 0.0
    goal_theta: float = float(np.pi)

    # Weights
    w_pos: float = 10.0
    w_theta: float = 1.0
    w_v: float = 0.01  # penalize pusher local velocity magnitude (m/s)^2

    w_pos_T: float = 50.0
    w_theta_T: float = 10.0

    # Small shaping: discourage rapid mode switches (optional)
    w_mode_switch: float = 0.0
    # Smoothness on contact parameter and pushing velocity (optional)
    w_s_switch: float = 0.0
    w_v_switch: float = 0.0


class PusherSliderQSContactModeSystem(DynamicalSystem):
    """QS pusher-slider with explicit per-step contact mode and contact location."""

    def __init__(
        self,
        params: Optional[PusherSliderQSContactModeParams] = None,
        cost_weights: Optional[PusherSliderQSContactModeCostWeights] = None,
        *,
        box_init: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        if params is None:
            params = PusherSliderQSContactModeParams()
        if cost_weights is None:
            cost_weights = PusherSliderQSContactModeCostWeights()
        super().__init__(params, cost_weights, dt=float(params.dt))
        self._P = params
        self._W = cost_weights
        self._x0 = jnp.array([float(box_init[0]), float(box_init[1]), float(box_init[2])], dtype=jnp.float32)

    @property
    def name(self) -> str:
        return "pusher_slider_qs_contact_mode"

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def control_dim(self) -> int:
        # [edge_mode, s_along_edge, v_local_x, v_local_y]
        return 4

    @property
    def default_initial_state(self) -> jnp.ndarray:
        return self._x0

    @property
    def u_max(self) -> float:
        return float(self._P.u_max)

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Use step() for discrete-time PusherSliderQSContactMode.")

    # ------------------------------------------------------------------
    # Contact geometry from control
    # ------------------------------------------------------------------
    def _edge_logits_and_soft_weights(self, edge_mode_cont: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (logits, soft_weights) over the 4 edges from a continuous selector in [1,4]."""
        P = self._P
        temp = jnp.maximum(jnp.asarray(float(P.edge_temp), dtype=jnp.float32), 1e-4)
        ks = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        logits = -((edge_mode_cont - ks) / temp) ** 2
        w = jax.nn.softmax(logits)
        return logits, w

    def _edge_weights(self, edge_mode_cont: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Return weights over 4 edges according to `params.edge_selection`.

        - "mixture": weights are soft and sum to 1 (not physically correct, but smooth).
        - "soft_gumbel": fixed-noise Gumbel-Softmax weights are used in both passes.
        - "st_argmax": forward selects one edge (argmax), backward uses soft weights gradients.
        - "st_gumbel": forward selects one edge using fixed per-timestep Gumbel noise, backward uses soft weights gradients.
        """
        P = self._P
        logits, w_soft = self._edge_logits_and_soft_weights(edge_mode_cont)

        mode = str(P.edge_selection).lower()
        if mode == "mixture":
            return w_soft

        if mode == "soft_gumbel":
            tau = jnp.maximum(
                jnp.asarray(float(P.gumbel_tau), dtype=jnp.float32), 1e-4
            )
            key = jax.random.PRNGKey(int(P.gumbel_seed))
            key = jax.random.fold_in(key, jnp.asarray(t, dtype=jnp.int32))
            uniform = jax.random.uniform(
                key,
                shape=(4,),
                minval=1e-6,
                maxval=1.0 - 1e-6,
                dtype=jnp.float32,
            )
            gumbel = -jnp.log(-jnp.log(uniform))
            return jax.nn.softmax((logits + gumbel) / tau)

        if mode == "st_argmax":
            k = jnp.argmax(w_soft)
            w_hard = jax.nn.one_hot(k, 4, dtype=jnp.float32)
            return jax.lax.stop_gradient(w_hard - w_soft) + w_soft

        if mode == "st_gumbel":
            # Fixed per-timestep Gumbel noise: deterministic for a given t.
            # Note: This makes the log-likelihood deterministic (important for SMC/MCMC),
            # but uses common random numbers shared across particles.
            tau = jnp.maximum(jnp.asarray(float(P.gumbel_tau), dtype=jnp.float32), 1e-4)
            key = jax.random.PRNGKey(int(P.gumbel_seed))
            key = jax.random.fold_in(key, jnp.asarray(t, dtype=jnp.int32))
            u = jax.random.uniform(key, shape=(4,), minval=1e-6, maxval=1.0 - 1e-6, dtype=jnp.float32)
            g = -jnp.log(-jnp.log(u))
            y = jax.nn.softmax((logits + g) / tau)
            k = jnp.argmax(y)
            w_hard = jax.nn.one_hot(k, 4, dtype=jnp.float32)
            return jax.lax.stop_gradient(w_hard - y) + y

        raise ValueError(
            f"Unknown edge_selection={P.edge_selection!r}. Use 'mixture', "
            "'soft_gumbel', 'st_argmax', or 'st_gumbel'."
        )

    def _contact_point_and_normal_local(self, edge_mode_cont: jnp.ndarray, s: jnp.ndarray, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (contact_point_local, outward_normal_local) in BOX frame."""
        P = self._P
        w = float(P.box_w)
        h = float(P.box_h)

        # Clamp s to [-s_max_frac, s_max_frac] (already clamped in control, but keep safe).
        smax = float(P.s_max_frac)
        smax = float(np.clip(smax, 0.0, 1.0))
        s = jnp.clip(s, -smax, smax)

        # Contact points for each edge (in box frame).
        cp1 = jnp.array([+w / 2.0, s * h / 2.0], dtype=jnp.float32)  # right (+x)
        cp2 = jnp.array([-w / 2.0, s * h / 2.0], dtype=jnp.float32)  # left (-x)
        cp3 = jnp.array([s * w / 2.0, +h / 2.0], dtype=jnp.float32)  # top (+y)
        cp4 = jnp.array([s * w / 2.0, -h / 2.0], dtype=jnp.float32)  # bottom (-y)

        n1 = jnp.array([+1.0, 0.0], dtype=jnp.float32)
        n2 = jnp.array([-1.0, 0.0], dtype=jnp.float32)
        n3 = jnp.array([0.0, +1.0], dtype=jnp.float32)
        n4 = jnp.array([0.0, -1.0], dtype=jnp.float32)

        w4 = self._edge_weights(edge_mode_cont, t)  # (4,)
        cps = jnp.stack([cp1, cp2, cp3, cp4], axis=0)  # (4,2)
        ns = jnp.stack([n1, n2, n3, n4], axis=0)  # (4,2)

        cp = jnp.sum(w4[:, None] * cps, axis=0)
        n = jnp.sum(w4[:, None] * ns, axis=0)
        n = n / (jnp.linalg.norm(n) + 1e-8)
        return cp, n

    def _project_v_local_push_cone(self, edge_mode: jnp.ndarray, s: jnp.ndarray, v_local: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Project local pusher velocity into a feasible "push-only cone".

        Constraints (in box frame):
        - No pulling: v · n_out <= 0, where n_out is the outward edge normal.
        - Direction cone: velocity must lie within +/- max_push_angle_deg of the inward normal (-n_out).
          Equivalently, let v_n = v·n_out (<=0) and v_t be tangential component. Then:
              ||v_t|| <= tan(max_angle) * (-v_n)
          If v_n >= 0 (pulling), we set v=0 (no contact motion).
        """
        P = self._P
        _cp_local, n_out = self._contact_point_and_normal_local(edge_mode, s, t)
        # Inward normal
        n_in = -n_out
        # Tangent (90deg rotation)
        t_hat = jnp.array([-n_out[1], n_out[0]], dtype=jnp.float32)

        v_n_out = jnp.dot(v_local, n_out)  # outward-normal component (scalar)
        v_in = jnp.dot(v_local, n_in)      # inward-normal component (scalar) = -v_n_out
        v_t = jnp.dot(v_local, t_hat)      # signed tangential component (scalar)

        # Enforce push-only: if not pushing inward, no velocity.
        v_in_pos = jax.nn.relu(v_in)  # >=0, equals v_in when v_in>0 else 0
        # If v_in <= 0 => v_in_pos=0 => everything becomes 0.

        # Enforce cone: |v_t| <= tan(angle) * v_in_pos
        ang = jnp.deg2rad(jnp.asarray(float(P.max_push_angle_deg), dtype=jnp.float32))
        tan_max = jnp.tan(jnp.clip(ang, 0.0, 1.55))  # cap just below pi/2
        max_t = tan_max * v_in_pos
        v_t_clamped = jnp.clip(v_t, -max_t, max_t)

        v_proj = v_in_pos * n_in + v_t_clamped * t_hat
        return v_proj

    def _pusher_center_world(self, b_xy: jnp.ndarray, theta: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (pusher_center_world, contact_point_world, outward_normal_world)."""
        P = self._P
        edge_mode = jnp.clip(u[0], 1.0, 4.0)
        s = u[1]
        cp_local, n_local = self._contact_point_and_normal_local(edge_mode, s, t)
        R = _rot2(theta)
        cp_world = b_xy + R @ cp_local
        n_world = R @ n_local
        p_world = cp_world + float(P.pusher_radius) * n_world
        return p_world, cp_world, n_world

    # ------------------------------------------------------------------
    # QS contact kinematics
    # ------------------------------------------------------------------
    def _qs_twist_from_contact_velocity(self, r_world: jnp.ndarray, v_contact_world: jnp.ndarray) -> jnp.ndarray:
        """Solve for box twist V=[vx,vy,omega] minimizing 0.5*V^T A V s.t. J V = v_contact."""
        P = self._P
        rx, ry = r_world[0], r_world[1]
        # Contact Jacobian mapping twist -> contact point velocity in world
        # v_c = [vx - omega*ry, vy + omega*rx]
        J = jnp.array([[1.0, 0.0, -ry], [0.0, 1.0, rx]], dtype=jnp.float32)

        if bool(P.use_geom_rg):
            w = float(P.box_w)
            h = float(P.box_h)
            rg2 = float((w * w + h * h) / 12.0)
        else:
            rg2 = float(P.rg_override) ** 2
        # A = diag(1,1,rg2) => A^{-1} = diag(1,1,1/rg2)
        Ainv = jnp.array([1.0, 1.0, 1.0 / max(rg2, 1e-6)], dtype=jnp.float32)

        JT = jnp.transpose(J)  # (3,2)
        Ainv_JT = (Ainv[:, None] * JT)
        M = J @ Ainv_JT
        Minv = jnp.linalg.inv(M + 1e-6 * jnp.eye(2, dtype=jnp.float32))
        V = Ainv_JT @ (Minv @ v_contact_world)
        V = V.at[2].set(jnp.clip(V[2], -float(P.max_omega), float(P.max_omega)))
        return V

    # ------------------------------------------------------------------
    # Control clamping
    # ------------------------------------------------------------------
    def clamp_control(self, u_raw: jnp.ndarray) -> jnp.ndarray:
        """Clamp to the contact-mode control manifold.

        - edge_mode is mapped to [1,4] (still continuous; may be rounded in dynamics if hard mode is enabled).
        - s is mapped to [-1,1]
        - v_local is mapped to [-u_max, u_max] per component
        """
        P = self._P
        u_raw = jnp.asarray(u_raw, dtype=jnp.float32)

        # Support broadcasting over time: (...,4)
        edge = 2.5 + 1.5 * jnp.tanh(u_raw[..., 0])  # in [1,4]
        smax = float(np.clip(float(P.s_max_frac), 0.0, 1.0))
        s = smax * jnp.tanh(u_raw[..., 1])  # in [-smax, smax]
        v = float(P.u_max) * jnp.tanh(u_raw[..., 2:4])  # (..,2)
        return jnp.concatenate([edge[..., None], s[..., None], v], axis=-1)

    def _clamp_physical_control(self, u: jnp.ndarray) -> jnp.ndarray:
        """Clamp a control assumed to already be in physical parameterization.

        This is a safety clamp used by `step`, `stage_cost`, and visualization so that:
        - Edge is within [1,4]
        - s is within [-1,1]
        - v_local is within [-u_max, u_max]
        """
        P = self._P
        u = jnp.asarray(u, dtype=jnp.float32)
        edge = jnp.clip(u[0], 1.0, 4.0)
        smax = float(np.clip(float(P.s_max_frac), 0.0, 1.0))
        s = jnp.clip(u[1], -smax, smax)
        v = jnp.clip(u[2:4], -float(P.u_max), float(P.u_max))
        return jnp.array([edge, s, v[0], v[1]], dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Dynamics + cost
    # ------------------------------------------------------------------
    def _step_with_t(self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        P = self._P
        dt = float(P.dt)

        x = jnp.asarray(x, dtype=jnp.float32)
        # `step` consumes *physical/clamped* controls (like other systems).
        # Raw/unconstrained controls should be passed through `trajectory_cost_from_raw`,
        # which calls `clamp_control` before evaluating costs/rollouts.
        u_cmd = self._clamp_physical_control(u)

        b_xy = x[0:2]
        th = x[2]

        # Contact point and pusher center (for kinematics)
        _p_world, cp_world, _n_world = self._pusher_center_world(b_xy, th, u_cmd, t)
        r_world = cp_world - b_xy

        # Pusher velocity in world = R(theta) * v_local (push-only constraint enforced in box frame)
        edge_mode = jnp.clip(u_cmd[0], 1.0, 4.0)
        s = u_cmd[1]
        v_local = self._project_v_local_push_cone(edge_mode, s, u_cmd[2:4], t)
        R = _rot2(th)
        v_world = R @ v_local

        # QS twist and integrate box pose
        V = self._qs_twist_from_contact_velocity(r_world, v_world)
        b_xy_next = b_xy + dt * V[0:2]
        th_next = _wrap_angle(th + dt * V[2])

        return jnp.array([b_xy_next[0], b_xy_next[1], th_next], dtype=jnp.float32)

    def step(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        # Default: treat as t=0 if called directly.
        return self._step_with_t(x, u, jnp.asarray(0, dtype=jnp.int32))

    def stage_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        W = self._W
        x = jnp.asarray(x, dtype=jnp.float32)
        u_cmd = self._clamp_physical_control(u)

        b_xy = x[0:2]
        th = x[2]
        goal = jnp.array([float(W.goal_x), float(W.goal_y)], dtype=jnp.float32)
        pos_err = b_xy - goal
        th_err = _wrap_angle(th - float(W.goal_theta))

        # Use projected velocity for effort penalty (matches dynamics constraint).
        v_local = self._project_v_local_push_cone(u_cmd[0], u_cmd[1], u_cmd[2:4], jnp.asarray(0, dtype=jnp.int32))
        return (
            float(W.w_pos) * jnp.sum(pos_err * pos_err)
            + float(W.w_theta) * (th_err * th_err)
            + float(W.w_v) * jnp.sum(v_local * v_local)
        )

    def terminal_cost(self, x: jnp.ndarray) -> float:
        W = self._W
        x = jnp.asarray(x, dtype=jnp.float32)
        b_xy = x[0:2]
        th = x[2]
        goal = jnp.array([float(W.goal_x), float(W.goal_y)], dtype=jnp.float32)
        pos_err = b_xy - goal
        th_err = _wrap_angle(th - float(W.goal_theta))
        return float(W.w_pos_T) * jnp.sum(pos_err * pos_err) + float(W.w_theta_T) * (th_err * th_err)

    def trajectory_cost(self, x0: jnp.ndarray, U: jnp.ndarray) -> float:
        """Total cost with optional penalty on mode switching."""
        W = self._W
        U = jnp.asarray(U, dtype=jnp.float32)
        # Override base rollout so we can pass timestep index into edge selection (for st_gumbel).
        def step_cost_fn(carry, inp):
            x = carry
            u, t = inp
            c = self.stage_cost(x, u)
            x_next = self._step_with_t(x, u, t)
            return x_next, c

        H = int(U.shape[0])
        ts = jnp.arange(H, dtype=jnp.int32)
        xT, costs = jax.lax.scan(step_cost_fn, jnp.asarray(x0, dtype=jnp.float32), (U, ts))
        base = jnp.sum(costs) + self.terminal_cost(xT)
        w_sw = float(W.w_mode_switch)
        if w_sw != 0.0:
            # Penalize changes in the (continuous) edge selector.
            du_mode = U[1:, 0] - U[:-1, 0]
            base = base + w_sw * jnp.sum(du_mode * du_mode)

        # Penalize switches in along-edge contact coordinate s.
        w_s = float(W.w_s_switch)
        if w_s != 0.0:
            ds = U[1:, 1] - U[:-1, 1]
            base = base + w_s * jnp.sum(ds * ds)

        # Penalize switches in the *push-projected* local pushing velocity.
        w_v = float(W.w_v_switch)
        if w_v != 0.0:
            def proj_v(u, t):
                u_cmd = self._clamp_physical_control(u)
                return self._project_v_local_push_cone(u_cmd[0], u_cmd[1], u_cmd[2:4], t)

            H = int(U.shape[0])
            ts = jnp.arange(H, dtype=jnp.int32)
            Vloc = jax.vmap(proj_v)(U, ts)  # (H,2)
            dV = Vloc[1:] - Vloc[:-1]
            base = base + w_v * jnp.sum(dV * dV)
        return base

    def switch_cost(self, u_prev: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Pairwise switch penalty between consecutive *clamped* controls.

        ``u_prev`` is the control applied at ``t - 1`` and ``u`` the control at
        ``t`` (both already passed through ``clamp_control``).  Summing this
        over consecutive pairs reproduces the three switch terms added in
        ``trajectory_cost`` exactly, so a trajectory cost can be accumulated
        segment by segment (used by the growing-window sampler).
        """
        W = self._W
        u_prev = jnp.asarray(u_prev, dtype=jnp.float32)
        u = jnp.asarray(u, dtype=jnp.float32)
        total = jnp.asarray(0.0, dtype=jnp.float32)
        w_sw = float(W.w_mode_switch)
        if w_sw != 0.0:
            total = total + w_sw * (u[0] - u_prev[0]) ** 2
        w_s = float(W.w_s_switch)
        if w_s != 0.0:
            total = total + w_s * (u[1] - u_prev[1]) ** 2
        w_v = float(W.w_v_switch)
        if w_v != 0.0:
            def proj_v(control, time_index):
                u_cmd = self._clamp_physical_control(control)
                return self._project_v_local_push_cone(u_cmd[0], u_cmd[1], u_cmd[2:4], time_index)

            t = jnp.asarray(t, dtype=jnp.int32)
            dV = proj_v(u, t) - proj_v(u_prev, t - 1)
            total = total + w_v * jnp.sum(dV * dV)
        return total

    def rollout(self, x0: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        """Roll out trajectory with timestep-aware edge selection."""
        x0 = jnp.asarray(x0, dtype=jnp.float32)
        U = jnp.asarray(U, dtype=jnp.float32)
        H = int(U.shape[0])
        ts = jnp.arange(H, dtype=jnp.int32)

        def step_fn(x, inp):
            u, t = inp
            x_next = self._step_with_t(x, u, t)
            return x_next, x_next

        _, Xs = jax.lax.scan(step_fn, x0, (U, ts))
        return jnp.vstack([x0[None, :], Xs])

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def save_animation(
        self,
        filename: str,
        X: np.ndarray,
        U: np.ndarray,
        title: str = "",
        dpi: int = 150,
        fps: Optional[int] = None,
        arrow_scale: float = 0.15,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from matplotlib.patches import Circle, Polygon

        X = np.asarray(X)
        U = np.asarray(U)
        fps = fps or max(1, int(round(1.0 / float(self.dt))))

        box_size = (float(self._P.box_w), float(self._P.box_h))
        goal_xy = np.array([float(self._W.goal_x), float(self._W.goal_y)], dtype=np.float64)
        goal_th = float(self._W.goal_theta)

        # Precompute pusher centers for all frames (T = len(X)).
        T = int(X.shape[0])
        if U.shape[0] < max(1, T - 1):
            raise ValueError(f"U must have at least T-1 controls. Got U={U.shape}, X={X.shape}")

        def rect_corners(center_xy: tuple[float, float], size: tuple[float, float], theta: float) -> np.ndarray:
            cx, cy = float(center_xy[0]), float(center_xy[1])
            w, h = float(size[0]), float(size[1])
            local = np.array(
                [
                    [-w / 2, -h / 2],
                    [w / 2, -h / 2],
                    [w / 2, h / 2],
                    [-w / 2, h / 2],
                ],
                dtype=np.float64,
            )
            c = math.cos(theta)
            s = math.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float64)
            world = local @ R.T
            world[:, 0] += cx
            world[:, 1] += cy
            return world

        @jax.jit
        def pusher_center_and_v_world_for_frame(x, u, t):
            b_xy = x[0:2]
            th = x[2]
            u_cmd = self._clamp_physical_control(u)
            p_world, _cp, _n = self._pusher_center_world(b_xy, th, u_cmd, t)

            # Push-only velocity in box frame, then rotate to world.
            v_local = self._project_v_local_push_cone(u_cmd[0], u_cmd[1], u_cmd[2:4], t)
            v_world = _rot2(th) @ v_local
            return p_world, v_world

        # Frames 0..T-2 use U[i]; last frame reuses U[T-2].
        U_pad = np.concatenate([U[: T - 1], U[(T - 2) : (T - 1)]], axis=0) if T >= 2 else U[:1]
        # Use timestep index consistent with the control used for that frame.
        t_for_u = jnp.minimum(jnp.arange(T, dtype=jnp.int32), jnp.asarray(max(0, T - 2), dtype=jnp.int32))
        P_world, V_world = jax.vmap(pusher_center_and_v_world_for_frame)(jnp.asarray(X), jnp.asarray(U_pad), t_for_u)
        P_world = np.array(P_world)
        V_world = np.array(V_world)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.set_aspect("equal", "box")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)

        goal_poly = Polygon(rect_corners((goal_xy[0], goal_xy[1]), box_size, goal_th), closed=True, color=(0.65, 0.95, 0.65, 0.35))
        ax.add_patch(goal_poly)
        box_poly = Polygon(rect_corners((0.0, 0.0), box_size, 0.0), closed=True, color=(0.2, 0.4, 0.95, 1.0))
        ax.add_patch(box_poly)
        pusher = Circle((0, 0), radius=float(self._P.pusher_radius), color=(0.2, 0.2, 0.2, 1.0))
        ax.add_patch(pusher)
        # Arrow for pusher velocity (world-frame), showing the *push-only projected* velocity.
        vel_arrow = ax.quiver(
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=(0.9, 0.2, 0.2, 0.9),
            width=0.006,
        )

        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        def init():
            return (goal_poly, box_poly, pusher, vel_arrow, time_text)

        def update(i: int):
            bx, by = float(X[i, 0]), float(X[i, 1])
            th = float(X[i, 2])
            px, py = float(P_world[i, 0]), float(P_world[i, 1])
            vx, vy = float(V_world[i, 0]), float(V_world[i, 1])
            goal_poly.set_xy(rect_corners((goal_xy[0], goal_xy[1]), box_size, goal_th))
            box_poly.set_xy(rect_corners((bx, by), box_size, th))
            pusher.center = (px, py)
            vel_arrow.set_offsets(np.array([[px, py]], dtype=np.float64))
            vel_arrow.set_UVC(np.array([arrow_scale * vx], dtype=np.float64), np.array([arrow_scale * vy], dtype=np.float64))
            time_text.set_text(f"t = {i * float(self.dt):.2f}s")
            return (goal_poly, box_poly, pusher, vel_arrow, time_text)

        ani = FuncAnimation(fig, update, frames=len(X), init_func=init, blit=False)

        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            ani.save(filename, writer=writer, dpi=dpi)
            print(f"[Video] Saved to {filename}")
        except Exception as e:
            print(f"[Video] ffmpeg not available ({e}). Falling back to GIF...")
            from matplotlib.animation import PillowWriter

            gif_name = filename.rsplit(".", 1)[0] + ".gif"
            ani.save(gif_name, writer=PillowWriter(fps=fps))
            print(f"[Video] Saved to {gif_name}")

        plt.close(fig)
