"""Acrobot (underactuated double pendulum) with RK4 discretization.

This is a pure-analytical acrobot dynamics model implemented in JAX-friendly code,
meant to be easy to mirror in CasADi for IPOPT single-shooting comparisons.

State: x = [theta1, theta2, dtheta1, dtheta2]
  - theta1: angle of link 1 from vertical (down = 0)
  - theta2: relative angle of link 2 w.r.t link 1
Control: u (torque at elbow)
Goal: upright swing-up (theta1 = pi, theta2 = 0, velocities 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import jax.numpy as jnp
import numpy as np

from .base import DynamicalSystem, SystemParams, CostWeights


def wrap_angle(x):
    """Wrap angle to [-pi, pi] (smooth)."""
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


@dataclass
class AcrobotRK4Params(SystemParams):
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    lc1: float = 0.5
    lc2: float = 0.5
    I1: Optional[float] = None
    I2: Optional[float] = None
    g: float = 9.81
    b1: float = 0.1
    b2: float = 0.1
    u_max: float = 5.0

    def __post_init__(self):
        if self.I1 is None:
            self.I1 = self.m1 * self.l1**2 / 12.0
        if self.I2 is None:
            self.I2 = self.m2 * self.l2**2 / 12.0


@dataclass
class AcrobotRK4CostWeights(CostWeights):
    # Running cost: keep angle tracking modest, but avoid strongly penalizing
    # velocities/torque (the acrobot needs momentum to swing up).
    w_theta1: float = 2.0
    w_theta2: float = 1.0
    w_dtheta1: float = 0.01
    w_dtheta2: float = 0.01
    w_u: float = 0.005

    # Terminal cost: strong upright + near-zero velocity.
    w_theta1_T: float = 80.0
    w_theta2_T: float = 80.0
    w_dtheta1_T: float = 25.0
    w_dtheta2_T: float = 25.0

    theta1_goal: float = field(default_factory=lambda: float(np.pi))
    theta2_goal: float = 0.0


class AcrobotRK4System(DynamicalSystem):
    """Acrobot dynamics with RK4 integration."""

    def __init__(
        self,
        params: Optional[AcrobotRK4Params] = None,
        cost_weights: Optional[AcrobotRK4CostWeights] = None,
        dt: float = 0.02,
    ):
        if params is None:
            params = AcrobotRK4Params()
        if cost_weights is None:
            cost_weights = AcrobotRK4CostWeights()
        super().__init__(params, cost_weights, dt)
        self._params: AcrobotRK4Params = params
        self._weights: AcrobotRK4CostWeights = cost_weights

    @property
    def name(self) -> str:
        return "acrobot_rk4"

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def control_dim(self) -> int:
        return 1

    @property
    def default_initial_state(self) -> jnp.ndarray:
        return jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    @property
    def u_max(self) -> float:
        return self._params.u_max

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time dynamics: xdot = f(x,u)."""
        P = self._params
        theta1, theta2, dtheta1, dtheta2 = x
        u_scalar = u if u.ndim == 0 else u[0]

        m1, m2, l1, l2, lc1, lc2 = P.m1, P.m2, P.l1, P.l2, P.lc1, P.lc2
        I1, I2, g = P.I1, P.I2, P.g

        s2 = jnp.sin(theta2)
        c2 = jnp.cos(theta2)
        s1 = jnp.sin(theta1)
        s12 = jnp.sin(theta1 + theta2)

        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * c2) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * c2) + I2
        d3 = m2 * lc2**2 + I2

        phi2 = m2 * l1 * lc2 * s2
        h1 = -phi2 * (2.0 * dtheta1 * dtheta2 + dtheta2**2)
        h2 = phi2 * dtheta1**2

        phi1 = (m1 * lc1 + m2 * l1) * g * s1 + m2 * lc2 * g * s12
        phi3 = m2 * lc2 * g * s12

        tau1 = -P.b1 * dtheta1
        tau2 = u_scalar - P.b2 * dtheta2

        det_M = d1 * d3 - d2 * d2
        det_M = jnp.maximum(det_M, 1e-8)

        rhs1 = tau1 - h1 - phi1
        rhs2 = tau2 - h2 - phi3

        ddtheta1 = (d3 * rhs1 - d2 * rhs2) / det_M
        ddtheta2 = (-d2 * rhs1 + d1 * rhs2) / det_M

        return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2], dtype=x.dtype)

    def step(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Discrete RK4 step."""
        # controls passed to step() are expected to already be clamped by trajectory_cost_from_raw/rollout_best
        u_scalar = u if u.ndim == 0 else u[0]
        u_scalar = jnp.clip(u_scalar, -self.u_max, self.u_max)  # bounded dynamics for safety

        def f(xi):
            return self.dynamics(xi, u_scalar)

        dt = self.dt
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Wrap angles for numerical stability / bounded representation
        return jnp.array([wrap_angle(x_next[0]), wrap_angle(x_next[1]), x_next[2], x_next[3]], dtype=x.dtype)

    def stage_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        W = self._weights
        u_scalar = u if u.ndim == 0 else u[0]
        e1 = wrap_angle(x[0] - W.theta1_goal)
        e2 = wrap_angle(x[1] - W.theta2_goal)
        return (
            W.w_theta1 * e1**2
            + W.w_theta2 * e2**2
            + W.w_dtheta1 * x[2] ** 2
            + W.w_dtheta2 * x[3] ** 2
            + W.w_u * u_scalar**2
        )

    def terminal_cost(self, x: jnp.ndarray) -> float:
        W = self._weights
        e1 = wrap_angle(x[0] - W.theta1_goal)
        e2 = wrap_angle(x[1] - W.theta2_goal)
        return (
            W.w_theta1_T * e1**2
            + W.w_theta2_T * e2**2
            + W.w_dtheta1_T * x[2] ** 2
            + W.w_dtheta2_T * x[3] ** 2
        )

    def save_animation(self, filename: str, X: np.ndarray, U: np.ndarray, title: str = "", dpi: int = 150, fps: Optional[int] = None):
        """Save acrobot animation as MP4 (or GIF fallback)."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        X = np.asarray(X)
        U = np.asarray(U)
        l1, l2 = float(self._params.l1), float(self._params.l2)
        fps = fps or max(1, int(round(1.0 / self.dt)))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal", "box")
        m = 1.3 * (l1 + l2)
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12)

        link1, = ax.plot([], [], lw=4, color="C0")
        link2, = ax.plot([], [], lw=4, color="C1")
        joint1, = ax.plot([0], [0], "ko", ms=6)  # shoulder
        joint2, = ax.plot([], [], "ko", ms=5)  # elbow
        end_eff, = ax.plot([], [], "o", ms=8, color="C3")  # end effector

        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)
        torque_text = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=10)

        def init():
            link1.set_data([], [])
            link2.set_data([], [])
            joint2.set_data([], [])
            end_eff.set_data([], [])
            time_text.set_text("")
            torque_text.set_text("")
            return link1, link2, joint1, joint2, end_eff, time_text, torque_text

        def update(i):
            theta1 = float(X[i, 0])
            theta2 = float(X[i, 1])

            x1 = l1 * math.sin(theta1)
            y1 = -l1 * math.cos(theta1)

            x2 = x1 + l2 * math.sin(theta1 + theta2)
            y2 = y1 - l2 * math.cos(theta1 + theta2)

            link1.set_data([0.0, x1], [0.0, y1])
            link2.set_data([x1, x2], [y1, y2])
            joint2.set_data([x1], [y1])
            end_eff.set_data([x2], [y2])
            time_text.set_text(f"t = {i*self.dt:.2f}s")

            u_val = U[i] if i < len(U) else 0
            if hasattr(u_val, "__len__"):
                u_val = u_val[0] if len(u_val) > 0 else 0
            torque_text.set_text(f"u = {float(u_val):+.2f} Nm" if i < len(U) else "u = —")

            return link1, link2, joint1, joint2, end_eff, time_text, torque_text

        ani = FuncAnimation(fig, update, frames=len(X), init_func=init, blit=True)

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


