"""Paper figure "Final-particle cost distributions" from ``benchmarks.run`` outputs.

(a) acrobot and (b) pusher-slider: final-particle cost boxplots of one seed per method, diamonds at the best particle;
(c) pusher-slider: the best particle of each method replayed, box outlines colored by time, dashed goal pose.

    python -m benchmarks.plot --results-dir outputs/benchmarks --acrobot-seed 0 --pusher-seed 9

Reads ``<results-dir>/<task>/<method>/seed<k>.npz`` for the methods grasp, rwm, tsmc_grad and mppi; a missing method
is left out of the figure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

from benchmarks.run import TASKS, build_system

INK = "#0b0b0b"
INK2 = "#52514e"
METHODS = {  # method: (label, color)
    "grasp": ("GRASP", "#2a78d6"),
    "rwm": ("RWM-TSMC", "#eb6834"),
    "tsmc_grad": ("TSMC\n(exact-grad)", "#1baf7a"),
    "mppi": ("Parallel-MPPI", "#eda100"),
}
TIME_CMAP = cm.viridis


def load(results_dir: Path, task: str, seed: int) -> dict:
    out = {}
    for method in METHODS:
        path = results_dir / task / method / f"seed{seed}.npz"
        if not path.exists():
            print(f"[{task}] no {path}; {method} left out", flush=True)
            continue
        z = np.load(path)
        costs = np.asarray(z["costs"], dtype=np.float64)
        best = int(np.argmin(costs))
        out[method] = dict(costs=costs, best_theta=np.asarray(z["samples"][best], dtype=np.float32),
                           best_cost=float(costs[best]))
    return out


def pusher_replay(system, theta: np.ndarray):
    H = TASKS["pusher"]["horizon"]
    U_raw = jnp.reshape(jnp.asarray(theta), (H, int(system.control_dim)))
    U = system.clamp_control(U_raw)
    x0 = system.default_initial_state
    X = np.asarray(system.rollout(x0, U))
    cost = float(system.trajectory_cost_from_raw(x0, U_raw))

    def pusher_center(x, u, t):
        p, _, _ = system._pusher_center_world(x[0:2], x[2], system._clamp_physical_control(u), t)
        return p

    P = np.asarray(jax.vmap(pusher_center)(jnp.asarray(X[:-1]), U, jnp.arange(H, dtype=jnp.int32)))
    return X, P, cost


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=14)


def draw_boxplot(ax, dists: dict, title: str, panel: str, ylabel: bool = True):
    keys = list(dists)
    colors = [METHODS[k][1] for k in keys]
    bp = ax.boxplot([dists[k]["costs"] for k in keys], widths=0.45, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=INK2, markeredgecolor="none", alpha=0.6),
                    medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=INK2, lw=1.0),
                    capprops=dict(color=INK2, lw=1.0))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.4)
    for i, k in enumerate(keys):
        ax.plot(i + 1, dists[k]["best_cost"], marker="D", ms=4.5, mfc=colors[i], mec=INK, mew=0.7, zorder=5)
    # clip the axis at the largest whisker
    whisker_top = max(max(w.get_ydata()) for w in bp["whiskers"])
    low = min(d["costs"].min() for d in dists.values())
    span = whisker_top - low
    ax.set_ylim(low - 0.06 * span, whisker_top + 0.10 * span)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([METHODS[k][0] for k in keys], fontsize=14)
    if ylabel:
        ax.set_ylabel("Trajectory cost", fontsize=16, color=INK)
    ax.set_title(title, fontsize=17, color=INK)
    ax.text(0.02, 0.97, panel, transform=ax.transAxes, fontsize=19, fontweight="bold", color=INK, va="top", ha="left")
    ax.yaxis.grid(True, color="#e4e3df", lw=0.7)
    ax.set_axisbelow(True)
    style_axes(ax)


def rect_corners(cx, cy, w, h, th):
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return local @ R.T + np.array([cx, cy])


def draw_pusher(ax, system, X, P, title):
    w, h = system._P.box_w, system._P.box_h
    norm = Normalize(0, X.shape[0] - 1)
    W = system._W
    ax.add_patch(Polygon(rect_corners(W.goal_x, W.goal_y, w, h, W.goal_theta), closed=True, fill=False, ls="--",
                         lw=1.3, edgecolor="#008300", zorder=4))
    for t in range(0, X.shape[0], 10):
        ax.add_patch(Polygon(rect_corners(X[t, 0], X[t, 1], w, h, X[t, 2]), closed=True, fill=False, lw=0.8,
                             edgecolor=TIME_CMAP(norm(t)), alpha=0.75, zorder=2))
    ax.add_patch(Polygon(rect_corners(X[-1, 0], X[-1, 1], w, h, X[-1, 2]), closed=True, fill=False, lw=2.0,
                         edgecolor=INK, zorder=5))
    ax.scatter(P[::4, 0], P[::4, 1], s=6, c=TIME_CMAP(norm(np.arange(0, len(P), 4))), zorder=3, linewidths=0)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.95, 0.72)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#e4e3df")
    ax.set_title(title, fontsize=17, color=INK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--acrobot-seed", type=int, default=0)
    parser.add_argument("--pusher-seed", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/optimization_benchmarks.pdf"))
    args = parser.parse_args()
    acro = load(args.results_dir, "acrobot", args.acrobot_seed)
    push = load(args.results_dir, "pusher", args.pusher_seed)
    system = build_system("pusher")
    replays = {}
    for method, d in push.items():
        X, P, cost = pusher_replay(system, d["best_theta"])
        replays[method] = (X, P)
        print(f"[pusher {method}] stored best cost {d['best_cost']:.2f}, replay {cost:.2f}", flush=True)

    fig = plt.figure(figsize=(12.5, 7.5), facecolor="white")
    columns = max(len(push), 1)
    gs = fig.add_gridspec(2, 2 * columns, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.16,
                          left=0.07, right=0.95, top=0.95, bottom=0.05)
    if acro:
        draw_boxplot(fig.add_subplot(gs[0, 0:columns]), acro, "Acrobot", "(a)")
    if push:
        draw_boxplot(fig.add_subplot(gs[0, columns:]), push, "Pusher-Slider", "(b)", ylabel=False)
    for j, method in enumerate(push):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        draw_pusher(ax, system, *replays[method], METHODS[method][0].replace("\n", " "))
        if j == 0:
            ax.text(-0.16, 1.10, "(c)", transform=ax.transAxes, fontsize=19, fontweight="bold", color=INK)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    fig.savefig(args.output.with_suffix(".png"), dpi=200)
    print(f"[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
