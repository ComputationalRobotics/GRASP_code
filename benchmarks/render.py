"""Render a pusher-slider trajectory of a ``benchmarks.run`` method: the lowest-cost final particle of one seed
(``--seed``; by default the seed with the lowest best cost), as a time-coloured stroboscopic PNG and an MP4.  The sample
is re-simulated with the system the runner used and its cost is checked against the recorded one; run it on the device
type the benchmark ran on.

    python -m benchmarks.render --results-dir outputs/benchmarks --method grasp --seed 9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import jax.numpy as jnp
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from benchmarks.run import TASKS, build_system, trajectory_cost

W, HH = 900, 760


def colour(a):  # dark blue -> teal -> yellow (viridis-like), a in [0, 1]
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    x = a * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    f = x - i
    return tuple(int(v) for v in stops[i] * (1 - f) + stops[i + 1] * f)


def render(folder: Path, out_dir: Path, label: str, font, big, seed: int | None = None) -> None:
    H = TASKS["pusher"]["horizon"]
    if seed is None:
        seed = min((json.loads(p.read_text()) for p in folder.glob("seed*.json")), key=lambda r: r["best"])["seed"]
    z = np.load(folder / f"seed{seed}.npz")
    i = int(np.argmin(z["costs"]))
    system = build_system("pusher")
    m = int(system.control_dim)
    J = float(trajectory_cost(system, H)(jnp.asarray(z["samples"][i], jnp.float32)))
    assert abs(J - z["costs"][i]) < 1e-3 * abs(z["costs"][i]), (J, z["costs"][i])
    U = system.clamp_control(jnp.asarray(z["samples"][i], jnp.float32).reshape(H, m))
    x0 = system.default_initial_state
    X = np.concatenate([np.asarray(x0)[None], np.asarray(system.rollout(x0, U))])

    P, Wt = system._P, system._W
    r_ = 0.5 * np.hypot(P.box_w, P.box_h) + 2 * P.pusher_radius  # fit every box pose plus the pusher
    lo = np.minimum(X[:, :2].min(0), [Wt.goal_x, Wt.goal_y]) - r_
    hi = np.maximum(X[:, :2].max(0), [Wt.goal_x, Wt.goal_y]) + r_
    sc = min((W - 40) / (hi[0] - lo[0]), (HH - 110) / (hi[1] - lo[1]))
    mid = (lo + hi) / 2

    def world(p):
        return (W / 2 + sc * (p[0] - mid[0]), (HH + 10) / 2 - sc * (p[1] - mid[1]))

    def box(x):
        c, th = x[:2], x[2]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        hw, hh = P.box_w / 2, P.box_h / 2
        return [world(c + R @ v) for v in np.array([[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]])]

    def pusher(k):
        kk = min(k, H - 1)
        pc, _, _ = system._pusher_center_world(jnp.asarray(X[kk][:2]), jnp.asarray(X[kk][2]), U[kk],
                                               jnp.asarray(kk, jnp.int32))
        return world(np.asarray(pc))

    def draw_state(dr, x, col, k, w=3):
        dr.polygon(box(x), outline=col, width=w)
        p, r = pusher(k), sc * P.pusher_radius
        dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=col, width=w)

    def base(title):
        img = Image.new("RGB", (W, HH), "white")
        dr = ImageDraw.Draw(img)
        g = box(np.array([Wt.goal_x, Wt.goal_y, Wt.goal_theta]))
        for j in range(4):  # dashed goal pose
            p0, p1 = np.array(g[j]), np.array(g[(j + 1) % 4])
            for s in np.linspace(0, 1, 14)[::2]:
                dr.line([tuple(p0 + (p1 - p0) * s), tuple(p0 + (p1 - p0) * min(s + 1 / 13, 1))],
                        fill=(150, 150, 150), width=3)
        dr.text((16, 12), title, font=big, fill="black")
        return img, dr

    title = f"pusher: {label}, seed {seed}, cost {J:.2f}"
    img, dr = base(title)
    for k in range(0, len(X), 12):
        draw_state(dr, X[k], colour(k / (len(X) - 1)), k)
    draw_state(dr, X[-1], colour(1.0), len(X) - 1)
    dr.text((16, HH - 34), "colour = time (dark -> yellow); dashed grey = goal pose", font=font, fill="black")
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"pusher_seed{seed}.png")
    frames = []
    for k in range(0, len(X), 2):
        im, d2 = base(title + f"   t = {k * float(system.dt):.2f} s")
        draw_state(d2, X[k], (30, 90, 200), k)
        frames.append(np.asarray(im))
    imageio.mimsave(out_dir / f"pusher_seed{seed}.mp4", frames, fps=25)
    print("pusher seed", seed, "cost", round(J, 2), "saved", out_dir / f"pusher_seed{seed}.png",
          out_dir / f"pusher_seed{seed}.mp4", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--method", default="grasp", choices=("grasp", "rwm", "tsmc_grad", "mppi"))
    parser.add_argument("--seed", type=int, default=None, help="default: the seed with the lowest best cost")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <results-dir>/renders/<method>")
    args = parser.parse_args()
    path = font_manager.findfont("DejaVu Sans")
    font, big = ImageFont.truetype(path, 18), ImageFont.truetype(path, 22)
    render(args.results_dir / "pusher" / args.method, args.output_dir or args.results_dir / "renders" / args.method,
           args.method, font, big, args.seed)


if __name__ == "__main__":
    main()
