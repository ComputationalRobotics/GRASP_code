"""Receding-horizon GRASP MPC on a batched MuJoCo Warp rollout.

Heavy windows solve the next ``window_knots`` delta-knots with tempered SMC.
Optional light windows re-score the shifted control library and refine its
best candidates. The selected plan's first ``commit_knots`` knots are executed,
and world 0's state is broadcast to every world. The next window's AR(1) prior
is conditioned on the last committed knot; warm start centers it on the
unexecuted suffix of the preceding plan.

Objective bookkeeping matches ``grasp.sim.rollout``: reward at absolute sim
step ``s`` uses reference index ``s`` of ``get_slice(ref_data, 1, T + 1)``,
``get_reward`` for ``s < T - 1`` and ``get_terminal_reward`` at ``s = T - 1``;
cost is ``-(sum of rewards) / T`` so window costs add up to the full-horizon
cost.  Delta-knots are expanded to simulator rate with a *local* linear ramp
(knot ``k`` applies at step ``k * n`` and ramps to knot ``k + 1``); the
committed trajectory is stored at step level, so the final validation is an
exact re-evaluation of exactly what was executed.

The window sampler is pluggable through ``sampler_factory(batch_cost, prior,
K, m, beta, window_index)``; the default is the host tempered SMC sampler.

MJWarp is not bit-deterministic across worlds or repeats, so the checks here
report and gate on tolerances instead of asserting equality.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Callable

import jax
import numpy as np
import torch

from grasp.optim.prior_ar1 import WindowAR1Prior
from grasp.optim.sampler import GraspSampler


def _finite_order(costs, candidates=None):
    """Rank only evaluable plans; never let a NaN/Inf win selection."""
    costs = np.asarray(costs, dtype=np.float64)
    valid = np.isfinite(costs)
    if candidates is not None:
        valid &= np.isfinite(np.asarray(candidates).reshape(len(costs), -1)).all(axis=1)
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise RuntimeError("no finite candidate plans remain")
    return ids[np.argsort(costs[ids], kind="stable")]


def _replica_cost_stats(costs):
    """Average finite replicas; a plan with no finite replicas stays ineligible."""
    costs = np.asarray(costs, dtype=np.float64)
    mean = np.full(len(costs), np.inf)
    std = np.full(len(costs), np.inf)
    for i, row in enumerate(costs):
        finite = row[np.isfinite(row)]
        if finite.size:
            mean[i] = finite.mean()
            std[i] = finite.std()
    return mean, std



def local_interp(knots: torch.Tensor, steps_per_knot: int) -> torch.Tensor:
    """Expand (N, K, m) knots to (N, K*n, m): knot k at step k*n, linear ramp
    to knot k+1, last knot held."""
    N, K, m = knots.shape
    n = int(steps_per_knot)
    ramps = torch.arange(n, device=knots.device, dtype=knots.dtype) / float(n)
    following = torch.cat([knots[:, 1:], knots[:, -1:]], dim=1)
    out = knots[:, :, None, :] + ramps[None, None, :, None] * (following - knots)[
        :, :, None, :
    ]
    return out.reshape(N, K * n, m)


class GraspWindowEvaluator:
    """Windowed rollout from the *current* simulator state."""

    def __init__(
        self,
        *,
        config: Any,
        env: Any,
        mjwp: Any,
        ref_data: tuple,
        full_reference: torch.Tensor,
        real_actuator_ids: list[int],
        env_params: dict,
        steps_per_knot: int,
        total_steps: int,
    ):
        self.config = config
        self.env = env
        self.mjwp = mjwp
        from grasp.utils.interp import get_slice

        self.ref_full = get_slice(ref_data, 1, int(total_steps) + 1)
        self.full_reference = torch.as_tensor(
            full_reference, dtype=torch.float32, device=config.device
        )
        self.real_ids = [int(i) for i in real_actuator_ids]
        self.env_params = dict(env_params)
        self.n = int(steps_per_knot)
        self.T = int(total_steps)
        self.N = int(config.num_samples)
        self.num_worlds = self.N               # simulator worlds (== candidates for the plain evaluator)
        self.sim_steps = 0
        # pin_first_knot: knot 0 of every window (candidate rollouts AND execution) is replaced by the control offset
        # executed at the step just before the window, so a replan cannot jump the command.
        self.pin_first_knot = False
        self.last_exec_delta = None
        # joint_start: in the first window (k0 = 0) every candidate's knot 0 is ALSO its initial robot
        # configuration -- the actuated joints start at rest at their knot-0 position targets (reference control at step
        # 0 + knot-0 offset), objects at the reference start.  The initial configuration is thereby optimised jointly
        # with the first window's controls under the ordinary window cost and prior; nothing else changes.  Set up by
        # enable_joint_start(); cleared once the first window is committed.
        self.joint_start_pending = False
        self.joint_start_q = None            # (m,) chosen initial configuration of the actuated joints, after commit
    def enable_joint_start(self, model_cpu):
        jid = [int(model_cpu.actuator_trnid[a, 0]) for a in self.real_ids]
        dev = self.config.device
        self._js_qadr = torch.as_tensor([int(model_cpu.jnt_qposadr[j]) for j in jid], device=dev)
        self._js_dadr = torch.as_tensor([int(model_cpu.jnt_dofadr[j]) for j in jid], device=dev)
        lim = np.asarray(model_cpu.jnt_limited[jid]) > 0
        lo = np.where(lim, model_cpu.jnt_range[jid, 0], -np.inf); hi = np.where(lim, model_cpu.jnt_range[jid, 1], np.inf)
        self._js_lo = torch.as_tensor(lo, dtype=torch.float32, device=dev); self._js_hi = torch.as_tensor(hi, dtype=torch.float32, device=dev)
        self.joint_start_pending = True
    def joint_start_config(self, knot0) -> torch.Tensor:
        """(B, m) knot-0 offsets -> (B, m) initial joint positions (= knot-0 position targets), clipped to joint range."""
        k = torch.as_tensor(np.asarray(knot0, dtype=np.float32), device=self.config.device).reshape(-1, len(self.real_ids))
        q = self.full_reference[0, self.real_ids].unsqueeze(0) + k
        return torch.maximum(torch.minimum(q, self._js_hi), self._js_lo)
    def _set_start(self, q):
        """Overwrite the actuated joints of every world with q ((N, m) or (m,)), at rest."""
        import warp as wp
        with wp.ScopedDevice(self.env.device):
            qpos = wp.to_torch(self.env.data_wp.qpos); qvel = wp.to_torch(self.env.data_wp.qvel)
            qpos[:, self._js_qadr] = q.to(qpos.dtype) if q.ndim == 2 else q.to(qpos.dtype).unsqueeze(0)
            qvel[:, self._js_dadr] = 0.0
    def _maybe_joint_start(self, knots_t, k0: int):
        if self.joint_start_pending and int(k0) == 0:
            self._set_start(self.joint_start_config(knots_t[:, 0, :].detach().cpu().numpy()))
    # -- controls -------------------------------------------------------------
    def controls_for_window(self, knots: np.ndarray, k0: int) -> torch.Tensor:
        knots_t = torch.as_tensor(np.asarray(knots, dtype=np.float32), device=self.config.device)
        if knots_t.ndim != 3 or knots_t.shape[0] != self.N:
            raise ValueError(
                f"knots must have shape ({self.N}, K, m), got {tuple(knots_t.shape)}"
            )
        knots_t = self._pin(knots_t)
        K = int(knots_t.shape[1])
        s0 = int(k0) * self.n
        s1 = s0 + K * self.n
        if s1 > self.T:
            raise ValueError("window exceeds the reference horizon")
        full = self.full_reference[s0:s1].unsqueeze(0).repeat(self.N, 1, 1)
        full[:, :, self.real_ids] += local_interp(knots_t, self.n)
        return full

    def _pin(self, knots_t: torch.Tensor) -> torch.Tensor:
        if not self.pin_first_knot or self.last_exec_delta is None:
            return knots_t
        knots_t = knots_t.clone()
        knots_t[:, 0, :] = torch.as_tensor(self.last_exec_delta, dtype=knots_t.dtype, device=knots_t.device)
        return knots_t

    # -- rollout ----------------------------------------------------------------
    def _window_terminal(self, ref):
        """Terminal reward at a window end that is not the task end (the step reward was just computed there)."""
        term, _ = self.mjwp.get_terminal_reward(self.config, self.env, ref)
        return term

    def _step_and_reward(self, ctrl_t: torch.Tensor, s: int):
        self.mjwp.step_env(self.config, self.env, ctrl_t)
        ref = [r[s] for r in self.ref_full]
        if s == self.T - 1:
            rew, _ = self.mjwp.get_terminal_reward(self.config, self.env, ref)
        else:
            rew, _ = self.mjwp.get_reward(self.config, self.env, ref)
        return rew, ref

    def rollout_window(self, knots: np.ndarray, k0: int):
        """Per-step rewards (N, L) and the terminal reward at the window end
        (N,) evaluated from the current state; the state is restored."""
        ctrls = self.controls_for_window(knots, k0)
        L = int(ctrls.shape[1])
        s0 = int(k0) * self.n
        self.mjwp.save_state(self.env)
        self._maybe_joint_start(torch.as_tensor(np.asarray(knots, dtype=np.float32)), k0)
        init_param = self.mjwp.save_env_params(self.config, self.env)
        self.mjwp.load_env_params(self.config, self.env, self.env_params)
        rewards = torch.zeros((self.N, L), dtype=torch.float32, device=self.config.device)
        terminal_end = torch.zeros((self.N,), dtype=torch.float32, device=self.config.device)
        for t in range(L):
            s = s0 + t
            rew, ref = self._step_and_reward(ctrls[:, t], s)
            rewards[:, t] = rew
            if t == L - 1 and s != self.T - 1:
                terminal_end = self._window_terminal(ref)
        self.mjwp.load_state(self.env, None)
        self.mjwp.load_env_params(self.config, self.env, init_param)
        self.sim_steps += self.N * L
        return (
            rewards.detach().cpu().numpy().astype(np.float64),
            terminal_end.detach().cpu().numpy().astype(np.float64),
        )

    def window_cost(self, knots: np.ndarray, k0: int, phi_weight: float) -> np.ndarray:
        rewards, terminal_end = self.rollout_window(knots, k0)
        cost = -np.sum(rewards, axis=1) / self.T
        if phi_weight != 0.0:
            cost = cost + float(phi_weight) * (-terminal_end / self.T)
        return cost

    def _execute(self, ctrls: torch.Tensor, s0: int, deltas: np.ndarray):
        L_c = int(ctrls.shape[1])
        init_param = self.mjwp.save_env_params(self.config, self.env)
        self.mjwp.load_env_params(self.config, self.env, self.env_params)
        rewards = torch.zeros((self.N, L_c), dtype=torch.float32, device=self.config.device)
        import warp as wp
        qpos_exec, qvel_exec = [], []
        for t in range(L_c):
            rew, _ = self._step_and_reward(ctrls[:, t], s0 + t)
            rewards[:, t] = rew
            qpos_exec.append(wp.to_torch(self.env.data_wp.qpos)[0].detach().cpu().numpy().copy())
            qvel_exec.append(wp.to_torch(self.env.data_wp.qvel)[0].detach().cpu().numpy().copy())
            if (not np.isfinite(qpos_exec[-1]).all() or not np.isfinite(qvel_exec[-1]).all()
                    or not bool(torch.isfinite(rew[0]))):
                raise RuntimeError(f"non-finite executed state/reward at simulation step {s0 + t}; stopping execution")
        self.mjwp.load_env_params(self.config, self.env, init_param)
        self.sim_steps += self.N * L_c
        self.last_executed_states = (np.asarray(qpos_exec, dtype=np.float64), np.asarray(qvel_exec, dtype=np.float64))
        src = torch.zeros(self.N, dtype=torch.long)          # all worlds ran identical controls
        dst = torch.arange(self.N, dtype=torch.long)
        self.mjwp.copy_sample_state(self.config, self.env, src, dst)
        return deltas, rewards.detach().cpu().numpy().astype(np.float64)

    def commit(self, knots_window: np.ndarray, k0: int, commit_knots: int):
        """Execute the first ``commit_knots`` knots of one plan in every world
        (no restore), then broadcast world 0's state to all worlds.  Returns the
        executed step-level deltas (L_c, m) and per-world rewards (N, L_c)."""
        knots_batch = np.repeat(
            np.asarray(knots_window, dtype=np.float32)[None], self.N, axis=0
        )
        ctrls = self.controls_for_window(knots_batch, k0)
        L_c = int(commit_knots) * self.n
        s0 = int(k0) * self.n
        deltas = local_interp(
            self._pin(torch.as_tensor(knots_batch[:1], device=self.config.device)), self.n
        )[0, :L_c]
        deltas = deltas.detach().cpu().numpy().astype(np.float32)
        self.last_exec_delta = deltas[-1].copy()
        if self.joint_start_pending and s0 == 0:
            q0 = self.joint_start_config(np.asarray(knots_window, dtype=np.float32)[0])[0]
            self._set_start(q0)
            self.joint_start_q = q0.detach().cpu().numpy().astype(np.float64)
            self.joint_start_pending = False
        return self._execute(ctrls[:, :L_c], s0, deltas)


def default_sampler_factory(batch_cost, prior, K, m, beta, window_index):
    return GraspSampler(batch_cost, prior, horizon=K, control_dim=m, beta=beta)


class GraspMPC:
    def __init__(
        self,
        evaluator: GraspWindowEvaluator,
        *,
        total_knots: int,
        window_knots: int,
        commit_knots: int,
        control_dim: int,
        prior_rho: float,
        prior_sigma: float,
        beta: float,
        sampler_kwargs: dict,
        phi_weight: float = 1.0,
        seed: int = 0,
        trace_dir=None,
        sampler_factory: Callable | None = None,
        prior_channel_scale=None,
        library_reuse_period: int = 0,
        library_local_improve: bool = False,
        improve_scale: float = 0.3,
        improve_parents: int = 50,
    ):
        self.prior_channel_scale = (
            None if prior_channel_scale is None
            else np.asarray(prior_channel_scale, np.float32)
        )
        self.prior_window_ramp = 1.0
        self.warm_start_plan = False
        # Heavy windows save their stage populations. Light windows shift and
        # re-score those controls, optionally adding two-generation refinements.
        self.library_reuse_period = int(library_reuse_period)
        self.library_local_improve = bool(library_local_improve)
        self.improve_scale = float(improve_scale)
        self.improve_parents = int(improve_parents)
        if self.library_local_improve and not self.library_reuse_period:
            raise ValueError("library_local_improve requires library_reuse_period")
        self.robust_select_k = 0
        self.trace_dir = trace_dir
        self.ev = evaluator
        self.total_knots = int(total_knots)
        self.window_knots = int(window_knots)
        self.commit_knots = int(commit_knots)
        self.control_dim = int(control_dim)
        self.rho = float(prior_rho)
        self.sigma = float(prior_sigma)
        self.beta = float(beta)
        self.sampler_kwargs = dict(sampler_kwargs)
        self.phi_weight = float(phi_weight)
        self.seed = int(seed)
        self.sampler_factory = sampler_factory or default_sampler_factory
        if self.commit_knots <= 0 or self.commit_knots > self.window_knots:
            raise ValueError("commit_knots must lie in [1, window_knots]")

    def run(self, verbose: bool = True) -> dict:
        ev = self.ev
        N, m, n, T = ev.N, self.control_dim, ev.n, ev.T
        start = time.time()
        committed_steps = np.zeros((T, m), dtype=np.float32)
        committed_knots = np.zeros((self.total_knots, m), dtype=np.float32)
        committed_cost_world = np.zeros(N, dtype=np.float64)
        executed_qpos: list[np.ndarray] = []   # world-0 states actually visited (closed loop)
        executed_qvel: list[np.ndarray] = []
        u_prev = None
        k0 = 0
        windows: list[dict] = []
        sampler_rollouts = 0
        index = 0
        prev_plan = None
        library = None           # (M, K_prev, m) controls of the previous HEAVY window's tempering path (library reuse)
        prev_D_lib = 0
        while k0 < self.total_knots:
            index += 1
            K = min(self.window_knots, self.total_knots - k0)
            D = min(self.commit_knots, K)
            if 0 < self.total_knots - (k0 + D) < 2:
                D = K          # never leave a 1-knot final window: execute the rest of this plan
            final = k0 + K == self.total_knots
            phi_w = 0.0 if final else self.phi_weight
            warm_bias = None
            if self.warm_start_plan and u_prev is not None and prev_plan is not None:
                pp = np.asarray(prev_plan, dtype=np.float32).reshape(-1, m)
                if pp.shape[0] == 0:
                    pp = np.asarray(u_prev, dtype=np.float32).reshape(1, m)
                pad = np.repeat(pp[-1:], max(0, K - pp.shape[0]), axis=0)
                warm_bias = np.concatenate([pp, pad], axis=0)[:K]
            cscale = self.prior_channel_scale
            if cscale is not None and cscale.ndim == 2:
                if k0 + K > cscale.shape[0]:
                    raise ValueError(f"prior_channel_scale has {cscale.shape[0]} knots but window "
                                     f"[{k0},{k0 + K}) runs past the end")
                cscale = cscale[k0:k0 + K]
            if self.prior_window_ramp != 1.0:
                if not self.warm_start_plan:
                    raise ValueError("prior_window_ramp is implemented for the warm-start window prior only")
                base = np.ones((K, m), np.float32) if cscale is None else np.broadcast_to(
                    np.asarray(cscale, np.float32).reshape(-1, m), (K, m))
                ramp = (self.prior_window_ramp ** (np.arange(K) / max(K - 1, 1))).astype(np.float32)
                cscale = np.ascontiguousarray(base * ramp[:, None])
            prior = WindowAR1Prior(
                m, K, self.rho, self.sigma, u_prev, channel_scale=cscale,
                bias=warm_bias, bias_prev=(None if warm_bias is None else u_prev),
            )

            path_library = []      # every stage population's controls (library reuse)
            invalid_candidates = 0
            light = bool(self.library_reuse_period >= 2 and (index - 1) % self.library_reuse_period != 0 and library is not None)
            library_info = None
            def extend_with(knots, c):
                """(N, K, m) decision knots -> (N, K + c, m) with the AR(1)-mean continuation."""
                knots = np.asarray(knots, dtype=np.float32)
                if c <= 0:
                    return knots
                tail = np.empty((knots.shape[0], c, knots.shape[2]), np.float32); prev = knots[:, -1]
                for t in range(c):
                    prev = self.rho * prev; tail[:, t] = prev
                return np.concatenate([knots, tail], axis=1)

            def batch_cost(positions, _k0=k0, _K=K, _phi=phi_w):
                return ev.window_cost(np.asarray(positions, dtype=np.float32).reshape((-1, _K, m)), _k0, _phi)

            def batch_cost_padded(U, _N=N, _k0=k0, _phi=phi_w):
                """window cost of B plans (K, m), scored in batches of N (the last one padded)."""
                nonlocal invalid_candidates
                U = np.asarray(U, dtype=np.float32); B_ = len(U); J_ = np.empty(B_, dtype=np.float64)
                for b_ in range(0, B_, _N):
                    chunk = U[b_:b_ + _N]
                    pad = np.concatenate([chunk, np.repeat(chunk[-1:], _N - len(chunk), axis=0)], axis=0) if len(chunk) < _N else chunk
                    valid_controls = np.isfinite(pad.reshape(_N, -1)).all(axis=1)
                    pad = np.where(valid_controls[:, None, None], pad, 0.0)
                    J_[b_:b_ + len(chunk)] = np.asarray(ev.window_cost(pad, _k0, _phi), dtype=np.float64)[: len(chunk)]
                valid = np.isfinite(J_) & np.isfinite(U.reshape(B_, -1)).all(axis=1)
                invalid_candidates += int((~valid).sum())
                return np.where(valid, J_, np.inf)

            keys = jax.random.split(
                jax.random.fold_in(jax.random.PRNGKey(self.seed), 1000 + index), N
            )
            w_start = time.time()
            free_result = None
            if light:
                # LIGHT window: shift every library control by the executed D knots, append its AR(1)-mean tail,
                # truncate to this window's K, and re-score everything from the actual state (batches of N).
                lib = np.asarray(library, dtype=np.float32); Kp = int(lib.shape[1]); Dl = int(prev_D_lib)
                raw_suffix = lib[:, Dl:]                                                        # (M, Kp - Dl, m)
                raw_full = np.concatenate([raw_suffix, extend_with(raw_suffix, Dl)[:, Kp - Dl:]], axis=1)[:, :K]
                sets = {"raw": raw_full}
                cand = np.concatenate(list(sets.values()), axis=0)
                origin = np.concatenate([np.full(len(v), k_) for k_, v in sets.items()])
                shifted, uidx = np.unique(cand.reshape(len(cand), -1), axis=0, return_index=True)
                shifted = shifted.reshape((-1, K, m)); origin = origin[uidx]
                M_ = int(shifted.shape[0]); nb = int(np.ceil(M_ / N))
                J_lib = batch_cost_padded(shifted)
                keep = _finite_order(J_lib, shifted)
                shifted, origin, J_lib = shifted[keep], origin[keep], J_lib[keep]
                M_ = len(J_lib)
                order = np.arange(M_)
                improve_info = None
                if self.library_local_improve:
                    rng_i = np.random.default_rng(self.seed * 1000 + index)
                    P = int(min(self.improve_parents, M_)); parents = order[:P]
                    # per-(knot, channel) innovation std of this window's prior, so children respect the channel scales
                    sig, rho = np.asarray(prior.sigma_vec, dtype=np.float32), float(self.rho)
                    def unwhiten_delta(dz):                       # innovation deltas (B, K, m) -> control deltas via the AR(1) recursion
                        du = np.empty_like(dz); acc = np.zeros(dz.shape[::2], np.float32)
                        for t in range(dz.shape[1]):
                            acc = rho * acc + sig[t] * dz[:, t]; du[:, t] = acc
                        return du
                    B = N; per = int(np.ceil(B / P)); pidx = np.repeat(parents, per)[:B]
                    xi = rng_i.standard_normal((B, K, m)).astype(np.float32) * self.improve_scale      # common random numbers
                    plain = shifted[pidx] + unwhiten_delta(xi)
                    def score(Uc):
                        return batch_cost_padded(Uc)
                    lib_best = float(J_lib[order[0]])
                    # gen1 = plain children of the library's top parents (noise xi); the second noise batch xi2 is shared
                    # by the one-shot extra children (same library parents) and by gen2 (parents = top of library U gen1)
                    J_gen1 = score(plain)
                    xi2 = rng_i.standard_normal((B, K, m)).astype(np.float32) * self.improve_scale
                    oneshot2 = shifted[pidx] + unwhiten_delta(xi2)
                    J_one2 = score(oneshot2)
                    pool = np.concatenate([shifted, plain], axis=0); J_pool = np.concatenate([J_lib, J_gen1])
                    src = np.concatenate([np.zeros(len(shifted), bool), np.ones(B, bool)])         # True = came from gen1
                    parents2 = np.argsort(J_pool)[:P]; pidx2 = np.repeat(parents2, per)[:B]
                    gen2 = pool[pidx2] + unwhiten_delta(xi2)
                    J_gen2 = score(gen2)
                    best_A = float(min(J_gen1.min(), J_one2.min())); best_B = float(min(J_gen1.min(), J_gen2.min()))
                    improve_info = dict(mode="twogen", parents=P, children_per_set=B, scale=self.improve_scale,
                                        best_library=lib_best, best_gen1=float(J_gen1.min()),
                                        best_oneshot_2000=best_A, best_twogen_1000_1000=best_B,
                                        best_oneshot_second_batch=float(J_one2.min()), best_gen2=float(J_gen2.min()),
                                        n_gen1_better=int(np.sum(J_gen1 < lib_best)), n_oneshot2_better=int(np.sum(J_one2 < lib_best)),
                                        n_gen2_better=int(np.sum(J_gen2 < lib_best)),
                                        gen2_parents_from_gen1=int(np.sum(src[parents2])),
                                        twogen_gain_over_oneshot=float(best_A - best_B))
                    new_sets = [("gen1", plain, J_gen1), ("oneshot2", oneshot2, J_one2), ("gen2", gen2, J_gen2)]
                    # union for selection
                    shifted = np.concatenate([shifted] + [v for _, v, _ in new_sets], axis=0)
                    origin = np.concatenate([origin] + [np.full(len(v), k_) for k_, v, _ in new_sets])
                    J_lib = np.concatenate([J_lib] + [j for _, _, j in new_sets]); M_ = len(J_lib); nb += len(new_sets)
                    order = _finite_order(J_lib, shifted)
                jmin_lib = float(J_lib[order[0]])
                top = order[:N] if len(order) >= N else np.resize(order, N)
                free_result = SimpleNamespace(samples=shifted[top].reshape((N, K * m)), costs=J_lib[top], best_index=0,
                                              info=dict(completed=True, total_rollout_calls=nb * N, num_iterations=0))
                library_info = dict(mode="light", library_size=int(lib.shape[0]), candidates=M_, batches=nb,
                                    best_long=jmin_lib, median_long=float(np.median(J_lib)),
                                    n_within_3nat=int(np.sum(self.beta * (J_lib - jmin_lib) <= 3.0)),
                                    rollouts_long=int(nb * N), executed_set=str(origin[order[0]]),
                                    best_long_by_set={k_: float(J_lib[origin == k_].min()) for k_ in sets if np.any(origin == k_)},
                                    n_by_set={k_: int(np.sum(origin == k_)) for k_ in sets})
                if improve_info is not None:
                    library_info["local_improve"] = improve_info
                    library_info["best_long_by_set"] = {k_: float(J_lib[origin == k_].min()) for k_ in np.unique(origin)}
                    library_info["executed_set"] = str(origin[order[0]])
                sampler = None
                init = shifted[top]
            else:
                init = np.asarray(
                    jax.vmap(lambda key: prior.sample(key, K, control_dim=m))(keys),
                    dtype=np.float64,
                )
                init[0] = prior.conditional_mean()
                sampler = self.sampler_factory(batch_cost, prior, K, m, self.beta, index)
                if getattr(ev, "pin_first_knot", False) and ev.last_exec_delta is not None:
                    # knot 0 is overridden by the executed control in every rollout, so it is not a decision variable:
                    # sample it as that constant.  Otherwise the surrogate drifts it along slopes the cost does not have
                    # and the AR(1) prior conditions knot 1 on a value that is never executed.
                    if not hasattr(sampler, "frozen_mask"):
                        raise ValueError("pin_first_knot needs a sampler that supports frozen_mask")
                    init[:, 0, :] = np.asarray(ev.last_exec_delta, dtype=np.float64)[None, :]
                    frozen = np.zeros((K, m), dtype=bool); frozen[0] = True
                    sampler.frozen_mask = frozen.reshape(-1)
            trace = None
            if self.trace_dir is not None:
                from pathlib import Path
                from grasp.optim.trace import OptimizationTrace
                trace = OptimizationTrace(Path(self.trace_dir) / f"window_{index:03d}")
                if index == 1:
                    import sys
                    source_root = Path(__file__).resolve().parents[1]
                    sources = {
                        name: (source_root / name).read_text()
                        for name in (
                            "mpc/grasp_mpc.py",
                            "optim/sampler.py",
                            "optim/trace.py",
                            "mpc/run.py",
                        )
                    }
                    trace("run_context", argv=sys.argv, sources=sources,
                        config_repr=repr(ev.config) if hasattr(ev, "config") else "test",
                        note="Dynamic fields are diagnostic snapshots; complete backend caches and bit-exact replay are not guaranteed.")
                state = {}
                for name in ("qpos", "qvel", "act", "ctrl", "time", "qacc",
                             "qacc_warmstart", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat"):
                    field = getattr(ev.env.data_wp, name, None)
                    if field is not None and hasattr(field, "numpy"):
                        state[name] = field.numpy()
                trace("window_input", window=index, k0=k0, K=K, D=D,
                    seed=self.seed, sampler_seed=self.seed + 1000 * index,
                    state=state, initial_particles=init, prior_rho=self.rho, prior_sigma=self.sigma,
                    prior_u_prev=u_prev, beta=self.beta, phi_weight=phi_w,
                    sampler_kwargs=self.sampler_kwargs,
                    full_reference=ev.full_reference.detach().cpu().numpy(),
                    steps_per_knot=n, total_steps=T)
            run_extra = {"trace_callback": trace} if trace is not None else {}
            if self.library_reuse_period and sampler is not None and free_result is None:
                # library reuse: keep every stage's pre-resampling population, no extra rollouts
                def library_hook(stage, temperature, particles, costs, **_):
                    path_library.append(np.asarray(particles, dtype=np.float32).reshape((-1, K, m)).copy())
                if hasattr(sampler, "stage_hook"):
                    sampler.stage_hook = library_hook
                else:
                    _user_trace_lib = run_extra.get("trace_callback")

                    def _trace_with_library(event, **data):
                        if event == "resample":
                            library_hook(**data)
                        if _user_trace_lib is not None:
                            _user_trace_lib(event, **data)
                    run_extra = {"trace_callback": _trace_with_library}
            if free_result is not None:
                result = free_result
            else:
                result = sampler.run(
                    seed=self.seed + 1000 * index,
                    num_particles=N,
                    initial_particles=init.reshape((N, K * m)),
                    verbose=verbose,
                    **run_extra,
                    **self.sampler_kwargs,
                )
            if trace is not None:
                trace("window_result", samples=result.samples, costs=result.costs,
                    best_index=result.best_index, completed=bool(result.info["completed"]))
            if not bool(result.info["completed"]):
                raise RuntimeError(f"window {index}: tempering did not complete")
            calls = int(result.info["total_rollout_calls"])
            pop_costs = np.asarray(result.costs, dtype=np.float64)
            if light:
                best_index = 0           # result.samples is ordered by window cost
            else:
                best_index = int(_finite_order(pop_costs, result.samples)[0])
            best = np.asarray(result.samples[best_index], dtype=np.float32).reshape((K, m))
            best_cost = float(pop_costs[best_index])
            sampler_rollouts += calls
            robust_info = None
            if self.robust_select_k > 0:
                cand_s = np.asarray(result.samples, dtype=np.float32).reshape((-1, K, m))
                cand_c = np.asarray(pop_costs, dtype=np.float64)
                _, first = np.unique(cand_s.reshape(len(cand_s), -1), axis=0, return_index=True)   # distinct plans only
                first = first[_finite_order(cand_c[first], cand_s[first])]
                sel_r = first[: min(int(self.robust_select_k), len(first))]
                reps = N // len(sel_r)
                batch_r = np.repeat(cand_s[sel_r], reps, axis=0)
                if len(batch_r) < N:
                    batch_r = np.concatenate([batch_r, np.repeat(cand_s[sel_r[:1]], N - len(batch_r), axis=0)], axis=0)
                rew_r, term_r = ev.rollout_window(batch_r, k0)
                c_r = (-np.sum(rew_r, axis=1) / T + phi_w * (-term_r / T))[: len(sel_r) * reps].reshape(len(sel_r), reps)
                invalid_candidates += int((~np.isfinite(c_r)).sum())
                mean_r, std_r = _replica_cost_stats(c_r)
                j_r = int(_finite_order(mean_r)[0])
                robust_info = dict(k=int(len(sel_r)), reps=int(reps), single_best_cost=float(best_cost),
                                   single_best_mean=float(mean_r[0]), chosen_rank=int(j_r), chosen_single_cost=float(cand_c[sel_r[j_r]]),
                                   chosen_mean=float(mean_r[j_r]), chosen_std=float(std_r[j_r]),
                                   single_best_std=float(std_r[0]), invalid_replicas=int((~np.isfinite(c_r)).sum()),
                                   replica_policy="finite_only",
                                   finite_replicas_by_candidate=np.isfinite(c_r).sum(axis=1).tolist(),
                                   chosen_finite_replicas=int(np.isfinite(c_r[j_r]).sum()))
                best = cand_s[sel_r[j_r]].copy(); best_cost = float(mean_r[j_r])
                sampler_rollouts += N
                if verbose:
                    print(f"[robust-select] window {index:02d}: top {len(sel_r)} x {reps} rollouts; single-rollout best "
                          f"{robust_info['single_best_cost']:.5f} has mean {robust_info['single_best_mean']:.5f}; executed rank "
                          f"{j_r} mean {robust_info['chosen_mean']:.5f} (std {robust_info['chosen_std']:.5f}); "
                          f"finite_only counts={robust_info['finite_replicas_by_candidate']} "
                          f"invalid={robust_info['invalid_replicas']}", flush=True)
            # Re-evaluate the selected plan in every simulator world before committing.
            rew_eval, term_eval = ev.rollout_window(np.repeat(best[None], N, axis=0), k0)
            cost_eval = -np.sum(rew_eval, axis=1) / T + phi_w * (-term_eval / T)
            finite_reeval = np.isfinite(cost_eval)
            if not finite_reeval.any():
                raise RuntimeError(f"window {index}: selected plan has no finite re-evaluations")
            # commit D knots, then check B: executed vs planned rewards, same slot
            deltas, rew_commit = ev.commit(best, k0, D)
            if trace is not None:
                trace("commit", plan=best, executed_control_offsets=deltas, planned_rewards=rew_eval,
                    executed_rewards=rew_commit,
                    executed_qpos=ev.last_executed_states[0],
                    executed_qvel=ev.last_executed_states[1])
            s0 = k0 * n
            committed_steps[s0 : s0 + D * n] = deltas
            executed_qpos.append(ev.last_executed_states[0])
            executed_qvel.append(ev.last_executed_states[1])
            committed_knots[k0 : k0 + D] = best[:D]
            committed_cost_world += -np.sum(rew_commit, axis=1) / T
            plan_vs_exec = np.abs(rew_commit[0] - rew_eval[0, : D * n])
            u_prev = best[D - 1]          # AR(1) continuity across windows
            prev_plan = best[D:].copy()   # unexecuted suffix for the next window's warm start
            if self.library_reuse_period:
                if light:
                    if index % self.library_reuse_period == 0:
                        library = None                               # next window is heavy again
                    else:
                        library = np.asarray(shifted, dtype=np.float32)   # all candidates scored at this state (library U children) carry on
                        prev_D_lib = D
                        library_info["carried_to_next_light"] = int(library.shape[0])
                else:
                    pops = list(path_library) + [np.asarray(result.samples, dtype=np.float32).reshape((N, K, m))]
                    library = np.concatenate(pops, axis=0) if pops else None
                    prev_D_lib = D
                    library_info = dict(mode="heavy", library_size=int(library.shape[0]) if library is not None else 0)
            k0 += D
            # Both energies of the executed plan, in nats: beta * window cost J, and the window prior's energy
            # (-log p0 up to a constant: half the sum of squared whitened AR(1) innovations about the prior centre).
            try:
                exec_prior_energy = float(prior.energy(np.asarray(best, dtype=np.float32)))
            except Exception:
                exec_prior_energy = None
            exec_beta_j = float(self.beta * best_cost) if best_cost is not None and np.isfinite(best_cost) else None
            record = {
                "invalid_candidate_evaluations": invalid_candidates + int(result.info.get("invalid_evaluations", 0)),
                "initial_invalid_particles": int(result.info.get("initial_invalid_particles", 0)),
                "beta_J_executed": exec_beta_j,
                "prior_energy_executed": exec_prior_energy,
                "robust_select": robust_info,
                "window": index,
                "k0": int(k0 - D),
                "K": int(K),
                "D": int(D),
                "num_stages": int(result.info["num_iterations"]),
                "library_reuse": library_info,
                "best_plan": best.tolist(),
                "sampler_rollouts": calls,
                "acceptance": [float(a) for a in np.asarray(result.info["acceptance_rates"]).ravel()]
                if "acceptance_rates" in result.info
                else None,
                "step_sizes": [float(a) for a in np.asarray(result.info["step_sizes"]).ravel()]
                if "step_sizes" in result.info
                else None,
                "mala_preconditioner": result.info.get("mala_preconditioner"),
                "best_cost": best_cost,
                "reeval_cost_slot0": float(cost_eval[0]),
                "reeval_invalid_replicas": int((~finite_reeval).sum()),
                "reeval_cost_slot_spread": float(np.ptp(cost_eval[finite_reeval])),
                "reeval_cost_slot_std": float(np.std(cost_eval[finite_reeval])),
                "plan_vs_exec_reward_maxabs": float(np.max(plan_vs_exec)),
                "plan_vs_exec_reward_sum": float(np.sum(rew_commit[0] - rew_eval[0, : D * n])),
                "committed_cost_slot0": float(committed_cost_world[0]),
                "sim_steps": int(ev.sim_steps),
                "seconds": time.time() - w_start,
                "elapsed": time.time() - start,
            }
            windows.append(record)
            if verbose:
                print(
                    f"[mpc] window {index:02d} knots [{record['k0']},{record['k0'] + K}) "
                    f"stages={record['num_stages']} rollouts={calls} "
                    + (f"LIGHT[{library_info['candidates']} candidates, best long {library_info['best_long']:.5f}, {library_info['n_within_3nat']} within 3 nat"
                       + (f", lib {library_info['local_improve']['best_library']:.5f} oneshot2000 {library_info['local_improve']['best_oneshot_2000']:.5f} twogen {library_info['local_improve']['best_twogen_1000_1000']:.5f} gen2parents_from_gen1 {library_info['local_improve']['gen2_parents_from_gen1']} exec={library_info['executed_set']}" if library_info.get('local_improve') else "") + "] "
                       if library_info and library_info['mode'] == 'light' else "")
                    +                     f"best={best_cost:.6f} reeval0={cost_eval[0]:.6f} "
                    f"slot_spread={record['reeval_cost_slot_spread']:.2e} "
                    f"plan-exec|max={record['plan_vs_exec_reward_maxabs']:.2e} "
                    f"committed={committed_cost_world[0]:.6f} "
                    f"invalid={record['invalid_candidate_evaluations']} "
                    f"reeval_invalid={record['reeval_invalid_replicas']} "
                    f"sim={ev.sim_steps} t={record['elapsed']:.0f}s",
                    flush=True,
                )
        return {
            "committed_steps": committed_steps,
            "committed_knots": committed_knots,
            "committed_cost_world": committed_cost_world,
            "executed_qpos": np.concatenate(executed_qpos, axis=0),
            "executed_qvel": np.concatenate(executed_qvel, axis=0),
            "windows": windows,
            "sampler_rollouts": int(sampler_rollouts),
            "sim_steps": int(ev.sim_steps),
            "runtime_seconds": time.time() - start,
        }
