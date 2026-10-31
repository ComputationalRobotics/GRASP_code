"""GRASP: gradient-guided tempered SMC over control trajectories.

The simulator is reached only through a user-supplied batch function

    batch_cost(U_flat: np.ndarray (N, d) float32) -> np.ndarray (N,) float64

so any backend (MuJoCo Warp, a JAX simulator, a remote service) can serve rollouts.  JAX is used as a math
library only: the analytic prior score and the inner BlackJAX NUTS move on the frozen surrogate target.

Each tempering stage resamples by an adaptive ESS schedule, fits a quadratic surrogate to every rollout evaluated
so far (rejected proposals included), and runs Markov moves whose proposals follow the surrogate gradient while the
Metropolis-Hastings test uses the true rollout cost, so the tempered target is exact.

Kernels:
    surrogate_mala  Langevin proposal along the surrogate gradient, optionally preconditioned by the prior
                    covariance (``mala_preconditioner="prior_full"``).
    surrogate_nuts  NUTS on the frozen surrogate target with a delayed-acceptance correction on the true cost.
    rwm             random-walk Metropolis (no surrogate), the gradient-free baseline.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from grasp.optim.prior_ar1 import AR1Prior
from grasp.optim.surrogate import fit_quadratic_surrogate, make_dct_basis


def _jax_float_dtype():
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

BatchCostFn = Callable[[np.ndarray], np.ndarray]
KernelName = Literal["rwm", "surrogate_mala", "surrogate_nuts"]
KERNELS = ("rwm", "surrogate_mala", "surrogate_nuts")


def _ess_ratio_from_log_weights(log_weights: np.ndarray) -> float:
    shifted = np.asarray(log_weights, dtype=np.float64)
    shifted = shifted - np.max(shifted)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    return float(1.0 / np.sum(weights * weights) / weights.size)


def _next_temperature(
    temperature: float,
    log_likelihoods: np.ndarray,
    target_ess: float,
    *,
    bisection_steps: int = 48,
) -> float:
    """Choose the largest temperature step whose incremental ESS is targeted."""
    if temperature >= 1.0:
        return 1.0

    def ess_at(candidate: float) -> float:
        return _ess_ratio_from_log_weights(
            (candidate - temperature) * log_likelihoods
        )

    if ess_at(1.0) >= target_ess:
        return 1.0

    low = float(temperature)
    high = 1.0
    for _ in range(bisection_steps):
        middle = 0.5 * (low + high)
        if ess_at(middle) < target_ess:
            high = middle
        else:
            low = middle
    return float(low)


def _systematic_resample_np(
    rng: np.random.Generator, weights: np.ndarray
) -> np.ndarray:
    count = weights.shape[0]
    positions = (rng.uniform(0.0, 1.0 / count) + np.arange(count) / count)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def prior_full_factor(prior, horizon: int, control_dim: int) -> np.ndarray:
    """Per-(knot, channel) innovation std D of the window AR(1) prior x_k = rho x_{k-1} + D_k e_k, so that the prior
    covariance is L L^T with L = A^-1 diag(D) (A: unit lower bidiagonal with -rho below the diagonal), per channel.
    Knot 0 has the stationary std sigma_0 / sqrt(1 - rho^2) when the prior has no previous knot to condition on."""
    rho = float(prior.rho)
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        raise ValueError("prior_full requires abs(prior.rho) < 1")
    D = np.broadcast_to(np.asarray(prior.sigma_vec, dtype=np.float64), (horizon, control_dim)).copy()
    if getattr(prior, "u_prev", None) is None:
        D[0] /= np.sqrt(1.0 - rho * rho)
    if not np.all(np.isfinite(D)) or np.any(D <= 0):
        raise ValueError("prior_full requires positive finite innovation scales")
    return D


def prior_full_ops(D: np.ndarray, rho: float, sigma: float):
    """(metric_force, noise_transform, metric_norm_sq) of M = L L^T / sigma^2, L = A^-1 diag(D) per channel.
    Rows with D = 0 (a frozen knot 0) get no noise, no drift and are skipped by the norm."""
    K, m = D.shape

    def L_apply(w):
        W = w.reshape(-1, K, m); Y = np.empty_like(W); prev = np.zeros_like(W[:, 0])
        for k in range(K):
            prev = rho * prev + D[k] * W[:, k]; Y[:, k] = prev
        return Y.reshape(w.shape)

    def Lt_apply(v):
        V = v.reshape(-1, K, m); Z = np.empty_like(V); nxt = np.zeros_like(V[:, 0])
        for k in range(K - 1, -1, -1):
            nxt = V[:, k] + rho * nxt; Z[:, k] = nxt
        return (Z * D[None]).reshape(v.shape)

    live = D > 0

    def Linv_apply(r):
        R = r.reshape(-1, K, m); prev = np.concatenate([np.zeros_like(R[:, :1]), R[:, :-1]], axis=1)
        out = np.where(live[None], (R - rho * prev) / np.where(live, D, 1.0)[None], 0.0)
        return out.reshape(r.shape)

    return (lambda v: L_apply(Lt_apply(v)) / sigma**2,
            lambda xi: L_apply(xi) / sigma,
            lambda r: np.sum((Linv_apply(r) * sigma) ** 2, axis=1))


@dataclasses.dataclass
class SamplingResult:
    samples: np.ndarray
    costs: np.ndarray
    best_index: int
    info: dict


class GraspSampler:
    """Tempered SMC whose only simulator access is a host batch-cost call."""

    def __init__(
        self,
        batch_cost: BatchCostFn,
        prior: AR1Prior,
        horizon: int,
        control_dim: int = 1,
        beta: float = 1.0,
    ):
        self.batch_cost = batch_cost
        self._last_fit_costs = None
        # Optional (dimension,) bool: coordinates the cost never sees (e.g. a knot the executor overrides).  They must be
        # identical in every initial particle; surrogate_mala moves leave them fixed, so their prior term is a constant
        # and the free coordinates are conditioned on the true value.
        self.frozen_mask = None
        self.prior = prior
        self.horizon = int(horizon)
        self.control_dim = int(control_dim)
        self.dimension = self.horizon * self.control_dim
        self.beta = float(beta)
        self._invalid_evaluations = 0

        horizon_ = self.horizon
        control_dim_ = self.control_dim

        def log_prior_flat(position):
            if control_dim_ == 1:
                return prior.log_prob(position)
            return prior.log_prob(
                jnp.reshape(position, (horizon_, control_dim_))
            )

        self._log_prior_batch = jax.jit(jax.vmap(log_prior_flat))
        self._prior_score_batch = jax.jit(
            jax.vmap(jax.grad(log_prior_flat))
        )
        self._log_prior_flat = log_prior_flat

    # -- helpers -----------------------------------------------------------

    def _eval_costs(self, positions: np.ndarray) -> np.ndarray:
        raw_positions = np.asarray(positions, dtype=np.float64)
        valid_positions = np.all(np.isfinite(raw_positions), axis=1)
        # Do not feed non-finite controls to a batched simulator. Placeholder
        # evaluations retain the required batch shape and are always rejected.
        safe_positions = np.where(valid_positions[:, None], raw_positions, 0.0)
        costs = np.asarray(
            self.batch_cost(np.asarray(safe_positions, dtype=np.float32)),
            dtype=np.float64,
        )
        if costs.shape != (positions.shape[0],):
            raise ValueError(
                "batch_cost must return shape "
                f"({positions.shape[0]},), got {costs.shape}"
            )
        self._rollout_calls += int(positions.shape[0])
        invalid = ~valid_positions | ~np.isfinite(costs)
        self._invalid_evaluations += int(invalid.sum())
        costs = np.where(invalid, np.inf, costs)
        self._last_fit_costs = costs
        return costs

    def _log_prior(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._log_prior_batch(jnp.asarray(positions, dtype=_jax_float_dtype())),
            dtype=np.float64,
        )

    def _prior_score(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._prior_score_batch(
                jnp.asarray(positions, dtype=_jax_float_dtype())
            ),
            dtype=np.float64,
        )

    def _eval_surrogate(
        self,
        positions: np.ndarray,
        surrogate_params: tuple,
        basis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Values and gradients of the quadratic surrogate c + b^T y + y^T A y / 2, y = B^T (u - center) / scale."""
        center, scale, linear, quad = surrogate_params
        z = (positions - center[None, :]) / scale[None, :]
        y = z @ basis
        quad_y = y @ quad  # quad is symmetric
        values = y @ linear + 0.5 * np.sum(y * quad_y, axis=1)
        gradients = (linear[None, :] + quad_y) @ basis.T / scale[None, :]
        return values, gradients

    # -- main loop ---------------------------------------------------------

    def _crossfit_shrink(self, positions, costs, particles, center, scale, linear, quad, basis, *,
                         ridge, window, device, split_seed):
        """Shrink factor for the fitted gradient against estimation-noise inflation.

        With g_fit = g + e, the slope the surrogate claims along its own direction is |g_fit| while the
        true one is ~|g|^2/|g_fit|.  Two fits on disjoint random halves of the same rows have independent
        noise, so mean_particles(gA . gB) estimates |g|^2; c = that / mean |g_fit|^2, clipped to [0, 1].
        Returns (c, diagnostics).  The caller multiplies the whole quadratic by c (direction unchanged).
        """
        from grasp.optim.surrogate_torch import fit_quadratic_surrogate_torch
        full_basis = np.eye(positions.shape[1]) if basis is None else basis
        _, g_full = self._eval_surrogate(particles, (center, scale, linear, quad), full_basis)
        order = np.random.default_rng(split_seed).permutation(positions.shape[0])
        grads = []
        for idx in (np.sort(order[: len(order) // 2]), np.sort(order[len(order) // 2:])):
            lin_h, quad_h, _ = fit_quadratic_surrogate_torch(positions[idx], costs[idx], center, scale,
                                                             ridge=ridge, window=window, basis=basis, device=device)
            grads.append(self._eval_surrogate(particles, (center, scale, lin_h, quad_h), full_basis)[1])
        free = getattr(self, "_free_mask", None)
        if free is not None:
            g_full = g_full * free[None, :]
            grads = [g * free[None, :] for g in grads]
        signal = float(np.mean(np.sum(grads[0] * grads[1], axis=1)))
        total = float(np.mean(np.sum(g_full * g_full, axis=1)))
        factor = float(np.clip(signal / max(total, 1e-300), 0.0, 1.0))
        cos = float(np.mean(np.sum(grads[0] * grads[1], axis=1) / np.maximum(
            np.linalg.norm(grads[0], axis=1) * np.linalg.norm(grads[1], axis=1), 1e-300)))
        return factor, dict(signal=signal, total=total, half_cos=cos)

    def run(
        self,
        seed: int,
        num_particles: int = 100,
        *,
        kernel: KernelName = "surrogate_mala",
        target_ess: float = 0.8,
        num_mcmc_steps: int = 3,
        step_size: float = 0.1,
        nuts_mass: str = "identity",
        adapt_step_size: bool = False,
        adapt_step_target: float = 0.1,
        adapt_step_rate: float = 3.0,
        adapt_step_min_scale: float = 0.1,
        adapt_step_max_scale: float = 1.0,
        surrogate_ridge: float = 1e-3,
        surrogate_window: float = 3.0,
        surrogate_rank: int = 0,
        mala_preconditioner: str = "identity",
        database_max_points: int = 600_000,
        fit_max_points: int = 60_000,
        surrogate_fit_device: str = "cpu",
        nuts_max_doublings: int = 8,
        max_iterations: int = 100,
        initial_particles: np.ndarray | None = None,
        verbose: bool = True,
        surrogate_crossfit_shrink: bool = False,
        trace_callback=None,
    ) -> SamplingResult:
        """Temper from the prior (lambda = 0) to the target p0(u) exp(-beta J(u)) (lambda = 1).

        max_iterations = 0 means no stage limit.  The result's ``info["completed"]`` is False when the stage limit
        stopped the run before lambda reached 1.
        """
        kernel_name = str(kernel).lower()
        if kernel_name not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        self._mala_full = None
        if mala_preconditioner not in ("identity", "prior_full"):
            raise ValueError("mala_preconditioner must be 'identity' or 'prior_full'")
        if mala_preconditioner == "prior_full":
            # M = C / sigma^2 with C the window prior's own covariance (AR(1) correlation across knots AND the per
            # (knot, channel) innovation widths), so step_size stays in units of the scalar prior sigma and
            # proposals follow the prior's temporal correlation instead of perturbing each knot independently.
            if kernel_name != "surrogate_mala":
                raise ValueError("prior_full preconditioning requires surrogate_mala")
            self._mala_full = dict(D=prior_full_factor(self.prior, self.horizon, self.control_dim),
                                   rho=float(self.prior.rho), sigma=float(self.prior.sigma))
        if str(nuts_mass) not in ("identity", "prior_full"):
            raise ValueError("nuts_mass must be 'identity' or 'prior_full'")
        self._nuts_mass = str(nuts_mass)
        self._nuts_prior_full_inverse = None
        if surrogate_crossfit_shrink and (kernel_name not in ("surrogate_mala", "surrogate_nuts")
                                          or str(surrogate_fit_device) == "cpu"):
            # proposal-only refinement: where it does not apply the sampler runs without it
            if verbose:
                print("[crossfit-shrink] skipped: needs surrogate_mala/surrogate_nuts fitted on a torch device", flush=True)
            surrogate_crossfit_shrink = False
        adapt_step_size = bool(adapt_step_size)
        adapt_step_target = float(adapt_step_target)
        adapt_step_rate = float(adapt_step_rate)
        adapt_step_min_scale = float(adapt_step_min_scale)
        adapt_step_max_scale = float(adapt_step_max_scale)
        if adapt_step_max_scale < 1.0:
            raise ValueError("adapt_step_max_scale must be >= 1")
        if not 0.0 < adapt_step_target < 1.0:
            raise ValueError("adapt_step_target must be in (0, 1)")
        if adapt_step_rate <= 0.0:
            raise ValueError("adapt_step_rate must be positive")
        if not 0.0 < adapt_step_min_scale <= 1.0:
            raise ValueError("adapt_step_min_scale must be in (0, 1]")

        rng = np.random.default_rng(int(seed))
        jax_key = jax.random.PRNGKey(int(seed))
        self._rollout_calls = 0
        self._invalid_evaluations = 0

        if initial_particles is None:
            init_key = jax.random.fold_in(jax_key, 1)
            init_keys = jax.random.split(init_key, int(num_particles))
            particles = np.asarray(
                jax.vmap(
                    lambda k: self.prior.sample(
                        k, self.horizon, control_dim=self.control_dim
                    )
                )(init_keys),
                dtype=np.float64,
            ).reshape((int(num_particles), self.dimension))
        else:
            particles = np.asarray(
                initial_particles, dtype=np.float64
            ).reshape((-1, self.dimension))
            num_particles = int(particles.shape[0])
        particles = particles.copy()

        self._free_mask = None
        if self.frozen_mask is not None:
            frozen = np.asarray(self.frozen_mask, dtype=bool).reshape(-1)
            if frozen.shape != (self.dimension,) or kernel_name != "surrogate_mala":
                raise ValueError("frozen_mask needs surrogate_mala and shape (dimension,)")
            if self._mala_full is not None:
                fz = frozen.reshape(self.horizon, self.control_dim)
                if fz[1:].any() or not (fz[0].all() or not fz[0].any()):
                    raise ValueError("prior_full supports freezing whole knot 0 only")
                if fz[0].all():
                    # x_0 fixed: the free knots are the AR(1) conditional on it, innovations start at knot 1
                    self._mala_full["D"][0] = 0.0
            if not np.all(particles[:, frozen] == particles[:1, frozen]):
                raise ValueError("frozen coordinates must be identical in every initial particle")
            self._free_mask = (~frozen).astype(np.float64)
        if self._mala_full is not None:
            # fail fast: noise_transform, metric_force and metric_norm_sq must describe one PD metric
            mf, nt, nsq = prior_full_ops(self._mala_full["D"], self._mala_full["rho"], self._mala_full["sigma"])
            chk = np.random.default_rng(0).standard_normal((3, self.dimension))
            live = (self._mala_full["D"] > 0).reshape(-1)
            np.testing.assert_allclose(nsq(nt(chk)), np.sum(chk[:, live] ** 2, axis=1), rtol=1e-8)
            np.testing.assert_allclose(np.sum(chk[0] * mf(chk[1:2])[0]), np.sum(chk[1] * mf(chk[0:1])[0]), rtol=1e-8)
            # v.Mv = |L^T v|^2 / sigma^2; nt(I) has rows (L e_i)/sigma, i.e. it is (L/sigma)^T
            np.testing.assert_allclose(np.sum(chk[0] * mf(chk[:1])[0]),
                                       np.sum((nt(np.eye(self.dimension)) @ chk[0]) ** 2), rtol=1e-6)

        uses_surrogate = kernel_name != "rwm"
        basis: np.ndarray
        if uses_surrogate and 0 < int(surrogate_rank) * self.control_dim < (
            self.dimension
        ):
            basis = make_dct_basis(
                self.horizon, self.control_dim, int(surrogate_rank)
            )
        else:
            basis = np.eye(self.dimension)

        costs = self._eval_costs(particles)
        log_priors = self._log_prior(particles)
        initial_fit_costs = self._last_fit_costs.copy()
        invalid = ~np.isfinite(costs) | ~np.isfinite(log_priors)
        initial_invalid = int(invalid.sum())
        initial_resample_rounds = 0
        if invalid.all():
            raise RuntimeError("initial population has no finite candidates")
        # Rejection sampling from the prior conditioned on evaluability, not
        # cloning survivors. Keep valid incumbents and the fixed simulator N.
        while invalid.any():
            if initial_resample_rounds >= 16:
                raise RuntimeError("could not initialize a finite population after 16 resampling rounds")
            initial_resample_rounds += 1
            keys = jax.random.split(jax.random.fold_in(jax_key, 100000 + initial_resample_rounds), int(num_particles))
            fresh = np.asarray(jax.vmap(
                lambda key: self.prior.sample(key, self.horizon, control_dim=self.control_dim)
            )(keys), dtype=np.float64).reshape((int(num_particles), self.dimension))
            if self._free_mask is not None:
                frozen = self._free_mask == 0
                frozen_knots = frozen.reshape(self.horizon, self.control_dim)
                if frozen_knots[1:].any():
                    raise ValueError("invalid-particle resampling supports frozen coordinates in knot 0 only")
                # Condition the AR(1) draw on the pinned first knot, preserving
                # its subsequent innovations rather than only overwriting x0.
                chain = fresh.reshape(-1, self.horizon, self.control_dim)
                delta0 = (particles[0].reshape(self.horizon, self.control_dim)[0] - chain[:, 0]) * frozen_knots[0]
                chain += (float(self.prior.rho) ** np.arange(self.horizon))[None, :, None] * delta0[:, None, :]
                fresh[:, frozen] = particles[0, frozen]
            trial = np.where(invalid[:, None], fresh, particles)
            trial_costs = self._eval_costs(trial)
            trial_priors = self._log_prior(trial)
            recovered = invalid & np.isfinite(trial_costs) & np.isfinite(trial_priors)
            particles[recovered] = trial[recovered]
            costs[recovered] = trial_costs[recovered]
            log_priors[recovered] = trial_priors[recovered]
            initial_fit_costs[recovered] = self._last_fit_costs[recovered]
            invalid[recovered] = False
        self._last_fit_costs = initial_fit_costs
        if verbose and initial_invalid:
            print(f"[finite-candidates] initialization replaced {initial_invalid}/{num_particles} "
                  f"invalid particles in {initial_resample_rounds} rounds", flush=True)

        database_positions: list[np.ndarray] = []
        database_costs: list[np.ndarray] = []
        if uses_surrogate:
            database_positions.append(particles.copy())
            database_costs.append(self._last_fit_costs.copy())

        nuts_move = (self._build_nuts_move(int(nuts_max_doublings))
                     if kernel_name == "surrogate_nuts" else None)

        temperature = 0.0
        iteration = 0
        temperatures = [0.0]
        ess_values: list[float] = []
        acceptance_history: list[float] = []
        step_size_history: list[float] = []
        surrogate_r2_history: list[float] = []
        surrogate_shrink_history: list[float] = []
        surrogate_points_history: list[int] = []
        rollout_calls_history = [self._rollout_calls]
        cost_min_history = [float(np.min(costs))]
        cost_median_history = [float(np.median(costs))]
        adaptive_step_size = float(step_size)
        adaptive_step_min = float(step_size) * adapt_step_min_scale
        adaptive_step_max = float(step_size) * adapt_step_max_scale

        if verbose:
            print(
                f"[GRASP:{kernel_name}] N={num_particles}, "
                f"d={self.dimension}, initial costs="
                f"[{costs.min():.3f}, {np.median(costs):.3f}, "
                f"{costs.max():.3f}]"
            )

        while temperature < 1.0 - 1e-12 and (int(max_iterations) == 0 or iteration < int(max_iterations)):
            stage_started = time.perf_counter()
            log_likelihoods = -self.beta * costs
            next_temperature = _next_temperature(
                temperature, log_likelihoods, float(target_ess)
            )
            incremental = (next_temperature - temperature) * log_likelihoods
            ess_ratio = _ess_ratio_from_log_weights(incremental)
            shifted = np.exp(incremental - np.max(incremental))
            weights = shifted / np.sum(shifted)

            ancestors = _systematic_resample_np(rng, weights)
            if trace_callback is not None:
                trace_callback("resample", stage=iteration + 1,
                    temperature=temperature, next_temperature=next_temperature,
                    particles=particles, costs=costs, log_priors=log_priors,
                    weights=weights, ancestors=ancestors,
                    rng_state=rng.bit_generator.state)
            particles = particles[ancestors]
            costs = costs[ancestors]
            log_priors = log_priors[ancestors]

            surrogate_params = None
            fit_started = time.perf_counter()
            if uses_surrogate:
                center = np.mean(particles, axis=0)
                scale = np.maximum(np.std(particles, axis=0), 1e-3)
                fit_positions = np.concatenate(database_positions, axis=0)
                fit_costs = np.concatenate(database_costs, axis=0)
                fit_valid = np.isfinite(fit_costs) & np.all(np.isfinite(fit_positions), axis=1)
                fit_positions, fit_costs = fit_positions[fit_valid], fit_costs[fit_valid]
                if not len(fit_costs):
                    raise RuntimeError("surrogate fit has no finite observations")
                if fit_positions.shape[0] > int(fit_max_points):
                    chosen = rng.choice(
                        fit_positions.shape[0],
                        size=int(fit_max_points),
                        replace=False,
                    )
                    fit_positions = fit_positions[chosen]
                    fit_costs = fit_costs[chosen]
                fit_basis = (
                    None if basis.shape[0] == basis.shape[1] else basis
                )
                # diagnostic only: the data this stage's fit used.  Never read by the sampler itself.
                self._last_fit_data = (fit_positions, fit_costs, center, scale)
                if str(surrogate_fit_device) != "cpu":
                    # Same model and filters; the weighted least squares runs
                    # on a torch device so the fit no longer dominates a stage.
                    from grasp.optim.surrogate_torch import (
                        fit_quadratic_surrogate_torch,
                    )

                    linear, quad, r_squared = fit_quadratic_surrogate_torch(
                        fit_positions,
                        fit_costs,
                        center,
                        scale,
                        ridge=float(surrogate_ridge),
                        window=float(surrogate_window),
                        basis=fit_basis,
                        device=str(surrogate_fit_device),
                    )
                    if surrogate_crossfit_shrink:
                        shrink, _ = self._crossfit_shrink(
                            fit_positions, fit_costs, particles, center, scale, linear, quad, fit_basis,
                            ridge=float(surrogate_ridge), window=float(surrogate_window),
                            device=str(surrogate_fit_device), split_seed=[int(seed), iteration, 7])
                        surrogate_shrink_history.append(shrink)
                        linear = linear * shrink
                        quad = quad * shrink
                else:
                    linear, quad, r_squared = fit_quadratic_surrogate(
                        fit_positions,
                        fit_costs,
                        center,
                        scale,
                        ridge=float(surrogate_ridge),
                        window=float(surrogate_window),
                        basis=fit_basis,
                    )
                surrogate_params = (center, scale, linear, quad)
                surrogate_r2_history.append(r_squared)
                surrogate_points_history.append(
                    int(sum(b.shape[0] for b in database_positions))
                )

            fit_seconds = time.perf_counter() - fit_started
            stage_accepts: list[np.ndarray] = []
            t_next = float(next_temperature)
            stage_step_size = (
                adaptive_step_size if adapt_step_size else float(step_size)
            )

            for move_index in range(int(num_mcmc_steps)):
                if trace_callback is not None:
                    trace_callback("move_rng", stage=iteration + 1,
                        move=move_index, rng_state=rng.bit_generator.state,
                        jax_key=jax_key)
                if trace_callback is not None and move_index == 0:
                    trace_callback("stage_start", stage=iteration + 1,
                        temperature=t_next, step_size=stage_step_size,
                        particles=particles, costs=costs, log_priors=log_priors,
                        basis=basis, surrogate_params=surrogate_params,
                        rng_state=rng.bit_generator.state, jax_key=jax_key,
                        mala_preconditioner=mala_preconditioner)
                proposals, extras = self._propose(
                    kernel_name,
                    particles,
                    t_next,
                    rng,
                    jax_key,
                    iteration,
                    move_index,
                    step_size=stage_step_size,
                    surrogate_params=surrogate_params,
                    basis=basis,
                    nuts_move=nuts_move,
                )
                proposal_costs = self._eval_costs(proposals)
                proposal_fit_costs = self._last_fit_costs
                proposal_log_priors = self._log_prior(proposals)

                proposal_valid = (np.isfinite(proposal_costs) & np.isfinite(proposal_log_priors)
                                  & np.all(np.isfinite(proposals), axis=1))
                # Avoid 0 * inf at beta=0; an invalid proposal is rejected
                # explicitly regardless of its MH arithmetic or kernel.
                log_alpha = self._log_alpha(
                    kernel_name,
                    t_next,
                    costs,
                    np.where(proposal_valid, proposal_costs, costs),
                    log_priors,
                    np.where(proposal_valid, proposal_log_priors, log_priors),
                    extras,
                )
                log_alpha = np.where(proposal_valid, log_alpha, -np.inf)
                log_uniform = np.log(
                    rng.uniform(size=num_particles).clip(1e-38)
                )
                accept = proposal_valid & (log_uniform < np.where(np.isfinite(log_alpha), log_alpha, -np.inf))

                if trace_callback is not None:
                    trace_callback("move", stage=iteration + 1, move=move_index,
                        temperature=t_next, step_size=stage_step_size,
                        particles=particles, costs=costs, log_priors=log_priors,
                        proposals=proposals, proposal_costs=proposal_costs,
                        proposal_log_priors=proposal_log_priors, extras=extras,
                        accept=accept, log_alpha=log_alpha, log_uniform=log_uniform,
                        rng_state=rng.bit_generator.state)
                particles = np.where(accept[:, None], proposals, particles)
                costs = np.where(accept, proposal_costs, costs)
                log_priors = np.where(
                    accept, proposal_log_priors, log_priors
                )
                stage_accepts.append(accept.astype(np.float64))

                if uses_surrogate:
                    finite = proposal_valid & np.isfinite(proposal_fit_costs) & np.all(
                        np.isfinite(proposals), axis=1
                    )
                    if np.any(finite):
                        database_positions.append(proposals[finite].copy())
                        database_costs.append(proposal_fit_costs[finite].copy())
                    total_points = sum(
                        b.shape[0] for b in database_positions
                    )
                    while (
                        total_points > int(database_max_points)
                        and len(database_positions) > 1
                    ):
                        dropped = database_positions.pop(0)
                        database_costs.pop(0)
                        total_points -= dropped.shape[0]

            temperature = t_next
            iteration += 1
            temperatures.append(temperature)
            ess_values.append(ess_ratio)
            mean_accept = float(np.mean(np.concatenate(stage_accepts)))
            acceptance_history.append(mean_accept)
            step_size_history.append(stage_step_size)
            if adapt_step_size:
                adaptive_step_size = float(
                    np.clip(
                        stage_step_size
                        * np.exp(
                            adapt_step_rate
                            * (mean_accept - adapt_step_target)
                        ),
                        adaptive_step_min,
                        adaptive_step_max,
                    )
                )
            rollout_calls_history.append(self._rollout_calls)
            cost_min_history.append(float(np.min(costs)))
            cost_median_history.append(float(np.median(costs)))

            if verbose:
                extra_text = ""
                if uses_surrogate:
                    extra_text = (
                        f", R2={surrogate_r2_history[-1]:.3f}, "
                        f"pts={surrogate_points_history[-1]}"
                    )
                    if surrogate_shrink_history:
                        extra_text += f", shrink={surrogate_shrink_history[-1]:.3f}"
                print(
                    f"  iter={iteration:3d}, temp={temperature:.6f}, "
                    f"step={stage_step_size:.3e}, "
                    f"ESS={ess_ratio:.3f}, accept={mean_accept:.3f}, "
                    f"costs=[{costs.min():.3f}, "
                    f"{np.median(costs):.3f}, {costs.max():.3f}]"
                    f"{extra_text}, t_fit={fit_seconds:.1f}s, "
                    f"t_stage={time.perf_counter() - stage_started:.1f}s"
                )

        info = {
            "invalid_evaluations": int(self._invalid_evaluations),
            "initial_invalid_particles": initial_invalid,
            "initial_resample_rounds": initial_resample_rounds,
            "kernel": kernel_name,
            "temperatures": np.asarray(temperatures, dtype=np.float32),
            "ess_values": np.asarray(ess_values, dtype=np.float32),
            "acceptance_rates": np.asarray(
                acceptance_history, dtype=np.float32
            ),
            "step_sizes": np.asarray(step_size_history, dtype=np.float32),
            "rollout_calls": np.asarray(
                rollout_calls_history, dtype=np.int64
            ),
            "total_rollout_calls": int(self._rollout_calls),
            "cost_min_history": np.asarray(
                cost_min_history, dtype=np.float32
            ),
            "cost_median_history": np.asarray(
                cost_median_history, dtype=np.float32
            ),
            "num_iterations": int(iteration),
            "completed": bool(temperature >= 1.0 - 1e-12),
            "surrogate_rank": (
                int(surrogate_rank) if uses_surrogate else None
            ),
            "surrogate_r2": np.asarray(
                surrogate_r2_history, dtype=np.float32
            ),
            "surrogate_shrink": np.asarray(surrogate_shrink_history, dtype=np.float64),
            "surrogate_points": np.asarray(
                surrogate_points_history, dtype=np.int64
            ),
            "mala_preconditioner": mala_preconditioner,
            "cost_stats": {
                "min": float(np.min(costs)),
                "max": float(np.max(costs)),
                "mean": float(np.mean(costs)),
                "median": float(np.median(costs)),
                "std": float(np.std(costs)),
            },
        }

        self._free_mask = None
        return SamplingResult(
            samples=particles.astype(np.float32),
            costs=costs.copy(),
            best_index=int(np.argmin(costs)),
            info=info,
        )

    # -- proposal machinery -------------------------------------------------

    def _nuts_inverse_mass(self):
        """Inverse mass matrix of the inner NUTS chain: identity, or the prior covariance in units of the scalar prior
        sigma (``prior_full``).  The inner chain is reversible with respect to the analytic surrogate target for any
        mass matrix, and the outer delayed-acceptance ratio does not contain it, so the sampled target is unchanged."""
        if self._nuts_mass == "identity":
            return np.ones(self.dimension, dtype=np.float64)
        # M^-1 = C / sigma^2: the NUTS counterpart of surrogate_mala's prior_full metric.  Constant across stages,
        # so built once.
        if self._nuts_prior_full_inverse is None:
            if getattr(self.prior, "sigma_vec", None) is not None:
                D = prior_full_factor(self.prior, self.horizon, self.control_dim)
            else:   # scalar stationary AR(1): x_0 at the stationary std, then sigma innovations
                D = np.full((self.horizon, self.control_dim), float(self.prior.sigma))
                D[0] /= np.sqrt(1.0 - float(self.prior.rho) ** 2)
            metric_force, _, _ = prior_full_ops(D, float(self.prior.rho), float(self.prior.sigma))
            inverse = metric_force(np.eye(self.dimension))
            self._nuts_prior_full_inverse = 0.5 * (inverse + inverse.T)
        return self._nuts_prior_full_inverse

    def _build_nuts_move(self, max_doublings: int):
        from blackjax.mcmc import nuts as blackjax_nuts

        log_prior_flat = self._log_prior_flat
        beta = self.beta
        kernel = blackjax_nuts.build_kernel()

        def single(rng_key, position, temperature, step, center, scale,
                   linear, quad, basis, inv_mass):
            def surrogate_cost(value):
                reduced = basis.T @ ((value - center) / scale)
                return linear @ reduced + 0.5 * reduced @ (quad @ reduced)

            def logdensity(value):
                return log_prior_flat(value) - temperature * beta * (
                    surrogate_cost(value)
                )

            state = blackjax_nuts.init(position, logdensity)
            new_state, _ = kernel(
                rng_key,
                state,
                logdensity,
                step_size=step,
                inverse_mass_matrix=inv_mass,
                max_num_doublings=max_doublings,
            )
            return new_state.position

        return jax.jit(
            jax.vmap(
                single,
                in_axes=(0, 0, None, None, None, None, None, None, None,
                         None),
            )
        )

    def _propose(
        self,
        kernel_name: str,
        particles: np.ndarray,
        temperature: float,
        rng: np.random.Generator,
        jax_key,
        iteration: int,
        move_index: int,
        *,
        step_size: float,
        surrogate_params,
        basis: np.ndarray,
        nuts_move,
    ) -> tuple[np.ndarray, dict]:
        count = particles.shape[0]
        tb = temperature * self.beta

        if kernel_name == "rwm":
            proposals = particles + step_size * rng.standard_normal(
                particles.shape
            )
            return proposals, {}

        if kernel_name == "surrogate_mala":
            full = self._mala_full
            if full is not None:
                metric_force, noise_transform, metric_norm_sq = prior_full_ops(full["D"], full["rho"], full["sigma"])
            else:
                metric_force = noise_transform = lambda values: values
                metric_norm_sq = lambda values: np.sum(values**2, axis=1)

            free = getattr(self, "_free_mask", None)
            keep = (lambda v: v) if free is None else (lambda v: v * free[None, :])
            surrogate_here, grad_here = self._eval_surrogate(
                particles, surrogate_params, basis
            )
            force = keep(self._prior_score(particles) - tb * grad_here)
            mean_fwd = particles + 0.5 * step_size**2 * metric_force(force)
            proposals = mean_fwd + step_size * keep(noise_transform(rng.standard_normal(
                particles.shape
            )))
            surrogate_there, grad_there = self._eval_surrogate(
                proposals, surrogate_params, basis
            )
            force_rev = keep(self._prior_score(proposals) - tb * grad_there)
            mean_rev = proposals + 0.5 * step_size**2 * metric_force(force_rev)
            residual_fwd = proposals - mean_fwd
            residual_rev = particles - mean_rev
            return proposals, {
                "surrogate_here": surrogate_here,
                "surrogate_there": surrogate_there,
                # Shared Gaussian normalization cancels in log q_rev/q_fwd.
                "log_q_fwd": -0.5 / step_size**2 * metric_norm_sq(residual_fwd),
                "log_q_rev": -0.5 / step_size**2 * metric_norm_sq(residual_rev),
            }

        if kernel_name == "surrogate_nuts":
            move_key = jax.random.fold_in(
                jax.random.fold_in(jax_key, 1000 + iteration), move_index
            )
            move_keys = jax.random.split(move_key, count)
            dtype = _jax_float_dtype()
            center, scale, linear, quad = surrogate_params
            proposals = np.asarray(
                nuts_move(
                    move_keys,
                    jnp.asarray(particles, dtype=dtype),
                    jnp.asarray(temperature, dtype=dtype),
                    jnp.asarray(step_size, dtype=dtype),
                    jnp.asarray(center, dtype=dtype),
                    jnp.asarray(scale, dtype=dtype),
                    jnp.asarray(linear, dtype=dtype),
                    jnp.asarray(quad, dtype=dtype),
                    jnp.asarray(basis, dtype=dtype),
                    jnp.asarray(self._nuts_inverse_mass(), dtype=dtype),
                ),
                dtype=np.float64,
            )
            sur_here, _ = self._eval_surrogate(
                particles, surrogate_params, basis
            )
            sur_there, _ = self._eval_surrogate(
                proposals, surrogate_params, basis
            )
            return proposals, {
                "surrogate_here": sur_here,
                "surrogate_there": sur_there,
            }

        raise AssertionError(kernel_name)

    def _log_alpha(
        self,
        kernel_name: str,
        temperature: float,
        costs: np.ndarray,
        proposal_costs: np.ndarray,
        log_priors: np.ndarray,
        proposal_log_priors: np.ndarray,
        extras: dict,
    ) -> np.ndarray:
        tb = temperature * self.beta
        target_diff = (proposal_log_priors - tb * proposal_costs) - (
            log_priors - tb * costs
        )
        if kernel_name == "rwm":
            return target_diff
        if kernel_name == "surrogate_mala":
            return target_diff + extras["log_q_rev"] - extras["log_q_fwd"]
        if kernel_name == "surrogate_nuts":
            # Delayed acceptance: prior terms cancel between the true and
            # surrogate tempered targets, leaving the value-difference ratio.
            return tb * (
                (costs - proposal_costs)
                - (extras["surrogate_here"] - extras["surrogate_there"])
            )
        raise AssertionError(kernel_name)
