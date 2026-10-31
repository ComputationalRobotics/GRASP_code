# algorithms/tempered_smc.py
"""Adaptive Tempered Sequential Monte Carlo (SMC) sampler."""

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

from benchmarks.baselines.base import SamplingAlgorithm, SamplingResult
from benchmarks.systems.base import DynamicalSystem
from grasp.optim.prior_ar1 import AR1Prior


def compute_ess_from_log_weights(log_weights):
    """Compute ESS from log weights."""
    log_weights_norm = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_norm)
    ess = 1.0 / jnp.sum(weights ** 2)
    return ess


class TemperedSMCSampler(SamplingAlgorithm):
    """Adaptive Tempered SMC with HMC or NUTS as inner kernel."""
    
    def __init__(self, system: DynamicalSystem, prior: AR1Prior, beta: float = 1.0):
        super().__init__(system, prior, beta)
    
    @property
    def name(self) -> str:
        return "Tempered-SMC"
    
    def run(self, key: jax.Array, H: int,
            num_particles: int = 1000,
            kernel: str = "nuts",
            target_ess: float = 0.5,
            num_mcmc_steps: int = 10,
            step_size: float = 0.01,
            num_integration_steps: int = 5,
            max_num_doublings: int = 8,
            max_iterations: int = 100,
            initial_particles: jax.Array | None = None,
            record_energies: bool = False,
            sur_leapfrog: int = 1,
            ramp_model: str = "quad",
            ramp_eps: float = 0.01,
            ramp_dct_r: int = 8,
            ramp_fifo_cap: int = 4000,
            zo_directions: int = 8,
            zo_sigma: float = 0.05,
            lr_rollouts: int = 8,
            lr_action_std: float = 0.3) -> SamplingResult:
        """Run adaptive tempered SMC.
        
        Args:
            key: JAX random key
            H: Horizon
            num_particles: Number of particles
            kernel: "hmc" or "nuts"
            target_ess: Target ESS ratio (0-1)
            num_mcmc_steps: MCMC steps per SMC iteration
            step_size: HMC/NUTS step size
            num_integration_steps: Leapfrog steps (HMC only)
            max_num_doublings: Max tree depth (NUTS only)
            max_iterations: Maximum SMC iterations
            record_energies: If True, record per-particle costs at each tempering level
        """
        print("\n" + "="*60)
        print(f"Running {self.name} ({kernel.upper()}) for {self.system.name}...")
        print("="*60)
        
        log_prior_fn = self.make_log_prior_fn()
        log_likelihood_fn = self.make_log_likelihood_fn()
        
        # Initialize particles from prior (or reuse provided initial_particles for fair comparisons)
        if initial_particles is None:
            key, init_key = jax.random.split(key)
            init_keys = jax.random.split(init_key, num_particles)
            control_dim = int(getattr(self.system, "control_dim", 1))
            init = jax.vmap(lambda k: self.prior.sample(k, H, control_dim=control_dim))(init_keys)
            # BlackJAX expects vector positions; flatten (H,Du) -> (H*Du,)
            initial_particles = init.reshape((init.shape[0], -1))
            print(f"[{self.name}] Generated {num_particles} initial particles from prior")
        else:
            initial_particles = jnp.asarray(initial_particles)
            # Accept (N,H), (N,H,Du), or already-flattened (N,H*Du)
            if initial_particles.ndim == 3:
                if initial_particles.shape[1] != H:
                    raise ValueError(
                        f"initial_particles must have shape (N, {H}, Du), got {initial_particles.shape}"
                    )
                initial_particles = initial_particles.reshape((initial_particles.shape[0], -1))
            elif initial_particles.ndim == 2:
                if initial_particles.shape[1] != H and (initial_particles.shape[1] % H) != 0:
                    raise ValueError(
                        f"initial_particles must have shape (N, {H}) or (N, {H}*Du), got {initial_particles.shape}"
                    )
            else:
                raise ValueError(
                    f"initial_particles must have ndim 2 or 3, got {initial_particles.shape}"
                )
            num_particles = int(initial_particles.shape[0])
            print(f"[{self.name}] Using provided initial_particles: N={num_particles}")
        
        kname = kernel.lower()
        if kname == "nuts":
            final_particles, smc_info = self._run_nuts(
                key, log_prior_fn, log_likelihood_fn, initial_particles,
                target_ess, num_mcmc_steps, step_size, max_num_doublings, max_iterations,
                record_energies
            )
        elif kname == "hmc":
            final_particles, smc_info = self._run_hmc(
                key, log_prior_fn, log_likelihood_fn, initial_particles,
                target_ess, num_mcmc_steps, step_size, num_integration_steps, max_iterations,
                record_energies
            )
        elif kname == "mala":
            final_particles, smc_info = self._run_mala(
                key, log_prior_fn, log_likelihood_fn, initial_particles,
                target_ess, num_mcmc_steps, step_size, max_iterations,
                record_energies
            )
        elif kname in ("rampq", "rampmlp"):
            final_particles, smc_info = self._run_ramp(
                key, log_prior_fn, log_likelihood_fn, initial_particles,
                target_ess, num_mcmc_steps, max_iterations, record_energies,
                model=("mlp" if kname == "rampmlp" else "quad"),
                eps=float(ramp_eps), dct_r=int(ramp_dct_r),
                fifo_cap=int(ramp_fifo_cap),
            )
        elif kname in ("surzo", "surlr"):
            final_particles, smc_info = self._run_surrogate(
                key, log_prior_fn, log_likelihood_fn, initial_particles,
                target_ess, num_mcmc_steps, step_size, max_iterations,
                record_energies, force=kname[3:], sur_leapfrog=int(sur_leapfrog),
                zo_directions=int(zo_directions), zo_sigma=float(zo_sigma),
                lr_rollouts=int(lr_rollouts), lr_action_std=float(lr_action_std),
            )
        else:
            raise ValueError(f"Unknown kernel: {kernel}. Use 'nuts', 'hmc', 'mala', 'surzo' or 'surlr'.")
        
        # Pick best
        best_Uraw, best_cost = self.pick_best(final_particles)
        U, X = self.rollout_best(best_Uraw)
        
        # Compute cost statistics
        cost_fn = self.make_cost_evaluator()
        all_costs = cost_fn(final_particles)
        
        info_dict = {
            "temperatures": smc_info["temperatures"],
            "ess_values": smc_info.get("ess_values", []),
            "num_iterations": smc_info["num_iterations"],
            "kernel": kernel.upper(),
            "cost_stats": {
                "min": float(jnp.min(all_costs)),
                "max": float(jnp.max(all_costs)),
                "mean": float(jnp.mean(all_costs)),
                "median": float(jnp.median(all_costs)),
                "std": float(jnp.std(all_costs)),
            }
        }
        if "energy_values" in smc_info:
            info_dict["energy_values"] = smc_info["energy_values"]
        
        return SamplingResult(
            U=U,
            X=X,
            cost=best_cost,
            samples=np.array(final_particles),
            info=info_dict
        )
    
    def _run_hmc(self, key, log_prior_fn, log_likelihood_fn, initial_particles,
                 target_ess, num_mcmc_steps, step_size, num_integration_steps, max_iterations,
                 record_energies):
        """Run SMC with HMC kernel."""
        num_particles = initial_particles.shape[0]
        dim = int(initial_particles.shape[1])
        
        hmc_parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": jnp.ones(dim),
            "num_integration_steps": num_integration_steps,
        }
        
        smc = blackjax.adaptive_tempered_smc(
            log_prior_fn,
            log_likelihood_fn,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            extend_params(hmc_parameters),
            resampling.systematic,
            target_ess,
            num_mcmc_steps=num_mcmc_steps,
        )
        
        return self._run_smc_loop(key, smc, initial_particles, log_likelihood_fn,
                                   num_particles, max_iterations, "HMC", record_energies)
    
    def _run_nuts(self, key, log_prior_fn, log_likelihood_fn, initial_particles,
                  target_ess, num_mcmc_steps, step_size, max_num_doublings, max_iterations,
                  record_energies):
        """Run SMC with NUTS kernel."""
        num_particles = initial_particles.shape[0]
        dim = int(initial_particles.shape[1])
        
        nuts_parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": jnp.ones(dim),
            "max_num_doublings": max_num_doublings,
        }
        
        smc = blackjax.adaptive_tempered_smc(
            log_prior_fn,
            log_likelihood_fn,
            blackjax.nuts.build_kernel(),
            blackjax.nuts.init,
            extend_params(nuts_parameters),
            resampling.systematic,
            target_ess,
            num_mcmc_steps=num_mcmc_steps,
        )
        
        return self._run_smc_loop(key, smc, initial_particles, log_likelihood_fn,
                                   num_particles, max_iterations, "NUTS", record_energies)

    def _run_mala(self, key, log_prior_fn, log_likelihood_fn, initial_particles,
                  target_ess, num_mcmc_steps, step_size, max_iterations,
                  record_energies):
        """Run SMC with MALA kernel (Metropolis-adjusted Langevin algorithm).

        MALA uses a single gradient evaluation per MCMC step (no trajectories),
        often more memory-friendly than NUTS/HMC for expensive rollouts.
        """
        num_particles = initial_particles.shape[0]

        mala_parameters = {
            "step_size": step_size,
        }

        smc = blackjax.adaptive_tempered_smc(
            log_prior_fn,
            log_likelihood_fn,
            blackjax.mala.build_kernel(),
            blackjax.mala.init,
            extend_params(mala_parameters),
            resampling.systematic,
            target_ess,
            num_mcmc_steps=num_mcmc_steps,
        )

        return self._run_smc_loop(key, smc, initial_particles, log_likelihood_fn,
                                  num_particles, max_iterations, "MALA", record_energies)
    
    def _run_surrogate(self, key, log_prior_fn, log_likelihood_fn, initial_particles,
                       target_ess, num_mcmc_steps, step_size, max_iterations,
                       record_energies, force, sur_leapfrog,
                       zo_directions, zo_sigma, lr_rollouts, lr_action_std):
        """Manual adaptive tempered SMC with the surrogate-force HMC kernel.

        Mirrors TSMC_BlackBox_Retargeting_Extensions.pdf: eq (12) zo / eq (51) lr forces
        (zeta drawn once per proposal and frozen through the whole trajectory, sec 3.3),
        generalized leapfrog eq (20)-(21) with MH acceptance from the EXACT Hamiltonian
        at the endpoints (sec 3.4). Manual loop because blackjax's adaptive TSMC hides
        the temperature from the kernel. lr note: this cost has temporal-coupling switch
        penalties, so per-step cost-to-go is undefined; the trajectory-level LOO-baselined
        score is used instead (still unbiased for grad E[J]).
        """
        particles = jnp.asarray(initial_particles, dtype=jnp.float32)
        n, d = int(particles.shape[0]), int(particles.shape[1])
        sig_a = jnp.asarray(float(lr_action_std), dtype=jnp.float32)
        beta32 = jnp.asarray(float(self.beta), dtype=jnp.float32)
        grad_log_prior = jax.grad(log_prior_fn)

        def zo_grad_negloglik(u, kz):
            eps_dirs = jax.random.normal(kz, (int(zo_directions), d), dtype=jnp.float32)

            def pair(e):
                return (log_likelihood_fn(u + float(zo_sigma) * e)
                        - log_likelihood_fn(u - float(zo_sigma) * e))

            diffs = jax.vmap(pair)(eps_dirs)
            ghat_ll = (diffs[:, None] * eps_dirs).sum(0) / (
                2.0 * int(zo_directions) * float(zo_sigma))
            return -ghat_ll

        def lr_grad_negloglik(u, kz):
            B = int(lr_rollouts)
            eps = jax.random.normal(kz, (B, d), dtype=jnp.float32)
            a = u[None, :] + sig_a * eps
            J = -jax.vmap(log_likelihood_fn)(a) / beta32
            base = (jnp.sum(J) - J) / max(B - 1, 1)
            adv = J - base
            ghat_cost = (adv[:, None] * (a - u[None, :])).sum(0) / (B * sig_a ** 2)
            return beta32 * ghat_cost

        grad_negloglik = zo_grad_negloglik if force == "zo" else lr_grad_negloglik
        eps_lf = jnp.asarray(float(step_size), dtype=jnp.float32)
        L = int(sur_leapfrog)

        def one_step(rng, u0, temp32):
            kz, km, ka = jax.random.split(rng, 3)

            def gforce(u):
                return -grad_log_prior(u) + temp32 * grad_negloglik(u, kz)

            def V(u):
                return -(log_prior_fn(u) + temp32 * log_likelihood_fn(u))

            p0 = jax.random.normal(km, (d,), dtype=jnp.float32)

            def leap(carry, _):
                u, p = carry
                p = p - 0.5 * eps_lf * gforce(u)
                u = u + eps_lf * p
                p = p - 0.5 * eps_lf * gforce(u)
                return (u, p), None

            (u1, p1), _ = jax.lax.scan(leap, (u0, p0), None, length=L)
            dH = (V(u1) + 0.5 * jnp.sum(p1 * p1)) - (V(u0) + 0.5 * jnp.sum(p0 * p0))
            dH = jnp.nan_to_num(dH, nan=jnp.inf, posinf=jnp.inf)
            ok = jnp.log(jax.random.uniform(ka, (), minval=1e-12)) < -dH
            return jnp.where(ok, u1, u0), ok.astype(jnp.float32)

        sweep = jax.jit(jax.vmap(one_step, in_axes=(0, 0, None)))
        loglik_all = jax.jit(jax.vmap(log_likelihood_fn))

        kname = f"SUR{force.upper()}-L{L}"
        temperatures = [0.0]
        ess_values = []
        energies_history = []
        iteration = 0
        temp = 0.0

        init_costs = -loglik_all(particles) / self.beta
        print(f"  [SMC-{kname}] Initial particle costs: min={float(jnp.min(init_costs)):.1f}, "
              f"max={float(jnp.max(init_costs)):.1f}, median={float(jnp.median(init_costs)):.1f}")
        if record_energies:
            energies_history.append(np.array(init_costs))

        while temp < 1.0 and iteration < max_iterations:
            ll = loglik_all(particles)

            def ess_ratio_at(t_next):
                ilw = (t_next - temp) * ll
                return float(compute_ess_from_log_weights(ilw)) / n

            if ess_ratio_at(1.0) >= float(target_ess):
                t_next = 1.0
            else:
                lo, hi = float(temp), 1.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if ess_ratio_at(mid) < float(target_ess):
                        hi = mid
                    else:
                        lo = mid
                t_next = lo

            ilw = (t_next - float(temp)) * ll
            ess = float(compute_ess_from_log_weights(ilw)) / n
            w = jnp.exp(ilw - jax.scipy.special.logsumexp(ilw))
            key, rs_key = jax.random.split(key)
            idx = resampling.systematic(rs_key, w, n)
            particles = particles[idx]

            temp32 = jnp.asarray(t_next, dtype=jnp.float32)
            accs = []
            for _ in range(int(num_mcmc_steps)):
                key, sk = jax.random.split(key)
                ks = jax.random.split(sk, n)
                particles, ok = sweep(ks, particles, temp32)
                accs.append(float(jnp.mean(ok)))

            temp = float(t_next)
            iteration += 1
            temperatures.append(temp)
            ess_values.append(ess)
            costs_after = -loglik_all(particles) / self.beta
            if record_energies:
                energies_history.append(np.array(costs_after))
            print(f"  [SMC-{kname}] Iter {iteration:3d}: λ={temp:.4f}, ESS={ess:.3f}, "
                  f"acc={float(np.mean(accs)):.3f}, "
                  f"costs=[{float(jnp.min(costs_after)):.1f}, {float(jnp.median(costs_after)):.1f}, "
                  f"{float(jnp.max(costs_after)):.1f}]")

        print(f"  [SMC-{kname}] Completed in {iteration} iterations. Final λ = {temp:.4f}")
        info = {
            "temperatures": temperatures,
            "ess_values": ess_values,
            "num_iterations": iteration,
        }
        if record_energies:
            info["energy_values"] = np.asarray(energies_history, dtype=np.float32)
        return particles, info

    def _run_ramp(self, key, log_prior_fn, log_likelihood_fn, initial_particles,
                  target_ess, num_mcmc_steps, max_iterations, record_energies,
                  model, eps, dct_r, fifo_cap):
        """RAMP-TSMC (black_box_tsmc_paper Algorithm 1 + RAMP-MALA eq (accept)/(proposal)).

        Per stage: standardize by the current cloud, project to the dct_r lowest DCT
        frequencies per channel, fit Jhat on the FIFO rollout buffer (quadratic by
        weighted ridge, or a small MLP), freeze it; RAMP-MALA proposes with the
        surrogate drift and accepts with the TRUE rollout cost + full q-correction.
        Every proposal's (u', J') enters the buffer, accepted or not.
        Defaults NOT specified by the draft (my choices, exposed as flags):
        dct_r=8, eps=0.01, fifo_cap=4000, top-5% J outlier trim, ridge=1e-3.
        """
        particles = jnp.asarray(initial_particles, dtype=jnp.float32)
        n, d = int(particles.shape[0]), int(particles.shape[1])
        Du = int(getattr(self.system, "control_dim", 1))
        H = d // Du
        r = int(dct_r)
        m = r * Du
        beta = float(self.beta)
        eps_cur = float(eps)  # adapted per stage toward acc ~ 0.55 (prev-stage info only)
        grad_log_prior = jax.grad(log_prior_fn)
        loglik_all = jax.jit(jax.vmap(log_likelihood_fn))

        # orthonormal DCT-II basis, r lowest temporal frequencies per channel
        t_idx = np.arange(H)
        Bt = np.zeros((H, r), dtype=np.float32)
        for k_f in range(r):
            v = np.cos(np.pi * (t_idx + 0.5) * k_f / H)
            v *= (np.sqrt(1.0 / H) if k_f == 0 else np.sqrt(2.0 / H))
            Bt[:, k_f] = v
        Bt = jnp.asarray(Bt)

        def project(z_flat):
            z = z_flat.reshape((H, Du))
            return jnp.einsum("tk,tc->kc", Bt, z).reshape((m,))

        def lift(y_flat):
            y = y_flat.reshape((r, Du))
            return jnp.einsum("tk,kc->tc", Bt, y).reshape((d,))

        # FIFO rollout buffer
        J_cur = -loglik_all(particles) / beta
        buf_U = [np.array(particles)]
        buf_J = [np.array(J_cur)]

        kname = f"RAMP-{model.upper()}"
        temperatures = [0.0]
        ess_values = []
        energies_history = []
        iteration = 0
        temp = 0.0
        print(f"  [SMC-{kname}] Initial particle costs: min={float(jnp.min(J_cur)):.1f}, "
              f"max={float(jnp.max(J_cur)):.1f}, median={float(jnp.median(J_cur)):.1f}")
        if record_energies:
            energies_history.append(np.array(J_cur))

        mlp_params = None  # warm-start across stages

        while temp < 1.0 and iteration < max_iterations:
            ll = -beta * J_cur

            def ess_ratio_at(t_next):
                return float(compute_ess_from_log_weights((t_next - temp) * ll)) / n

            if ess_ratio_at(1.0) >= float(target_ess):
                t_next = 1.0
            else:
                lo, hi = float(temp), 1.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if ess_ratio_at(mid) < float(target_ess):
                        hi = mid
                    else:
                        lo = mid
                t_next = lo
            ilw = (t_next - float(temp)) * ll
            ess = float(compute_ess_from_log_weights(ilw)) / n
            w = jnp.exp(ilw - jax.scipy.special.logsumexp(ilw))
            key, rs_key = jax.random.split(key)
            idx = resampling.systematic(rs_key, w, n)
            particles = particles[idx]
            J_cur = J_cur[idx]

            # ---- fit surrogate on the buffer, frozen for this stage ----
            U_all = np.concatenate(buf_U, axis=0)[-int(fifo_cap):]
            J_all = np.concatenate(buf_J, axis=0)[-int(fifo_cap):]
            keep = J_all <= np.quantile(J_all, 0.95)  # outlier trim (my choice)
            U_fit = jnp.asarray(U_all[keep])
            J_fit = jnp.asarray(J_all[keep], dtype=jnp.float32)
            center = jnp.mean(particles, axis=0)
            scale = jnp.clip(jnp.std(particles, axis=0), 1e-3, None)
            Y_fit = jax.vmap(lambda u: project((u - center) / scale))(U_fit)  # (K,m)
            Jm, Js = jnp.mean(J_fit), jnp.clip(jnp.std(J_fit), 1e-6, None)
            Jn = (J_fit - Jm) / Js

            if model == "quad":
                iu, ju = np.triu_indices(m)
                sym = jnp.asarray((iu != ju) * 1.0 + 1.0, dtype=jnp.float32)  # 2 off-diag, 1 diag

                def feats(y):
                    return jnp.concatenate([jnp.ones((1,)), y, y[iu] * y[ju]])

                F = jax.vmap(feats)(Y_fit)                       # (K,P)
                P = F.shape[1]
                G = F.T @ F + 1e-3 * jnp.eye(P)
                coef = jnp.linalg.solve(G, F.T @ Jn)
                resid = F @ coef - Jn
                r2 = 1.0 - float(jnp.mean(resid ** 2))
                if not np.isfinite(r2) or r2 < 0.0:
                    # degenerate buffer (collapsed cloud): refit heavily ridged;
                    # if still bad, drop the force for this stage (pure precond. RW)
                    coef = jnp.linalg.solve(F.T @ F + 1.0 * jnp.eye(P), F.T @ Jn)
                    resid = F @ coef - Jn
                    r2 = 1.0 - float(jnp.mean(resid ** 2))
                    if not np.isfinite(r2) or r2 < 0.0:
                        coef = jnp.zeros_like(coef)
                b_lin = coef[1:1 + m]
                q_coef = coef[1 + m:]

                def grad_y(y):
                    gq = jnp.zeros((m,))
                    contrib = q_coef * sym
                    gq = gq.at[iu].add(contrib * y[ju])
                    gq = gq.at[ju].add(contrib * y[iu] * (iu != ju))
                    return b_lin + gq
            else:
                import optax

                def mlp(pars, y):
                    hcur = y
                    for (Wm, bm) in pars[:-1]:
                        hcur = jnp.tanh(hcur @ Wm + bm)
                    Wm, bm = pars[-1]
                    return (hcur @ Wm + bm)[0]

                if mlp_params is None:
                    key, mk = jax.random.split(key)
                    ks = jax.random.split(mk, 3)
                    sizes = [(m, 64), (64, 64), (64, 1)]
                    mlp_params = [
                        (0.5 * jax.random.normal(kk, sz) / np.sqrt(sz[0]),
                         jnp.zeros((sz[1],))) for kk, sz in zip(ks, sizes)]

                def loss(pars):
                    pred = jax.vmap(lambda y: mlp(pars, y))(Y_fit)
                    return jnp.mean((pred - Jn) ** 2)

                opt = optax.adam(3e-3)
                ost = opt.init(mlp_params)

                @jax.jit
                def tr_step(pars, ost_):
                    lval, g = jax.value_and_grad(loss)(pars)
                    upd, ost_ = opt.update(g, ost_)
                    return optax.apply_updates(pars, upd), ost_, lval

                lval = None
                for _ in range(500):
                    mlp_params, ost, lval = tr_step(mlp_params, ost)
                r2 = 1.0 - float(lval)
                pars_frozen = jax.tree_util.tree_map(lambda x: x, mlp_params)

                def grad_y(y):
                    return jax.grad(lambda yy: mlp(pars_frozen, yy))(y)

            # ---- RAMP-MALA in STANDARDIZED coordinates (the draft's construction):
            # z=(u-center)/scale frozen per stage; equivalent to population-
            # preconditioned MALA in u-space. No 1/scale term in the force, so the
            # drift stays bounded when the cloud contracts.
            temp32 = jnp.asarray(t_next, dtype=jnp.float32)
            Js32 = jnp.asarray(float(Js), dtype=jnp.float32)
            eps32 = jnp.asarray(float(eps_cur), dtype=jnp.float32)

            def ghat_z(z):
                u = center + scale * z
                gp = scale * grad_log_prior(u)
                gJ = Js32 * lift(grad_y(project(z)))
                return gp - temp32 * beta * gJ

            def propose_z(rng, z):
                mean_f = z + 0.5 * eps32 ** 2 * ghat_z(z)
                z_new = mean_f + eps32 * jax.random.normal(rng, (d,), dtype=jnp.float32)
                return z_new, mean_f

            propose_all = jax.jit(jax.vmap(propose_z))
            ghat_all = jax.jit(jax.vmap(ghat_z))
            logp0_all = jax.jit(jax.vmap(log_prior_fn))

            accs = []
            Z = (particles - center[None, :]) / scale[None, :]
            for _ in range(int(num_mcmc_steps)):
                key, pk, ak = jax.random.split(key, 3)
                z_prop, mean_f = propose_all(jax.random.split(pk, n), Z)
                u_prop = center[None, :] + scale[None, :] * z_prop
                J_prop = -loglik_all(u_prop) / beta
                mean_rev = z_prop + 0.5 * eps32 ** 2 * ghat_all(z_prop)
                log_q_fwd = -jnp.sum((z_prop - mean_f) ** 2, axis=1) / (2 * eps32 ** 2)
                log_q_rev = -jnp.sum((Z - mean_rev) ** 2, axis=1) / (2 * eps32 ** 2)
                log_tgt = (logp0_all(u_prop) - temp32 * beta * J_prop) -                           (logp0_all(particles) - temp32 * beta * J_cur)
                log_a = log_tgt + log_q_rev - log_q_fwd
                ok = jnp.log(jax.random.uniform(ak, (n,), minval=1e-12)) < log_a
                particles = jnp.where(ok[:, None], u_prop, particles)
                Z = jnp.where(ok[:, None], z_prop, Z)
                J_cur = jnp.where(ok, J_prop, J_cur)
                accs.append(float(jnp.mean(ok)))
                buf_U.append(np.array(u_prop))   # kept even if rejected (Alg. 1)
                buf_J.append(np.array(J_prop))

            acc_mean = float(np.mean(accs)) if accs else 0.0
            eps_used = float(eps_cur)
            eps_cur = float(np.clip(eps_cur * np.exp(1.0 * (acc_mean - 0.55)), 1e-4, 0.5))

            temp = float(t_next)
            iteration += 1
            temperatures.append(temp)
            ess_values.append(ess)
            if record_energies:
                energies_history.append(np.array(J_cur))
            print(f"  [SMC-{kname}] Iter {iteration:3d}: λ={temp:.4f}, ESS={ess:.3f}, "
                  f"acc={float(np.mean(accs)):.3f}, eps={eps_used:.2e}, fitR2={r2:.3f}, "
                  f"costs=[{float(jnp.min(J_cur)):.1f}, {float(jnp.median(J_cur)):.1f}, "
                  f"{float(jnp.max(J_cur)):.1f}]")

        print(f"  [SMC-{kname}] Completed in {iteration} iterations. Final λ = {temp:.4f}")
        info = {"temperatures": temperatures, "ess_values": ess_values,
                "num_iterations": iteration}
        if record_energies:
            info["energy_values"] = np.asarray(energies_history, dtype=np.float32)
        return particles, info

    def _run_smc_loop(self, key, smc, initial_particles, log_likelihood_fn,
                      num_particles, max_iterations, kernel_name, record_energies):
        """Run the SMC loop with diagnostics."""
        state = smc.init(initial_particles)
        
        temperatures = [0.0]
        ess_values = []
        energies_history = []
        iteration = 0
        
        @jax.jit
        def compute_log_likelihoods(particles):
            return jax.vmap(log_likelihood_fn)(particles)
        
        print(f"  [SMC-{kernel_name}] Starting adaptive tempered SMC...")
        
        # Print initial cost range
        init_log_liks = compute_log_likelihoods(initial_particles)
        init_costs = -init_log_liks / self.beta  # Undo beta scaling
        print(f"  [SMC-{kernel_name}] Initial particle costs: min={float(jnp.min(init_costs)):.1f}, "
              f"max={float(jnp.max(init_costs)):.1f}, median={float(jnp.median(init_costs)):.1f}")
        
        if record_energies:
            energies_history.append(np.array(init_costs))
        
        prev_lambda = 0.0
        
        while state.tempering_param < 1.0 and iteration < max_iterations:
            particles_before = state.particles
            log_liks_before = compute_log_likelihoods(particles_before)
            
            key, step_key = jax.random.split(key)
            state, info = smc.step(step_key, state)
            
            current_lambda = float(state.tempering_param)
            delta_lambda = current_lambda - prev_lambda
            
            # ESS after reweighting
            incremental_log_weights = delta_lambda * log_liks_before
            ess_after_reweight = compute_ess_from_log_weights(incremental_log_weights)
            ess_ratio = float(ess_after_reweight) / num_particles
            
            # Cost range
            log_liks_after = compute_log_likelihoods(state.particles)
            costs_after = -log_liks_after / self.beta
            cost_min = float(jnp.min(costs_after))
            cost_max = float(jnp.max(costs_after))
            cost_median = float(jnp.median(costs_after))
            
            # MCMC diagnostics
            mcmc_info = info.update_info
            mean_acc = float(jnp.mean(mcmc_info.acceptance_rate))
            # Divergences exist for HMC/NUTS but not for MALA.
            if hasattr(mcmc_info, "is_divergent"):
                num_div = int(jnp.sum(mcmc_info.is_divergent))
            else:
                num_div = 0
            
            # NUTS also reports trajectory length
            if hasattr(mcmc_info, "num_integration_steps"):
                mean_L = float(jnp.mean(mcmc_info.num_integration_steps))
                diag_str = f"acc={mean_acc:.3f}, div={num_div}, L={mean_L:.1f}"
            else:
                diag_str = f"acc={mean_acc:.3f}, div={num_div}"
            
            temperatures.append(current_lambda)
            ess_values.append(ess_ratio)
            iteration += 1
            
            if record_energies:
                energies_history.append(np.array(costs_after))
            
            print(f"  [SMC-{kernel_name}] Iter {iteration:3d}: λ={current_lambda:.4f} (Δλ={delta_lambda:.6f}), "
                  f"ESS={ess_ratio:.3f}, {diag_str}, costs=[{cost_min:.1f}, {cost_median:.1f}, {cost_max:.1f}]")
            
            prev_lambda = current_lambda
        
        print(f"  [SMC-{kernel_name}] Completed in {iteration} iterations. Final λ = {float(state.tempering_param):.4f}")
        
        info = {
            "temperatures": temperatures,
            "ess_values": ess_values,
            "num_iterations": iteration,
        }
        if record_energies:
            info["energy_values"] = np.asarray(energies_history, dtype=np.float32)
        
        return state.particles, info

