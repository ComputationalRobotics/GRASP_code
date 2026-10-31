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
            record_energies: bool = False) -> SamplingResult:
        """Run adaptive tempered SMC.
        
        Args:
            key: JAX random key
            H: Horizon
            num_particles: Number of particles
            kernel: "nuts", "hmc" or "mala"
            target_ess: Target ESS ratio (0-1)
            num_mcmc_steps: MCMC steps per SMC iteration
            step_size: NUTS/HMC/MALA step size
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
        else:
            raise ValueError(f"Unknown kernel: {kernel}. Use 'nuts', 'hmc' or 'mala'.")
        
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

