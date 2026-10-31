# algorithms/base.py
"""Base class for sampling algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.systems.base import DynamicalSystem
from grasp.optim.prior_ar1 import AR1Prior


@dataclass
class SamplingResult:
    """Result from a sampling algorithm.
    
    Attributes:
        U: Best control sequence found (H,) or (H, control_dim)
        X: Best state trajectory (H+1, state_dim)
        cost: Cost of best trajectory
        samples: All samples/particles (N, H) or (N, H, control_dim)
        info: Additional algorithm-specific information
    """
    U: np.ndarray
    X: np.ndarray
    cost: float
    samples: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)


class SamplingAlgorithm(ABC):
    """Abstract base class for sampling algorithms.
    
    A sampling algorithm takes a dynamical system and prior, and samples
    from the posterior over control sequences.
    """
    
    def __init__(self, system: DynamicalSystem, prior: AR1Prior, beta: float = 1.0):
        """Initialize the sampling algorithm.
        
        Args:
            system: The dynamical system
            prior: Prior over control sequences
            beta: Inverse temperature (higher = more peaked on low-cost solutions)
        """
        self.system = system
        self.prior = prior
        self.beta = beta
        self._x0 = system.default_initial_state
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm."""
        pass
    
    def set_initial_state(self, x0: jnp.ndarray):
        """Set the initial state for trajectory optimization."""
        self._x0 = jnp.asarray(x0, dtype=jnp.float32)
    
    @abstractmethod
    def run(self, key: jax.Array, H: int, **kwargs) -> SamplingResult:
        """Run the sampling algorithm.
        
        Args:
            key: JAX random key
            H: Horizon (number of control steps)
            **kwargs: Algorithm-specific parameters
            
        Returns:
            SamplingResult with best trajectory and samples
        """
        pass
    
    # -------------------------------------------------------------------------
    # Helper methods for subclasses
    # -------------------------------------------------------------------------
    
    def make_log_posterior(self) -> Callable:
        """Create the log-posterior function for sampling.
        
        Returns a function: U_raw -> log p(U_raw | x0, beta)
        """
        x0 = self._x0
        beta = self.beta
        system = self.system
        prior = self.prior

        def _reshape_if_needed(U_raw):
            """Ensure multi-DoF controls are shaped (H,Du) when presented as a flat vector.

            BlackJAX often operates on flat vectors; for control_dim>1 we accept either:
            - (H, Du) array
            - (H*Du,) flat vector (time-major) that we reshape to (H,Du)
            """
            Du = int(getattr(system, "control_dim", 1))
            if Du <= 1:
                return U_raw
            if getattr(U_raw, "ndim", 0) == 1:
                n = int(U_raw.shape[0])
                if n % Du != 0:
                    raise ValueError(f"Flat control length {n} not divisible by control_dim={Du}")
                H = n // Du
                return jnp.reshape(U_raw, (H, Du))
            return U_raw
        
        def logprob(U_raw):
            U_raw = _reshape_if_needed(U_raw)
            # Prior energy
            E_prior = prior.energy(U_raw)
            # Trajectory cost (clamps internally)
            cost = system.trajectory_cost_from_raw(x0, U_raw)
            # Log posterior = -E_prior - beta * cost
            return -E_prior - beta * cost
        
        return logprob
    
    def make_log_prior_fn(self) -> Callable:
        """Create the log-prior function for SMC."""
        prior = self.prior
        
        def logprior(U_raw):
            Du = int(getattr(self.system, "control_dim", 1))
            if Du > 1 and U_raw.ndim == 1:
                n = int(U_raw.shape[0])
                if n % Du != 0:
                    raise ValueError(f"Flat control length {n} not divisible by control_dim={Du}")
                U_raw = jnp.reshape(U_raw, (n // Du, Du))
            return prior.log_prob(U_raw)
        
        return logprior
    
    def make_log_likelihood_fn(self) -> Callable:
        """Create the log-likelihood function for SMC.
        
        Note: This returns -beta * cost, so the full posterior is:
        log p(U | x0) = log_prior(U) + log_likelihood(U)
        """
        x0 = self._x0
        beta = self.beta
        system = self.system
        
        def loglikelihood(U_raw):
            Du = int(getattr(system, "control_dim", 1))
            if Du > 1 and U_raw.ndim == 1:
                n = int(U_raw.shape[0])
                if n % Du != 0:
                    raise ValueError(f"Flat control length {n} not divisible by control_dim={Du}")
                U_raw = jnp.reshape(U_raw, (n // Du, Du))
            cost = system.trajectory_cost_from_raw(x0, U_raw)
            return -beta * cost
        
        return loglikelihood
    
    def make_cost_evaluator(self) -> Callable:
        """Create a batched cost evaluator."""
        x0 = self._x0
        system = self.system
        
        @jax.jit
        def cost_batch(U_raw_batch):
            def cost_one(U_raw):
                Du = int(getattr(system, "control_dim", 1))
                if Du > 1 and U_raw.ndim == 1:
                    n = int(U_raw.shape[0])
                    U_raw = jnp.reshape(U_raw, (n // Du, Du))
                return system.trajectory_cost_from_raw(x0, U_raw)
            return jax.vmap(cost_one)(U_raw_batch)
        
        return cost_batch
    
    def pick_best(self, samples: jnp.ndarray) -> tuple:
        """Pick the best sample from a batch.
        
        Args:
            samples: Batch of samples (N, H) or (N, H, control_dim)
            
        Returns:
            (best_U_raw, best_cost)
        """
        cost_fn = self.make_cost_evaluator()
        costs = cost_fn(samples)
        i = jnp.argmin(costs)
        return samples[i], float(costs[i])
    
    def rollout_best(self, U_raw: jnp.ndarray) -> tuple:
        """Rollout the best trajectory.
        
        Args:
            U_raw: Raw control sequence
            
        Returns:
            (U_clamped, X_trajectory)
        """
        Du = int(getattr(self.system, "control_dim", 1))
        if Du > 1 and U_raw.ndim == 1:
            n = int(U_raw.shape[0])
            U_raw = jnp.reshape(U_raw, (n // Du, Du))
        U = self.system.clamp_control(U_raw)
        X = self.system.rollout(self._x0, U)
        return np.array(U), np.array(X)

