# systems/base.py
"""Base class for dynamical systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import jax.numpy as jnp
from jax import lax
import numpy as np


@dataclass
class SystemParams:
    """Base class for system parameters."""
    pass


@dataclass
class CostWeights:
    """Base class for cost weights."""
    pass


class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems.
    
    A dynamical system defines:
    - State space and control space
    - Dynamics (how state evolves given control)
    - Cost function (what we're optimizing)
    - Visualization methods
    """
    
    def __init__(self, params: SystemParams, cost_weights: CostWeights, dt: float = 0.02):
        self.params = params
        self.cost_weights = cost_weights
        self.dt = dt
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the system."""
        pass
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state space."""
        pass
    
    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Dimension of control space."""
        pass
    
    @property
    @abstractmethod
    def default_initial_state(self) -> jnp.ndarray:
        """Default initial state."""
        pass
    
    @property
    @abstractmethod
    def u_max(self) -> float:
        """Maximum control magnitude."""
        pass
    
    @abstractmethod
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Compute state derivatives: dx/dt = f(x, u)."""
        pass
    
    @abstractmethod
    def step(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Single integration step."""
        pass
    
    @abstractmethod
    def stage_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """Running cost at a single timestep."""
        pass
    
    @abstractmethod
    def terminal_cost(self, x: jnp.ndarray) -> float:
        """Terminal cost at final state."""
        pass
    
    def clamp_control(self, u_raw: jnp.ndarray) -> jnp.ndarray:
        """Clamp control to valid range (smooth tanh by default)."""
        return self.u_max * jnp.tanh(u_raw / self.u_max)
    
    def rollout(self, x0: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        """Roll out trajectory given initial state and control sequence.
        
        Args:
            x0: Initial state (state_dim,)
            U: Control sequence (H, control_dim) or (H,) for 1D control
            
        Returns:
            X: State trajectory (H+1, state_dim)
        """
        def step_fn(x, u):
            x_next = self.step(x, u)
            return x_next, x_next
        _, Xs = lax.scan(step_fn, x0, U)
        return jnp.vstack([x0[None, :], Xs])
    
    def trajectory_cost(self, x0: jnp.ndarray, U: jnp.ndarray) -> float:
        """Compute total trajectory cost.
        
        Args:
            x0: Initial state
            U: Control sequence (already clamped/valid)
            
        Returns:
            Total cost (running + terminal)
        """
        def step_fn(x, u):
            c = self.stage_cost(x, u)
            x_next = self.step(x, u)
            return x_next, c
        
        xT, costs = lax.scan(step_fn, x0, U)
        return jnp.sum(costs) + self.terminal_cost(xT)
    
    def trajectory_cost_from_raw(self, x0: jnp.ndarray, U_raw: jnp.ndarray) -> float:
        """Compute trajectory cost from raw (unclamped) controls.
        
        This is what samplers use - they optimize in unconstrained space,
        and we clamp before evaluating cost.
        """
        U = self.clamp_control(U_raw)
        return self.trajectory_cost(x0, U)
    
    # -------------------------------------------------------------------------
    # Visualization (override in subclasses for custom visualizations)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def save_animation(self, filename: str, X: np.ndarray, U: np.ndarray, 
                       title: str = "", dpi: int = 150, fps: Optional[int] = None):
        """Save animation of trajectory."""
        pass
    
    def plot_trajectory(self, ax, X: np.ndarray, U: np.ndarray, label: str = "", color: str = "C0"):
        """Plot trajectory on given axes (for comparison plots)."""
        # Default implementation - override for system-specific plots
        t = np.arange(len(X)) * self.dt
        ax.plot(t, X[:, 0], color=color, label=label)

