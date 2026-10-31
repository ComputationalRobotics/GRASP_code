"""AR(1) (autoregressive) priors over control sequences."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


class AR1Prior:
    """AR(1) prior over control sequences.
    
    Models temporal correlation around a configurable stationary mean:
        u[0] ~ N(mean, sigma^2 / (1 - rho^2))
        u[t] - mean = rho * (u[t-1] - mean) + sigma * eps[t]
    
    Args:
        rho: Autocorrelation coefficient (0 < rho < 1)
        sigma: Innovation standard deviation
    """
    
    def __init__(self, rho: float = 0.9, sigma: float = 2.0, mean: float = 0.0):
        self.rho = float(rho)
        self.sigma = float(sigma)
        self.mean = float(mean)
    
    def energy(self, U_raw: jnp.ndarray) -> float:
        """Compute negative log-prior energy (up to constant).
        
        Args:
            U_raw: Control sequence (H,) or (H, control_dim)
            
        Returns:
            Energy (higher = less probable)
        """
        rho, sigma = self.rho, self.sigma
        centered = U_raw - self.mean
        
        # Handle 1D or 2D control sequences
        if centered.ndim == 1:
            H = centered.shape[0]
            u0 = centered[0]
            E = (1.0 - rho**2) * (u0 * u0)
            if H > 1:
                du = centered[1:] - rho * centered[:-1]
                E = E + jnp.sum(du * du)
        else:
            # Multi-dimensional control
            H = centered.shape[0]
            u0 = centered[0]
            E = (1.0 - rho**2) * jnp.sum(u0 * u0)
            if H > 1:
                du = centered[1:] - rho * centered[:-1]
                E = E + jnp.sum(du * du)
        
        return 0.5 * E / (sigma**2)
    
    def log_prob(self, U_raw: jnp.ndarray) -> float:
        """Compute log-probability (up to constant).
        
        Args:
            U_raw: Control sequence (H,) or (H, control_dim)
            
        Returns:
            Log probability
        """
        return -self.energy(U_raw)
    
    def sample(self, key: jax.Array, H: int, control_dim: int = 1) -> jnp.ndarray:
        """Sample a control sequence from the prior.
        
        Args:
            key: JAX random key
            H: Horizon (number of timesteps)
            control_dim: Dimension of control (default 1)
            
        Returns:
            Control sequence (H,) if control_dim=1, else (H, control_dim)
        """
        rho, sigma = self.rho, self.sigma
        
        if control_dim == 1:
            return self._sample_1d(key, H)
        else:
            return self._sample_nd(key, H, control_dim)
    
    def _sample_1d(self, key: jax.Array, H: int) -> jnp.ndarray:
        """Sample 1D control sequence."""
        rho, sigma = self.rho, self.sigma
        key0, keyeps = jax.random.split(key)
        
        std0 = sigma / jnp.sqrt(jnp.maximum(1e-12, 1.0 - rho**2))
        u0_centered = std0 * jax.random.normal(key0, ())
        
        if H == 1:
            return jnp.array([u0_centered + self.mean], dtype=jnp.float32)
        
        eps = jax.random.normal(keyeps, shape=(H-1,), dtype=jnp.float32)
        
        def rec(prev, e):
            u = rho * prev + sigma * e
            return u, u
        
        _, centered_us = lax.scan(rec, u0_centered, eps)
        centered = jnp.concatenate([jnp.array([u0_centered], dtype=jnp.float32), centered_us])
        return centered + self.mean
    
    def _sample_nd(self, key: jax.Array, H: int, control_dim: int) -> jnp.ndarray:
        """Sample N-dimensional control sequence."""
        rho, sigma = self.rho, self.sigma
        key0, keyeps = jax.random.split(key)
        
        std0 = sigma / jnp.sqrt(jnp.maximum(1e-12, 1.0 - rho**2))
        u0_centered = std0 * jax.random.normal(key0, (control_dim,))
        
        if H == 1:
            return u0_centered[None, :] + self.mean
        
        eps = jax.random.normal(keyeps, shape=(H-1, control_dim), dtype=jnp.float32)
        
        def rec(prev, e):
            u = rho * prev + sigma * e
            return u, u
        
        _, centered_us = lax.scan(rec, u0_centered, eps)
        centered = jnp.concatenate([u0_centered[None, :], centered_us], axis=0)
        return centered + self.mean


class WindowAR1Prior:
    """AR(1) prior over a window of delta-knots, optionally conditioned on the
    last committed knot ``u_prev``.  Zero mean (deltas around the reference).
    """

    def __init__(
        self,
        control_dim: int,
        horizon: int,
        rho: float,
        sigma: float,
        u_prev: np.ndarray | None = None,
        channel_scale=None,
        bias: np.ndarray | None = None,
        bias_prev: np.ndarray | None = None,
    ):
        self.control_dim = int(control_dim)
        self.horizon = int(horizon)
        self.rho = float(rho)
        self.sigma = float(sigma)
        # Innovation std = sigma * channel_scale; channel_scale is one number per channel or a full
        # (horizon, control_dim) field.
        if channel_scale is None:
            scale = np.ones((self.horizon, self.control_dim), np.float32)
        else:
            scale = np.asarray(channel_scale, np.float32)
            if scale.ndim == 1:
                scale = np.broadcast_to(scale.reshape(1, -1), (self.horizon, self.control_dim))
        scale = np.ascontiguousarray(scale, np.float32)
        if scale.shape != (self.horizon, self.control_dim) or not np.all(scale > 0):
            raise ValueError("channel_scale must be positive with shape (control_dim,) or "
                             f"(horizon, control_dim); got {scale.shape}")
        self.channel_scale = jnp.asarray(scale)
        self.sigma_vec = self.sigma * self.channel_scale        # (horizon, control_dim)
        self.mean = 0.0
        self.reference = jnp.zeros((self.horizon, self.control_dim), jnp.float32)
        self.u_prev = (
            None
            if u_prev is None
            else jnp.asarray(np.asarray(u_prev, dtype=np.float32))
        )
        # The AR(1) is centred on ``bias`` rather than on zero.  bias is a (control_dim,) offset or a
        # (horizon, control_dim) path; warm start centres it on the previous window's unexecuted plan (shifted
        # by the commit), so the window refines the last solution instead of re-deriving it around the
        # reference.  bias_prev is the centre the committed knot u_prev is measured against (warm start: the
        # committed knot itself, deviation 0); by default it is the first row of the bias.
        if bias is None:
            self.bias = None
        else:
            b = np.asarray(bias, dtype=np.float32)
            if b.ndim == 1:
                b = np.broadcast_to(b.reshape(1, -1), (self.horizon, self.control_dim))
            if b.shape != (self.horizon, self.control_dim):
                raise ValueError(f"bias must have shape (control_dim,) or (horizon, control_dim); got {b.shape}")
            self.bias = jnp.asarray(np.ascontiguousarray(b))
        self.bias_prev = (
            None if bias_prev is None else jnp.asarray(np.asarray(bias_prev, dtype=np.float32).reshape(-1))
        )

    def _up(self):
        """u_prev as a deviation from the AR(1) centre (None when there is no committed knot)."""
        if self.u_prev is None:
            return None
        if self.bias is None:
            return self.u_prev
        return self.u_prev - (self.bias_prev if self.bias_prev is not None else self.bias[0])

    def energy(self, actions):
        U = jnp.asarray(actions, dtype=jnp.float32).reshape(
            (self.horizon, self.control_dim)
        )
        sv0, sv_rest = self.sigma_vec[0], self.sigma_vec[1:]
        if self.bias is not None:
            U = U - self.bias
        up = self._up()
        if up is None:
            first = (1.0 - self.rho**2) * jnp.sum((U[0] / sv0) ** 2)
        else:
            first = jnp.sum(((U[0] - self.rho * up) / sv0) ** 2)
        rest = jnp.sum(((U[1:] - self.rho * U[:-1]) / sv_rest) ** 2)
        return 0.5 * (first + rest)

    def log_prob(self, actions):
        return -self.energy(actions)

    def conditional_mean(self) -> np.ndarray:
        """Mean path: zeros, or the geometric decay of ``u_prev``."""
        bias = None if self.bias is None else np.asarray(self.bias, dtype=np.float64)
        if self.u_prev is None:
            base = np.zeros((self.horizon, self.control_dim), dtype=np.float64)
            return (base if bias is None else base + bias).astype(np.float32)
        up = np.asarray(self._up(), dtype=np.float64)
        powers = self.rho ** np.arange(1, self.horizon + 1, dtype=np.float64)
        out = powers[:, None] * up[None, :]
        return (out if bias is None else out + bias).astype(np.float32)

    def sample(self, key, horizon: int, control_dim: int = 1):
        if (int(horizon), int(control_dim)) != (self.horizon, self.control_dim):
            raise ValueError("sample shape does not match the window prior")
        key_first, key_rest = jax.random.split(key)
        noise_first = jax.random.normal(key_first, (self.control_dim,), jnp.float32)
        up = self._up()
        if up is None:
            first = self.sigma_vec[0] / jnp.sqrt(1.0 - self.rho**2) * noise_first
        else:
            first = self.rho * up + self.sigma_vec[0] * noise_first
        if self.horizon == 1:
            out = first[None, :]
            return out if self.bias is None else out + self.bias
        innovations = self.sigma_vec[1:] * jax.random.normal(
            key_rest, (self.horizon - 1, self.control_dim), jnp.float32
        )

        def transition(previous, innovation):
            current = self.rho * previous + innovation
            return current, current

        _, rest = lax.scan(transition, first, innovations)
        out = jnp.concatenate([first[None, :], rest], axis=0)
        return out if self.bias is None else out + self.bias
