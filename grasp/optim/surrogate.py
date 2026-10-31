"""Quadratic surrogate of the rollout cost and the temporal DCT basis it is fitted in."""

from __future__ import annotations

import numpy as np


def make_dct_basis(
    horizon: int, control_dim: int, rank: int
) -> np.ndarray:
    """Return an orthonormal temporal DCT-II basis of shape ``(H * m, r * m)``.

    Columns span the ``rank`` lowest temporal frequencies applied
    independently to each control channel; the flat layout matches
    ``reshape(horizon, control_dim)`` row-major ordering.
    """
    times = np.arange(int(horizon), dtype=np.float64)
    columns = []
    for frequency in range(int(rank)):
        column = np.cos(np.pi * (times + 0.5) * frequency / float(horizon))
        column *= np.sqrt(
            (1.0 if frequency == 0 else 2.0) / float(horizon)
        )
        columns.append(column)
    temporal = np.stack(columns, axis=1)
    if int(control_dim) == 1:
        return temporal
    return np.kron(temporal, np.eye(int(control_dim)))


def fit_quadratic_surrogate(
    positions: np.ndarray,
    costs: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    ridge: float = 1e-3,
    window: float = 3.0,
    basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a locally weighted ridge quadratic to stored rollout evaluations.

    The model in standardized coordinates ``z = (u - center) / scale`` is
    ``J(z) = c + b^T y + 0.5 y^T A y`` with ``y = basis^T z`` (``y = z`` when
    ``basis`` is ``None``) and symmetric ``A``, so the raw-space gradient is
    ``basis (b + A y) / scale``. Points far from the current particle
    population (measured by ``window`` in standardized units, normalized by
    dimension) receive exponentially small weight.

    The intercept is unpenalized, so adding a constant to all costs does not
    change the fitted gradient or curvature.

    Returns:
        ``(b, A, weighted_r_squared)`` in float64, with ``b`` and ``A`` in
        basis coordinates.
    """
    positions = np.asarray(positions, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-8)

    z = (positions - center[None, :]) / scale[None, :]
    normalized_distance_sq = np.sum(z * z, axis=1) / z.shape[1]

    # Robustness: exclude non-finite evaluations and points far outside the
    # trust region around the current population, then winsorize the retained
    # costs so a handful of extreme rollouts cannot dominate the weighted
    # least squares (the "surrogate poisoning" failure mode).
    keep = np.isfinite(costs) & np.all(np.isfinite(z), axis=1)
    trusted = keep & (
        normalized_distance_sq <= (3.0 * float(window)) ** 2
    )
    if np.any(trusted):
        keep = trusted
    z = z[keep]
    costs = costs[keep]

    median_cost, upper_cost = np.quantile(costs, [0.5, 0.9])
    cost_cap = median_cost + 10.0 * max(upper_cost - median_cost, 1e-9)
    inliers = costs <= cost_cap
    if np.sum(inliers) >= max(8, z.shape[1]):
        z = z[inliers]
        costs = costs[inliers]
    count = z.shape[0]

    weights = np.exp(
        -np.sum(z * z, axis=1) / (2.0 * float(window) ** 2 * z.shape[1])
    )
    if basis is not None:
        z = z @ np.asarray(basis, dtype=np.float64)
    dimension = z.shape[1]

    upper_i, upper_j = np.triu_indices(dimension)
    features = np.concatenate(
        [np.ones((count, 1)), z, z[:, upper_i] * z[:, upper_j]], axis=1
    )
    feature_count = features.shape[1]

    sqrt_weights = np.sqrt(weights)[:, None]
    design = sqrt_weights * features
    target = sqrt_weights[:, 0] * costs
    gram = design.T @ design
    # The energy zero is arbitrary: regularize the field, not the intercept.
    # Penalizing theta[0] lets constant costs leak into linear/quadratic terms.
    regularized = np.arange(1, gram.shape[0])
    gram[regularized, regularized] += float(ridge) * max(1, count)
    if np.any(weights > 0):
        theta = np.linalg.solve(gram, design.T @ target)
    else:
        # No effective local observations: retain a constant, not a field.
        theta = np.zeros(feature_count, dtype=np.float64)
        theta[0] = float(np.median(costs))

    predictions = features @ theta
    residual = costs - predictions
    weighted_mean = np.sum(weights * costs) / max(np.sum(weights), 1e-12)
    total = np.sum(weights * (costs - weighted_mean) ** 2)
    explained = 1.0 - np.sum(weights * residual**2) / max(total, 1e-12)

    linear = theta[1 : 1 + dimension]
    quad = np.zeros((dimension, dimension))
    quad[upper_i, upper_j] = theta[1 + dimension :]
    quad = quad + quad.T  # doubles diagonal: d/dz_k of q_kk z_k^2 is 2 q_kk z_k
    return linear, quad, float(explained)
