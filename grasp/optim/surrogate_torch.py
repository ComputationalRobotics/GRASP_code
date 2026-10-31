"""GPU (torch) port of ``fit_quadratic_surrogate``.

Same model, same robust filtering, same locality weights, same ridge and the
same returned quantities as ``grasp.optim.surrogate.
fit_quadratic_surrogate``; only the weighted least squares is executed on a
torch device in float64, with the Gram matrix accumulated over row chunks so
the full design matrix is never materialised.
"""

from __future__ import annotations

import numpy as np


def fit_quadratic_surrogate_torch(
    positions: np.ndarray,
    costs: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    ridge: float = 1e-3,
    window: float = 3.0,
    basis: np.ndarray | None = None,
    return_constant: bool = False,
    device: str = "cuda",
    chunk_rows: int = 8192,
):
    import torch

    dev = torch.device(device)
    dtype = torch.float64
    positions = torch.as_tensor(np.asarray(positions, dtype=np.float64), device=dev)
    costs = torch.as_tensor(np.asarray(costs, dtype=np.float64), device=dev)
    center = torch.as_tensor(np.asarray(center, dtype=np.float64), device=dev)
    scale = torch.clamp(
        torch.as_tensor(np.asarray(scale, dtype=np.float64), device=dev), min=1e-8
    )

    z = (positions - center[None, :]) / scale[None, :]
    normalized_distance_sq = torch.sum(z * z, dim=1) / z.shape[1]

    keep = torch.isfinite(costs) & torch.all(torch.isfinite(z), dim=1)
    trusted = keep & (normalized_distance_sq <= (3.0 * float(window)) ** 2)
    if bool(torch.any(trusted)):
        keep = trusted
    z = z[keep]
    costs = costs[keep]

    quantiles = torch.quantile(costs, torch.tensor([0.5, 0.9], dtype=dtype, device=dev))
    median_cost, upper_cost = float(quantiles[0]), float(quantiles[1])
    cost_cap = median_cost + 10.0 * max(upper_cost - median_cost, 1e-9)
    inliers = costs <= cost_cap
    if int(torch.sum(inliers)) >= max(8, z.shape[1]):
        z = z[inliers]
        costs = costs[inliers]
    count = int(z.shape[0])

    weights = torch.exp(-torch.sum(z * z, dim=1) / (2.0 * float(window) ** 2 * z.shape[1]))
    if basis is not None:
        z = z @ torch.as_tensor(np.asarray(basis, dtype=np.float64), device=dev)
    dimension = int(z.shape[1])

    upper_i, upper_j = np.triu_indices(dimension)
    upper_i_t = torch.as_tensor(upper_i, device=dev)
    upper_j_t = torch.as_tensor(upper_j, device=dev)
    feature_count = 1 + dimension + upper_i.size
    sqrt_weights = torch.sqrt(weights)

    def features_of(z_block: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                torch.ones((z_block.shape[0], 1), dtype=dtype, device=dev),
                z_block,
                z_block[:, upper_i_t] * z_block[:, upper_j_t],
            ],
            dim=1,
        )

    gram = torch.zeros((feature_count, feature_count), dtype=dtype, device=dev)
    rhs = torch.zeros((feature_count,), dtype=dtype, device=dev)
    step = max(int(chunk_rows), 1)
    for start in range(0, count, step):
        stop = min(start + step, count)
        block = features_of(z[start:stop])
        design = sqrt_weights[start:stop, None] * block
        target = sqrt_weights[start:stop] * costs[start:stop]
        gram += design.T @ design
        rhs += design.T @ target
    # Match the CPU fit: the arbitrary energy offset must not affect guidance.
    gram.diagonal()[1:].add_(float(ridge) * max(1, count))
    if bool(torch.any(weights > 0)):
        theta = torch.linalg.solve(gram, rhs)
    else:
        theta = torch.zeros_like(rhs)
        theta[0] = torch.quantile(costs, 0.5)

    # R^2 with the same weights, again in chunks.
    weighted_mean = torch.sum(weights * costs) / torch.clamp(torch.sum(weights), min=1e-12)
    total = torch.sum(weights * (costs - weighted_mean) ** 2)
    residual_sum = torch.zeros((), dtype=dtype, device=dev)
    for start in range(0, count, step):
        stop = min(start + step, count)
        predictions = features_of(z[start:stop]) @ theta
        residual_sum += torch.sum(weights[start:stop] * (costs[start:stop] - predictions) ** 2)
    explained = 1.0 - float(residual_sum) / max(float(total), 1e-12)

    theta_np = theta.detach().cpu().numpy()
    linear = theta_np[1 : 1 + dimension]
    quad = np.zeros((dimension, dimension))
    quad[upper_i, upper_j] = theta_np[1 + dimension :]
    quad = quad + quad.T
    if return_constant:
        return float(theta_np[0]), linear, quad, float(explained)
    return linear, quad, float(explained)
