"""Full-horizon tracking cost of complete control sequences.

Used for the reference cost J0 before optimization and for re-evaluating the executed control trajectory from the
initial state after the MPC run.  The cost of a control sequence is minus its mean per-step tracking reward
(the last step uses the terminal reward), the same quantity the MPC windows add up.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

RolloutFn = Callable[
    [Any, Any, torch.Tensor, tuple[torch.Tensor, ...], dict[str, Any]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]],
]


class FullHorizonCost:
    """Evaluate ``(num_samples, horizon_steps, nu)`` control batches with one rollout from the current state.

    Args:
        config: Processed configuration; ``num_samples`` is the number of simulator worlds.
        env: Batched MuJoCo Warp environment.
        rollout: Callable returned by ``grasp.sim.rollout.make_rollout_fn``.
        ref_slice: Reference rows used by the per-step rewards.
        env_params: Simulator parameters loaded for the rollout.
    """

    def __init__(self, config: Any, env: Any, rollout: RolloutFn, ref_slice: tuple[torch.Tensor, ...],
                 env_params: Mapping[str, Any]) -> None:
        if bool(config.terminate_resample):
            raise ValueError("terminate_resample must be False so the rollout defines a deterministic cost J(U)")
        self.config = config
        self.env = env
        self.rollout = rollout
        self.ref_slice = ref_slice
        self.env_params = dict(env_params)

    def __call__(self, full_controls: torch.Tensor) -> np.ndarray:
        full_controls = torch.as_tensor(full_controls, dtype=torch.float32, device=self.config.device)
        expected = (int(self.config.num_samples), int(self.config.horizon_steps), int(self.config.nu))
        if tuple(full_controls.shape) != expected:
            raise ValueError(f"full controls must have shape {expected}, got {tuple(full_controls.shape)}")
        _, rewards, _, _ = self.rollout(self.config, self.env, full_controls.clone(), self.ref_slice,
                                        dict(self.env_params))
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.config.device)
        if tuple(rewards.shape) != (int(self.config.num_samples),):
            raise ValueError(f"rollout reward must have shape ({self.config.num_samples},), got {tuple(rewards.shape)}")
        return (-rewards).detach().cpu().numpy().astype(np.float64)
