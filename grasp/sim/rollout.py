# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Source: SPIDER, https://github.com/facebookresearch/spider, spider/optimizers/sampling.py (make_rollout_fn)
# at commit 44717007de41cbef7565dff7ff9f4453557a2d3d, adapted for GRASP.
# Licensed under CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.

"""Batched rollout of control sequences against a reference."""

from __future__ import annotations

import torch

from grasp.config import Config


def make_rollout_fn(
    step_env,
    save_state,
    load_state,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    save_env_params,
    load_env_params,
    copy_sample_state,
):
    def rollout(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_param: dict,
    ) -> torch.Tensor:
        """Rollout the control actions to get reward

        Args:
            config: Config
            env: Environment
            ctrls: Control actions, shape (horizon_steps, nu)
            ref_slice: Reference slice, shape (nq, nv, nu, ncon, ncon_pos)

        Returns:
            Reward, shape (num_samples,)
            Info: dict, including trace (N, H, n_trace, 3)
        """
        # save initial state
        init_state = save_state(env)

        # save initial env params (active group pointer)
        init_env_param = save_env_params(config, env)

        # select rollout graph/data pointers for this rollout
        env = load_env_params(config, env, env_param)

        # rollout to get reward
        N, H = ctrls.shape[:2]
        trace_list = []
        cum_rew = torch.zeros(N, device=config.device)
        info_list = []
        for t in range(H):
            # step the environment
            step_env(config, env, ctrls[:, t])  # (N, nu)
            # get reward
            ref = [r[t] for r in ref_slice]
            rew, info = (
                get_reward(config, env, ref)
                if t < H - 1
                else get_terminal_reward(config, env, ref)
            )
            cum_rew += rew
            # get trace
            trace = get_trace(config, env)
            trace_list.append(trace)
            info_list.append(info)
            # Resampling: replace bad samples with good samples periodically
            terminate = get_terminate(config, env, ref)
            if (
                config.terminate_resample
                and t < H - 1
                and terminate.any()
                and (not terminate.all())
            ):
                # if terminate.all():
                #     # if all terminate, replace low reward samples with high reward samples
                #     # top indices: top 50% samples
                #     good_indices = torch.topk(
                #         cum_rew, k=int(0.5 * config.num_samples), largest=True
                #     ).indices
                #     bad_indices = torch.topk(
                #         cum_rew, k=int(0.5 * config.num_samples), largest=False
                #     ).indices
                # else:
                # bad indices is terminate
                bad_indices = torch.nonzero(terminate).squeeze(-1)
                good_indices = torch.nonzero(~terminate).squeeze(-1)
                # make sure good indices shape is the same as bad indices
                if good_indices.shape[0] > bad_indices.shape[0]:
                    good_indices = good_indices[: bad_indices.shape[0]]
                elif good_indices.shape[0] < bad_indices.shape[0]:
                    random_idx = torch.randint(
                        0, good_indices.shape[0], (bad_indices.shape[0],)
                    )
                    good_indices = good_indices[random_idx]

                # Replace bad sample simulation state with good sample simulation state
                copy_sample_state(config, env, good_indices, bad_indices)

                # Replace bad samples control with good samples (for initial control and current timestep only)
                ctrls[bad_indices, :t] = ctrls[good_indices, :t]

                # Replace bad samples cumulative reward with good samples reward
                cum_rew[bad_indices] = cum_rew[good_indices]

        info_combined = {
            k: torch.stack([info[k] for info in info_list], axis=0)
            for k in info_list[0].keys()
        }
        mean_info = {k: v.mean(axis=0) for k, v in info_combined.items()}
        mean_rew = cum_rew / H

        # reset all envs back to initial state
        env = load_state(env, init_state)

        # reset env params
        env = load_env_params(config, env, init_env_param)

        # get info
        trace_list = torch.stack(trace_list, dim=1)
        info = {
            "trace": trace_list,  # (N, H, n_trace, 3)
            **mean_info,
        }
        return ctrls, mean_rew, terminate, info

    return rollout
