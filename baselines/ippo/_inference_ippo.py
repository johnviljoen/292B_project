"""
This file contains the functionality to run different agents together in a single
GPUDrive environment

We do this by basing it off of the ippo.py file under algorithms/sb3/ppo/ippo
"""

import torch
import numpy as np
from typing import List, Dict
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

def run_inference(
    env: VecEnv,
    policy_a: ActorCriticPolicy,
    policy_b: ActorCriticPolicy,
    num_steps: int,
    device: torch.device,
) -> Dict[str, List]:
    """
    Run inference using two different policies and collect data for analysis.

    :param env: The vectorized environment.
    :param policy_a: The first policy (used for specific agents).
    :param policy_b: The second policy (used for all other agents).
    :param num_steps: Number of steps to run.
    :param policy_a_agent_indices: Indices of agents using policy_a.
    :param device: PyTorch device.
    :param action_space: Action space of the environment.
    :param controlled_agent_mask: Mask of controlled agents in the environment.
    :param dead_agent_mask: Mask of dead agents in the environment.
    :return: Collected data for analysis.
    """

    def create_and_log_video():
        base_env = env._env
        
        action_tensor = torch.zeros(
            (base_env.num_worlds, base_env.max_agent_count)
        )

        obs = base_env.reset()
        control_mask = env.controlled_agent_mask.clone()
        obs = obs[control_mask].reshape(
            env.num_envs, env.obs_dim
        )

        frames = []

        for step_num in range(exp_config.episode_len):
            actions, _ = policy.predict(obs.detach().cpu().numpy())




    # Initialize storage for collected data
    collected_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'infos': [],
    }

    # Reset environment
    obs = env.reset()
    # Convert obs to tensor
    last_obs = torch.tensor(obs).to(device)
    # Assuming last_episode_starts is all True at the beginning
    last_episode_starts = np.ones((env.num_envs,), dtype=bool)

    policy_a.policy.set_training_mode(False)
    policy_b.policy.set_training_mode(False)

    for step in range(num_steps):
        with torch.no_grad():
            obs_tensor = last_obs

            # Create dummy actions
            actions = torch.full(
                fill_value=float("nan"), size=(env.num_envs,)
            ).to(device)

            # Get indices of alive agents
            # alive_agent_mask = ~(
            #     dead_agent_mask[controlled_agent_mask].reshape(
            #         env.num_envs, 1
            #     )
            # ).squeeze(dim=1)

            # Create masks for agents using policy A and policy B
            policy_a_agent_mask = torch.zeros(
                env.num_envs, dtype=torch.bool, device=device
            )
            policy_a_agent_mask[policy_a_agent_indices] = True
            policy_b_agent_mask = ~policy_a_agent_mask

            # Create masks for alive agents using policy A and policy B
            alive_and_policy_a_mask = alive_agent_mask & policy_a_agent_mask
            alive_and_policy_b_mask = alive_agent_mask & policy_b_agent_mask

            # Select observations for alive agents using each policy
            obs_tensor_policy_a = obs_tensor[alive_and_policy_a_mask]
            obs_tensor_policy_b = obs_tensor[alive_and_policy_b_mask]

            # Compute actions for policy A agents
            if obs_tensor_policy_a.shape[0] > 0:
                actions_a, _ = policy_a.predict(
                    obs_tensor_policy_a, deterministic=True
                )
            else:
                actions_a = None

            # Compute actions for policy B agents
            if obs_tensor_policy_b.shape[0] > 0:
                actions_b, _ = policy_b.predict(
                    obs_tensor_policy_b, deterministic=True
                )
            else:
                actions_b = None

            # Assign actions back to overall actions tensor
            if actions_a is not None:
                actions[alive_and_policy_a_mask] = torch.tensor(
                    actions_a
                ).float().to(device)
            if actions_b is not None:
                actions[alive_and_policy_b_mask] = torch.tensor(
                    actions_b
                ).float().to(device)

        # Rescale and perform action
        if isinstance(action_space, spaces.Box):
            if policy_a.squash_output:
                # Unscale the actions to match env bounds
                clipped_actions = policy_a.unscale_action(actions)
            else:
                # Clip the actions to avoid out of bound error
                clipped_actions = torch.clamp(
                    actions, action_space.low[0], action_space.high[0]
                )
            # Convert to numpy array
            clipped_actions = clipped_actions.cpu().numpy()
        else:
            # Discrete actions
            clipped_actions = actions.cpu().numpy()

        # Take a step in the environment
        new_obs, rewards, dones, infos = env.step(clipped_actions)

        # Store data for analysis
        collected_data['observations'].append(last_obs.cpu().numpy())
        collected_data['actions'].append(clipped_actions)
        collected_data['rewards'].append(rewards)
        collected_data['dones'].append(dones)
        collected_data['infos'].append(infos)

        # Update last_obs and last_episode_starts
        last_obs = torch.tensor(new_obs).to(device)
        last_episode_starts = dones

        # If all episodes are done, you can reset the environment or break
        if np.all(dones):
            break

    # Return collected data
    return collected_data

if __name__ == "__main__":
    
    from pygpudrive.env.config import EnvConfig, SceneConfig
    from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
    import dataclasses
    import pyrallis
    from baselines.ippo.config import ExperimentConfig
    from algorithms.sb3.ppo.ippo import IPPO

    exp_config = pyrallis.parse(config_class=ExperimentConfig)

    scene_config = SceneConfig(
        path=exp_config.data_dir,
        num_scenes=exp_config.num_worlds,
        discipline=exp_config.selection_discipline,
        k_unique_scenes=exp_config.k_unique_scenes,
    )

    # ENVIRONMENT CONFIG
    env_config = dataclasses.replace(
        EnvConfig(),
        reward_type=exp_config.reward_type,
        collision_weight=exp_config.collision_weight,
        goal_achieved_weight=exp_config.goal_achieved_weight,
        off_road_weight=exp_config.off_road_weight,
    )

    # MAKE SB3-COMPATIBLE ENVIRONMENT
    env = SB3MultiAgentEnv(
        config=env_config,
        scene_config=scene_config,
        # Control up to all agents in the scene
        max_cont_agents=env_config.max_num_agents_in_scene,
        device=exp_config.device,
    )

    num_steps = 100
    device = torch.device("cuda")

    policy_a = IPPO.load("wandb/run-20241123_105439-gpudrive_11_23_10_38_3scenes/files/policy_10021031.zip", device=device)
    policy_b = IPPO.load("wandb/run-20241123_101909-gpudrive_11_23_10_09_3scenes/files/policy_10013074.zip", device=device)

    run_inference(
        env,
        policy_a,
        policy_b,
        num_steps,
        device,
    )