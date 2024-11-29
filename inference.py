from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
import copy
import gpudrive
import imageio
from gymnasium import spaces
import torch.nn.functional as F
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from algorithms.sb3.ppo.ippo import IPPO
from tqdm import tqdm

# CONFIGURE
TOTAL_STEPS = 90
MAX_CONTROLLED_AGENTS = 128
NUM_WORLDS = 50 # 50
NUM_ROLLOUTS_PER_POLICY_PAIR = 100
device = torch.device('cuda')

env_config = EnvConfig(dynamics_model="classic")
render_config = RenderConfig()
# scene_config = SceneConfig("data/", NUM_WORLDS)

import pyrallis
from baselines.ippo.config import ExperimentConfig

exp_config = pyrallis.parse(config_class=ExperimentConfig)
exp_config.num_worlds = NUM_WORLDS

scene_config = SceneConfig(
    path=exp_config.data_dir,
    num_scenes=exp_config.num_worlds,
    discipline=exp_config.selection_discipline,
    k_unique_scenes=exp_config.k_unique_scenes,
)

# MAKE ENV
env = GPUDriveTorchEnv(
    config=env_config,
    scene_config=scene_config,
    max_cont_agents=MAX_CONTROLLED_AGENTS,  # Number of agents to control
    # device="cpu",
    render_config=render_config,
)

n_envs = env.cont_agent_mask.sum().item()


def policy_inference(policy, obs, done):

    # EDIT_1: Mask out invalid observations (NaN axes and/or dead agents)
    # Create dummy actions, values and log_probs (NaN)
    actions = torch.full(
        fill_value=float("nan"), size=(n_envs,)
    ).to(device)
    log_probs = torch.full(
        fill_value=float("nan"),
        size=(n_envs,),
        dtype=torch.float32,
    ).to(device)
    values = (
        torch.full(
            fill_value=float("nan"),
            size=(n_envs,),
            dtype=torch.float32,
        )
        .unsqueeze(dim=1)
        .to(device)
    )

    # Get indices of alive agent ids
    done_worlds = torch.where(
        (done.nan_to_num(0) * env.get_controlled_agents_mask()).sum(dim=1)
        == env.get_controlled_agents_mask().sum(dim=1)
    )[0]

    if done_worlds.any().item():
        env.sim.reset(done_worlds.tolist())

    dead_agent_mask = done.to(torch.int64) # torch.logical_or(self.dead_agent_mask, done)

    # Convert env_dead_agent_mask to boolean tensor with the same shape as obs_tensor
    alive_agent_mask = ~(
        dead_agent_mask[env.get_controlled_agents_mask()].reshape(
            n_envs, 1
        )
    ).to(torch.bool)

    # indices_alive_agents = torch.nonzero(alive_agent_mask, as_tuple=False).squeeze()  # Shape [K]

    # Use boolean indexing to select elements in obs_tensor
    obs_tensor = obs[env.cont_agent_mask]
    obs_tensor_alive = obs_tensor[
        alive_agent_mask.expand_as(obs_tensor)
    ].reshape(-1, obs_tensor.shape[-1])

    indices_cont_agents = torch.nonzero(env.cont_agent_mask, as_tuple=False)  # Shape [M, 2]
    # indices_alive_in_obs = indices_cont_agents[indices_alive_agents]  # Shape [K, 2]
    # obs_tensor_alive = obs_tensor # obs_tensor[alive_agent_mask.flatten()]

    # Predict actions, vals and log_probs given obs
    actions_tmp, values_tmp, log_prob_tmp = policy(
        obs_tensor_alive
    )

    # Predict actions, vals and log_probs given obs
    (
        actions[alive_agent_mask.squeeze(dim=1)],
        values[alive_agent_mask.squeeze(dim=1)],
        log_probs[alive_agent_mask.squeeze(dim=1)],
    ) = (
        actions_tmp.float(),
        values_tmp.float(),
        log_prob_tmp.float(),
    )

    # Rescale and perform action
    clipped_actions = actions

    if isinstance(env.action_space, spaces.Box):
        if policy.squash_output:
            # Unscale the actions to match env bounds
            # if they were previously squashed (scaled in [-1, 1])
            clipped_actions = policy.unscale_action(
                clipped_actions
            )
        else:
            # Otherwise, clip the actions to avoid out of bound error
            # as we are sampling from an unbounded Gaussian distribution
            clipped_actions = torch.clamp(
                actions, env.action_space.low, env.action_space.high
            )

    world_actions = []
    for i in range(NUM_WORLDS):
        world_indices = torch.nonzero(indices_cont_agents[:, 0] == i, as_tuple=True)[0]
        _world_actions = clipped_actions[world_indices]
        world_actions.append(_world_actions)

    return world_actions

def rollout(policy_a, policy_b):
    # RUN
    obs = env.reset()
    done = env.get_dones()

    agent_a_index = []
    for i in range(NUM_WORLDS):
        index_first_zero = torch.nonzero(done[i] == 0, as_tuple=True)[0][0]
        agent_a_index.append(index_first_zero.item())

    done = torch.zeros([NUM_WORLDS, MAX_CONTROLLED_AGENTS], device=device)
    frames = []

    policy_a_reward = torch.zeros([NUM_WORLDS], device=device)
    for i in range(TOTAL_STEPS):
        # print(f"Step: {i}")

        policy_a_action = policy_inference(policy_a.policy, obs, done)
        policy_b_action = policy_inference(policy_b.policy, obs, done)

        acts = []
        for a_act, b_act in zip(policy_a_action, policy_b_action):
            
            # interleave the policies
            act = torch.cat([a_act[0:1], b_act[1:]])

            # pad out to correct shape
            padding_length = MAX_CONTROLLED_AGENTS - act.size(0)
            padded_acts = F.pad(act, (0, padding_length), "constant", 0)

            acts.append(padded_acts)
        
        acts = torch.vstack(acts)

        # # Take a random actions
        # rand_action = torch.Tensor(
        #     [
        #         [
        #             env.action_space.sample()
        #             for _ in range(
        #                 env_config.max_num_agents_in_scene * NUM_WORLDS
        #             )
        #         ]
        #     ]
        # ).reshape(NUM_WORLDS, env_config.max_num_agents_in_scene)

        # Step the environment
        env.step_dynamics(acts)

        frames.append(env.render(world_render_idx=0))

        obs = env.get_obs()
        reward = env.get_rewards()
        for j in range(NUM_WORLDS):
            policy_a_reward[j] += reward[j,agent_a_index[j]]
        done = env.get_dones()

    # import imageio
    imageio.mimsave("world1.gif", np.array(frames))

    return policy_a_reward

# policies
policy_a = IPPO.load("wandb/run-20241123_105439-gpudrive_11_23_10_38_3scenes/files/policy_10021031.zip", device=device)

policy_bx = [
    IPPO.load("wandb/run-20241123_101909-gpudrive_11_23_10_09_3scenes/files/policy_18854.zip", device=device),
    IPPO.load("wandb/run-20241123_101909-gpudrive_11_23_10_09_3scenes/files/policy_1781326.zip", device=device),
    IPPO.load("wandb/run-20241123_101909-gpudrive_11_23_10_09_3scenes/files/policy_3936818.zip", device=device),
    IPPO.load("wandb/run-20241123_101909-gpudrive_11_23_10_09_3scenes/files/policy_6330063.zip", device=device),
    IPPO.load("wandb/run-20241123_101909-gpudrive_11_23_10_09_3scenes/files/policy_8780169.zip", device=device),
    IPPO.load("wandb/run-20241123_101909-gpudrive_11_23_10_09_3scenes/files/policy_10013074.zip", device=device),
]

start = 0
end = NUM_ROLLOUTS_PER_POLICY_PAIR
num_bins = 50

policy_pair_reward = []

for policy_b in tqdm(policy_bx):

    policy_a_reward = []

    for i in tqdm(range(NUM_ROLLOUTS_PER_POLICY_PAIR)):

        _policy_a_reward = rollout(policy_a, policy_b)
        policy_a_reward.append(_policy_a_reward)

    policy_a_reward = torch.vstack(policy_a_reward)

    policy_pair_reward.append(copy.copy(policy_a_reward))

env.close()
