import wandb
import pyrallis
import torch
import os
import dataclasses
from typing import List
from datetime import datetime
from algorithms.sb3.ppo.ippo import IPPO
from baselines.ippo.config import ExperimentConfig
from pygpudrive.env.config import EnvConfig, SceneConfig
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
from stable_baselines3.common.utils import set_random_seed

def load_agents(agent_paths: List[str], device: str):
    """
    Load trained agents from the specified paths.

    :param agent_paths: List of file paths to the saved agent models.
    :param device: Device to load the agents on ('cpu' or 'cuda').
    :return: List of loaded agent models.
    """
    agents = []
    for path in agent_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Agent model not found at path: {path}")
        agent = IPPO.load(path, device=device)
        agents.append(agent)
    return agents

def run_inference(exp_config: ExperimentConfig, scene_config: SceneConfig, agent_paths: List[str]):
    """Run inference with multiple IPPO agents in a mixed setting."""
    
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

    # SET RANDOM SEED
    set_random_seed(exp_config.seed, using_cuda=exp_config.device == "cuda")

    # LOAD AGENTS
    agents = load_agents(agent_paths, device=exp_config.device)
    num_agents = len(agents)

    # RUN INFERENCE
    obs = env.reset()
    done = False
    while not done:
        actions = {}
        for i, agent_id in enumerate(env.agent_ids):
            # Select agent model for this agent_id
            agent_idx = i % num_agents  # Loop over agents if fewer models than agents
            agent = agents[agent_idx]
            # Get observation for this agent
            agent_obs = obs[agent_id]
            # Get action from the agent model
            action, _ = agent.predict(agent_obs, deterministic=True)
            actions[agent_id] = action
        # Step the environment
        obs, rewards, dones, infos = env.step(actions)
        # Check if all agents are done
        done = all(dones.values())

    env.close()

if __name__ == "__main__":
    # Parse experiment configuration
    exp_config = pyrallis.parse(config_class=ExperimentConfig)

    # Define scene configuration
    scene_config = SceneConfig(
        path=exp_config.data_dir,
        num_scenes=exp_config.num_worlds,
        discipline=exp_config.selection_discipline,
        k_unique_scenes=exp_config.k_unique_scenes,
    )

    # Specify paths to agent models
    # You can mix and match different agents by providing their saved model paths here
    agent_paths = [
        "wandb/run-20241123_105439-gpudrive_11_23_10_38_3scenes/files/policy_10021031.zip",
        # "path/to/agent_model_2.zip",
        # "path/to/agent_model_3.zip",
        # Add more paths as needed
    ]

    # Run inference with the specified agents
    run_inference(exp_config, scene_config, agent_paths)
