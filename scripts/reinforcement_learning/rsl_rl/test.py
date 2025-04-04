# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    #env = RslRlVecEnvWrapper(env)
    dt = env.unwrapped.physics_dt

    # reset environment
    #obs, _ = env.unwrapped._get_observations()
    timestep = 0
    env_unwrapped = env.unwrapped
    action = torch.zeros((env_unwrapped.num_envs, env_unwrapped.action_space.shape[1]),device=env_unwrapped.device)

    # speed = 0
    # turn = 0.5
    # #steering
    # action[0, 0] = -turn  # lf
    # action[0, 1] = -turn  # lh
    # action[0, 2] = turn  # rf
    # action[0, 3] = turn  # rh
    # # driving
    # action[0, 4] = speed # lf
    # action[0, 5] = -speed # lh
    # action[0, 6] = speed  # rf
    # action[0, 7] = -speed  # rh
    action[0,0] = 0.5
    action[0,1] = 0

    while simulation_app.is_running():
        # step simulation avec l'action définie
        obs = env_unwrapped.step(action)

        # policy = obs[0]["policy"]
        #
        # print("🔸 velocity:", policy[:, 0:3])  # self.vel_loc → 3
        # print("🔸 angular velocity:", policy[:, 3:6])  # self.angvel_loc * scale → 3
        # print("🔸 yaw:", policy[:, 6:7])  # normalize_angle(self.yaw) → 1
        # print("🔸 roll:", policy[:, 7:8])  # normalize_angle(self.roll) → 1
        # print("🔸 angle_to_target:", policy[:, 8:9])  # normalize_angle(self.angle_to_target) → 1
        # print("🔸 up projection:", policy[:, 9:10])  # self.up_proj → 1
        # print("🔸 heading projection:", policy[:, 10:11])  # self.heading_proj → 1
        # print("🔸 dof_vel_scaled:", policy[:, 11:19])  # 8 DoFs → 8
        # print("🔸 actions:", policy[:, 19:27])  # 8 actions → 8
        # print("🔸 lidar:", policy[:, 27:])

        simulation_app.update()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
