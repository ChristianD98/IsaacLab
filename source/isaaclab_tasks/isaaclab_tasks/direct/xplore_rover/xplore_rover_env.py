# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from isaaclab_assets.robots.xplore_rover import XPLORE_ROVER_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.rover_locomotion.rover_locomotion_env import RoverLocomotionEnv

from isaaclab.sensors import RayCasterCfg, patterns

@configclass
class XploreRoverEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 0.5
    action_space = 8
    observation_space = 36
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/mars_terrain",
        terrain_type="usd",
        usd_path="/home/chris/Documents/xplore/IsaacLab/worlds/marsyard2024.world.usdc",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = XPLORE_ROVER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 0.31

    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1


class XploreRoverEnv(RoverLocomotionEnv):
    cfg: XploreRoverEnvCfg

    def __init__(self, cfg: XploreRoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)