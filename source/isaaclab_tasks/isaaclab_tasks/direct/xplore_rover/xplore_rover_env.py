# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch

from lidar_viewer import LidarViewer

from isaaclab_assets.robots.xplore_rover import XPLORE_ROVER_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LidarPatternCfg

from isaaclab.sim import SphereLightCfg, DiskLightCfg, spawn_light

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

@configclass
class XploreRoverEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 150.0
    decimation = 2
    propulsion_action_scale = 20 # max 20 m/s
    steering_action_scale = torch.pi/4 # max +/- 45° turn
    action_space = 2
    observation_space = 36
    state_space = 0
    max_distance_lidar = 45.0 # 45m according to the datasheet
    goal = [9, 16, 0.65]

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/mars_terrain",
        terrain_type="usd",
        usd_path="/home/chris/Documents/xplore/IsaacLab/worlds/marsyard2024.world.usdc",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.4,
            dynamic_friction=0.2,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/dummy_link",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1)),
        attach_yaw_only=True,
        pattern_cfg=LidarPatternCfg(
            channels=32,
            vertical_fov_range=(-45.0, 45.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=360/2048
        ),
        #debug_vis=True,
        mesh_prim_paths=["/World/mars_terrain"],
        max_distance=max_distance_lidar,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = XPLORE_ROVER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.0001
    actions_cost_scale: float = 0.0001
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 0.31

    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1

    distance_reward_scale: float = 1.0

    lidar_cost_scale : float = 1.0

    yaw_cost_scale : float = 3.0


class XploreRoverEnv(DirectRLEnv):
    cfg: XploreRoverEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.propulsion_action_scale = self.cfg.propulsion_action_scale
        self.steering_action_scale = self.cfg.steering_action_scale
        self.max_distance_lidar = self.cfg.max_distance_lidar
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor(self.cfg.goal, dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.prev_yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)

        self.stuck_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim.device)

        self.lidar_viewer = LidarViewer()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # lidar
        self.lidar = RayCaster(self.cfg.lidar)
        self.scene.sensors["lidar"] = self.lidar

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # add goal marker
        goal_light_cfg = DiskLightCfg(
            radius=0.2,
            color=(0.0, 1.0, 0.0),
            intensity=1.0e5,
        )
        goal_light_cfg.func("/World/envs/env_0/goal_light", goal_light_cfg, self.cfg.goal)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        # actions[:, 0]: driving action (1: forward, -1: backward, 0: stop)
        # actions[:, 1]: steering action (1: left, -1: right, 0: straight)
        L = 0.84 # 0.84 m
        W = 0.74 # 0.74 m
        driving_cmd = self.actions[:, 0]  # 1: forward, -1: backward, 0: stationary
        steering_cmd = self.actions[:, 1]  # 1: left , -1: right, 0: straight)

        radius_min = 0.45
        epsilon = 1e-4
        radius = torch.where(
            steering_cmd.abs() > epsilon,
            (1 / steering_cmd.abs()) - 1 + radius_min,
            torch.full_like(steering_cmd, 1e6)
        )
        wheel_in = torch.atan(L / (radius - W / 2))
        wheel_out = torch.atan(L / (radius + W / 2))
        if steering_cmd >= 0:
            steering_lf = -wheel_in
            steering_lh = -wheel_in
            steering_rf = wheel_out
            steering_rh = wheel_out
        else:
            steering_lf = wheel_out
            steering_lh = wheel_out
            steering_rf = -wheel_in
            steering_rh = -wheel_in

        steering_actions = torch.stack([steering_lf, steering_lh, steering_rf, steering_rh], dim=-1)

        propulsion_speed = driving_cmd * self.propulsion_action_scale
        propulsion_actions = torch.stack(
            [propulsion_speed, -propulsion_speed, propulsion_speed, -propulsion_speed], dim=-1
        )

        self.robot.set_joint_velocity_target(propulsion_actions, joint_ids=self._joint_dof_idx[4:])
        self.robot.set_joint_position_target(steering_actions, joint_ids=self._joint_dof_idx[:4])

    def _compute_intermediate_values(self):
        self.base_position, self.base_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel
        self.lidar_3D_data = self.lidar.data.ray_hits_w

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.prev_potentials,
            self.potentials,
            self.distance_to_goal,
            self.yaw_delta,
            self.prev_yaw,
            self.stuck_counter,
            self.lidar_distance_data
        ) = compute_intermediate_values(
            self.targets,
            self.base_position,
            self.base_rotation,
            self.velocity,
            self.ang_velocity,
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
            self.prev_yaw,
            self.stuck_counter,
            self.lidar_3D_data,
            self.max_distance_lidar,
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
                self.lidar_distance_data,
            ),
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.distance_to_goal,
            self.cfg.distance_reward_scale,
            self.yaw,
            self.prev_yaw,
            self.cfg.yaw_cost_scale,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        flipped = self.up_proj < 0.2
        stuck = self.stuck_counter > 50
        return flipped | stuck, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self.stuck_counter[env_ids] = 0
        self.prev_yaw[env_ids] = 0.0

        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
        actions: torch.Tensor,
        reset_terminated: torch.Tensor,
        up_weight: float,
        heading_weight: float,
        heading_proj: torch.Tensor,
        up_proj: torch.Tensor,
        dof_vel: torch.Tensor,
        potentials: torch.Tensor,
        prev_potentials: torch.Tensor,
        actions_cost_scale: float,
        energy_cost_scale: float,
        dof_vel_scale: float,
        death_cost: float,
        alive_reward_scale: float,
        distance_to_goal: torch.Tensor,
        distance_reward_scale: float,
        yaw : torch.Tensor,
        prev_yaw : torch.Tensor,
        yaw_cost_scale : float,

):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    propulsion_velocities = dof_vel[:, 4:]
    electricity_cost = torch.sum(
        torch.abs(actions[:, 0].unsqueeze(-1) * propulsion_velocities * dof_vel_scale),
        dim=-1,
    )

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    distance_reward = distance_reward_scale / (1.0 + distance_to_goal)

    yaw_delta = torch.abs(yaw - prev_yaw)
    yaw_smooth_cost = yaw_cost_scale * normalize_angle(yaw_delta)



    total_reward = (
            progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            + distance_reward
            - actions_cost_scale * actions_cost
            - energy_cost_scale * electricity_cost
            - yaw_smooth_cost
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    # print("###################################################")
    # print("progress_reward = ", progress_reward)
    # print("alive_reward = ", alive_reward)
    # print("up_reward = ", up_reward)
    # print("heading_reward = ", heading_reward)
    # print("distance_reward = ", distance_reward)
    # print("actions_cost = ", actions_cost_scale * actions_cost)
    # print("electricity_cost = ", energy_cost_scale * electricity_cost)
    # print("lidar_cost = ", lidar_cost)
    # print("yaw_smooth_cost = ", yaw_smooth_cost)
    # print("Total reward:", total_reward.item())
    return total_reward


@torch.jit.script
def compute_intermediate_values(
        targets: torch.Tensor,
        base_position: torch.Tensor,
        base_rotation: torch.Tensor,
        velocity: torch.Tensor,
        ang_velocity: torch.Tensor,
        inv_start_rot: torch.Tensor,
        basis_vec0: torch.Tensor,
        basis_vec1: torch.Tensor,
        potentials: torch.Tensor,
        prev_potentials: torch.Tensor,
        dt: float,
        prev_yaw: torch.Tensor,
        stuck_counter: torch.Tensor,
        lidar_3D_data : torch.Tensor,
        max_distance_lidar : float,
):
    to_target = targets - base_position
    to_target[:, 2] = 0.0

    base_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        base_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        base_quat, velocity, ang_velocity, targets, base_position
    )

    yaw_delta = normalize_angle(torch.abs(yaw - prev_yaw))
    prev_yaw = yaw.clone()

    to_target = targets - base_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    distance_to_goal = torch.norm(targets - base_position, dim=-1)

    speed = torch.norm(velocity[:, :2], dim=-1)
    is_stuck_now = speed < 0.05

    stuck_counter[is_stuck_now] += 1
    stuck_counter[~is_stuck_now] = 0

    lidar_distance_data = torch.norm((lidar_3D_data - base_position.unsqueeze(1)), dim=-1)
    lidar_distance_data = torch.nan_to_num(lidar_distance_data, nan=max_distance_lidar, posinf=max_distance_lidar, neginf=0.0)

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        prev_potentials,
        potentials,
        distance_to_goal,
        yaw_delta,
        prev_yaw,
        stuck_counter,
        lidar_distance_data,
    )
