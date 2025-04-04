# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Ant robot."""

from __future__ import annotations

from enum import Enum
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Configuration
##

XPLORE_ROVER_CFG = ArticulationCfg(
    prim_path="/World/xplore_rover",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/chris/Documents/xplore/IsaacLab/xplore_description/rover.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "steering_lf_joint": 0.0,
            "steering_rf_joint": 0.0,
            "steering_lh_joint": 0.0,
            "steering_rh_joint": 0.0,
            "driving_lf_joint": 0.0,
            "driving_rf_joint": 0.0,
            "driving_lh_joint": 0.0,
            "driving_rh_joint": 0.0,
        },
    ),
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["steering_.*_joint"],
            stiffness=100,
            damping=10,
            effort_limit = 20,
        ),
        "driving": ImplicitActuatorCfg(
            joint_names_expr=["driving_.*_joint"],
            stiffness=0,
            damping=2,
            effort_limit = 100,

        ),
    },
)
