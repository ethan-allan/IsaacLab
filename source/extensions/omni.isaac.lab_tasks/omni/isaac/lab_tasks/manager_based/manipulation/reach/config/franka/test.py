# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import (
    ActionTermCfg as ActionTerm,
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# Import pre-defined robot configurations
from omni.isaac.lab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG


"""
Reward Functions
These functions compute different components of the reward signal used for reinforcement learning
"""

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute L2 norm of position error between commanded and current end-effector position."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # Transform desired position from local to world coordinates
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_link_state_w[:, :3], 
        asset.data.root_link_state_w[:, 3:7], 
        des_pos_b
    )
    curr_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Compute smoothed position error using tanh kernel for better gradient properties."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_link_state_w[:, :3],
        asset.data.root_link_state_w[:, 3:7],
        des_pos_b
    )
    curr_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def orientation_command_error(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Compute orientation error using quaternion shortest path."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_link_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], 3:7]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


"""
Scene Configuration
Defines the physical environment including robot, table, lighting, etc.
"""

@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the robotic reaching scene."""
    
    # World elements
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), 
            rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )
    
    # Robot configuration - will be set in derived classes
    robot: ArticulationCfg = MISSING
    
    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


"""
MDP Configuration Components
These classes define the different aspects of the Markov Decision Process
"""

@configclass
class CommandsCfg:
    """Configuration for command generation."""
    
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=MISSING,
            yaw=(-3.14, 3.14),
        ),
    )

@configclass
class ObservationsCfg:
    """Configuration for observations provided to the policy."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Define observation space for the learning policy."""
        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        pose_command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "ee_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Configuration for reward terms."""
    
    end_effector_position_tracking = RewTerm(
        func=position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING), 
            "command_name": "ee_pose"
        },
    )
    
    end_effector_position_tracking_fine_grained = RewTerm(
        func=position_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "ee_pose"
        },
    )
    
    end_effector_orientation_tracking = RewTerm(
        func=orientation_command_error,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose"
        },
    )
    
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


"""
Environment Configurations
These classes define different variants of the reaching environment
"""

@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for reach environment."""
    
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    
    def __post_init__(self):
        """Initialize simulation parameters."""
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0

@configclass
class FrankaJointPosReachEnvCfg(ReachEnvCfg):
    """Environment configuration for Franka with joint position control."""
    
    def __post_init__(self):
        super().__post_init__()
        # Configure Franka robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Set up joint position control
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True
        )
        # Configure end-effector tracking
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

@configclass
class FrankaIKReachEnvCfg(ReachEnvCfg):
    """Environment configuration for Franka with inverse kinematics control."""
    
    def __post_init__(self):
        super().__post_init__()
        # Configure Franka robot with high PD gains for better IK tracking
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Set up inverse kinematics control
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]
            ),
        )

# Configuration for demonstration/testing with fewer environments
@configclass
class FrankaReachEnvCfg_PLAY(FrankaJointPosReachEnvCfg):
    """Simplified environment configuration for testing."""
    
    def __post_init__(self):
        super().__post_init__()
        # Reduce number of environments for faster testing
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise/corruption
        self.observations.policy.enable_corruption = False

FrankaReachEnvCfg_PLAY()