# This script utilises the IsaacLab manager-based base environment tutorial
# Instead of creating the cartpole environment it creates the environment for screw removal
# 
# In its current state the environment is created using 00_screw_env_cfg.py and an action 
# and observatio5ns manager are used to interact with the Franka

# This script must be ran by the IsaacLab bash script:
# ./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-IK-Franka-v0
# ./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-IK-Franka-v0
#
# To add new training environments to the bash script edit the init file

#-------------------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------------------

#Standard python imports
import math
import torch
from dataclasses import MISSING

#Isaac specific imports
from omni.isaac.lab.envs import (ManagerBasedRLEnvCfg, 
                                 ManagerBasedRLEnv
)
from omni.isaac.lab.utils import configclass 
from omni.isaac.lab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    ActionTermCfg as ActionTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm
)
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

import omni.isaac.lab.sim as sim_utils                           # Provides the ground plane and dome light
from omni.isaac.lab.assets import  AssetBaseCfg                  # Provides base asset configuration class  (generic structure for storing assets)       
from omni.isaac.lab.scene import  InteractiveSceneCfg            # Provides interactive scene cfg (generic structure for scene cfg)
from omni.isaac.lab.utils import configclass                     # Config class decorator to override some features of python 3.7 dataclasses
                                                                 # Dataclasses require type annotation for memebres, this stops that. More info here: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.configclass 
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG        # Provides the Franka Panda robot configuration  



#Custom package imports
#from screw_env_cfg import FrankaTestSceneCfg
import omni.isaac.lab_tasks.manager_based.custom.inv_kin.cfg.mdp as mdp
#-------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------
#                               Configuration classes:
# 
# Sim Management Classes:
# FrankaTestSceneCfg - Sets up the physical scene
# EventCfg
# TerminationsCfg
# CommandsCfg
# ManagerBasedEnvCfg

# RL Classes:
# ObservationsCfg
# ActionsCfg
# RewardsCfg

# FrankaEnvCfg - Instantiates the other environments as a manager-based RL environment
#-------------------------------------------------------------------------------------------

@configclass
class FrankaTestSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    #assets_folder = f"/home/ethanallan175/IsaacLab/source/standalone/custom/assets/"

    # Spawns the ground plane under the world prim
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
                            spawn=sim_utils.GroundPlaneCfg())

    # Spawns the assembled screw assembly under the assembly prim
    
    # assembly = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/assembly",                                                                # ENV_REGEX_NS allows the part to be replicated for each environment instane
    #                         init_state=AssetBaseCfg.InitialStateCfg(pos=[0.3, 0.3, 0.0], rot=[0.707, 0 ,0.0, 0.707]),           # Pose and rotation in format [x,y,z] and [w,a,b,c] respectively           # 
    #                         spawn=sim_utils.UsdFileCfg(usd_path=assets_folder + "assembled.usd" ))
    # Spawns dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Spawns in the panda. Currently set to the high stiffness version.
    robot =FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    
#-------------------------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(        #Position of the target
            pos_x=(0.3, 0.65),
            pos_y=(-0.3, 0.3),
            pos_z=(0.2, 0.5),
            roll=(0.0, 0.0),
            pitch=(3.14, 3.14),  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )
#-------------------------------------------------------------------------------------------
@configclass
class EventCfg:
#Used to define all events that can occur in the environment and what to do
#Currently just using reset but there are also startup and interval

    # Resets all the joints to some random position
    reset_robot_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

#-------------------------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out=DoneTerm(func=mdp.time_out,time_out=True)
    

#-------------------------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # Need to come up with basic reward function, will maybe steal this from franka reach
    # Distance to point makes the most sense, just need to work out how to get there
    
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"), "command_name": "ee_pose"},
    )



#-------------------------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    # Environment observation specifications
    
    @configclass
    class PolicyCfg(ObsGroup):
        # Observation group for this policy

        # Observation terms
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)


        #Saved this for RL base env - can delete once we have a working version
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

        #Not sure what this does yet but it is in the cartpole example
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    #Create instance of policy
    policy: PolicyCfg = PolicyCfg()

#-------------------------------------------------------------------------------------------
@configclass
class ActionsCfg:

    # Assigns action as joint effort tensor
    # Provides the asset name (Franka), joint names (all joints except gripper)and scale
    arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
    arm_action =DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
    

#-------------------------------------------------------------------------------------------
# Manager-based env config is used to used to instantiate all of the 
# other managers and create the environment.
@configclass
class FrankaEnvCfg(ManagerBasedRLEnvCfg):

    # Scene configuration
    scene = FrankaTestSceneCfg(num_envs=2048, env_spacing=2.0)

    # Manager instantiation
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards= RewardsCfg()
    commands= CommandsCfg()
    terminations= TerminationsCfg()
    episode_length_s= 12
    # Decimation rate
    decimation = 2

    def __post__init__(self):
        self.viewer.eye = [4.5,0.0,6.0]
        self.viewer.lookat = [0.8, 0.0, 0.5]
        self.decimation =2
        self.sim.dt = 1.0 /60.0



