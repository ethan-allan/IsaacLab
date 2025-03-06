# This script utilises the IsaacLab manager-based base environment tutorial
# Instead of creating the cartpole environment it creates the environment for screw removal
# 
# In its current state the environment is created using 00_screw_env_cfg.py and an action 
# and observatio5ns manager are used to interact with the Franka


# Plan:
# 1. 

#----------------------------------------------------------------------------------------------
# Generic IsaacLab startup code.
# Reads cmd line arguments and creates a simulation app instance with them.
# Currently this demo won't do anything with cmd line arguments but normally used for env configs.
#----------------------------------------------------------------------------------------------


# launch omniverse app

#---------------------------------------------------------------------------------------------


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
    TerminationTermCfg as DoneTerm,
    CurriculumTermCfg as CurrTerm
)
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg, FrameTransformerCfg
import omni.isaac.lab.sim as sim_utils                           # Provides the ground plane and dome light
from omni.isaac.lab.assets import  AssetBaseCfg, RigidObjectCfg, RigidObject, ArticulationCfg   # Provides base asset configuration class  (generic structure for storing assets)       
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.scene import  InteractiveSceneCfg            # Provides interactive scene cfg (generic structure for scene cfg)
from omni.isaac.lab.utils import configclass                     # Config class decorator to override some features of python 3.7 dataclasses
                                                                 # Dataclasses require type annotation for memebres, this stops that. More info here: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.configclass 
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG        # Provides the Franka Panda robot configuration  
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


#Custom package imports
#from screw_env_cfg import FrankaTestSceneCfg

from . import mdp
from . import custom
#-------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------
# Configuration classes 
# Manager-based base env only supports the following configuration classes:
# 1. ActionsCfg
# 2. ObservationsCfg
# 3. EventCfg
# 4. TerminationsCfg
# 5. RewardsCfg
# 6. CommandsCfg
# 7. ManagerBasedEnvCfg
#-------------------------------------------------------------------------------------------

@configclass
class FrankaTestSceneCfg(InteractiveSceneCfg):
    """Configuration for screw pick and place"""

    # Spawns the ground plane under the world prim
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
                            spawn=sim_utils.GroundPlaneCfg())
        
    #Spawns in the peg rigid object
    #Don't use initial state if you are using gym
    peg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/peg",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/home/ethanallan175/IsaacLab/source/extensions/custom_assets/peg/peg.usd",
               
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    enable_gyroscopic_forces=True,
                    disable_gravity=False
                )
            ),  
        )
    sample = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/sample",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/home/ethanallan175/IsaacLab/source/extensions/custom_assets/sample/sample.usd",
               
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    enable_gyroscopic_forces=True,
                    disable_gravity=False
                )
            ),  
        )
    
    # Spawns dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Spawns in the panda. Currently set to the low stiffness version.
    robot =FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    
    # Cannot define visual markers in interactive scene, will define later
    ee_frame: FrameTransformerCfg = MISSING

#-------------------------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """Command terms for the MDP."""


    #This is very unclear but it generates the goal pose
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",  # will be set by agent env cfg
        resampling_time_range=(60.0, 60.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.4, 0.4), pos_y=(-0.7,-0.6), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )
#-------------------------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    #Linear reward based on the distance between the object and the end effector
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting = RewTerm(func=mdp.object_height, weight=1.0)

    #Punishes the agent for colliding with the ground
    #ground_collision = RewTerm(func=mdp.punish_ground_collision, params={"ground_height": 0.0}, weight=-100.0)
    #Flat reward for keeping the screw above the minimal height
    #lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)
    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )

    # Penalise rate of change of actions, dampens shakiness?
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    #Penalises use of joints (seems to be an alternative to penalising time)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

#-------------------------------------------------------------------------------------------
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    #After 10,000 steps the penalties are increased 
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


#-------------------------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """Environment observation specifications """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Generic manipulator observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # Observations for pick and place
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        
        #target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        
        #Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

#-------------------------------------------------------------------------------------------
@configclass
class ActionsCfg:

    # Assigns action as joint effort tensor
    # Provides the asset name (Franka), joint names (all joints except gripper)and scale

    arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
    # Gripper action, either open or closed
    gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.08},
            close_command_expr={"panda_finger_.*": 0.0},
        )


#------------------------------------------------------------------------------------------- 
@configclass
class EventCfg:
    """Configuration for events"""

    # Resets all the joints to some random position
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # # Resets the peg to a random position in the environment
    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (0.3, 0.6), "y": (-0.4, 0.4), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("peg", body_names="peg"),
    #     },
    # )

    

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.6), "y": (-0.4, 0.4), "z": (0.0, 0.0), "roll": (1.57, 1.57), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
       
            "asset_cfg": SceneEntityCfg("sample", body_names="sample"),
        },
    )
    reset_peg = EventTerm(
        func=custom.reset_peg,
        mode="reset",
        params={
            "sample_cfg": SceneEntityCfg("sample", body_names="sample"),
            "peg_cfg": SceneEntityCfg("peg", body_names="peg"),
        },
    )

#-------------------------------------------------------------------------------------------
@configclass
class TerminationsCfg:

    # Terminates the episode if the object is lifted above a certain height
    time_out=DoneTerm(func=mdp.time_out,time_out=True)
    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("peg")}
    # )

#-------------------------------------------------------------------------------------------
# Manager-based env config is used to used to instantiate all of the 
# other managers and create the environment.
@configclass
class FrankaEnvCfg(ManagerBasedRLEnvCfg):

    # Scene configuration
    scene = FrankaTestSceneCfg(num_envs=2, env_spacing=2.0)


   
    
    # Manager instantiation
    observations = ObservationsCfg()
    commands = CommandsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()
    episode_length_s= 60
    # Decimation rate
    decimation = 1

    ee_frame_cfg = FRAME_MARKER_CFG.copy()
    ee_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_frame_cfg.prim_path = "/Visuals/FrameTransformer"
    scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=ee_frame_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_hand",
                name="ee_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )
   
    def __post__init__(self):
        self.viewer.eye = [4.5,0.0,6.0]
        self.viewer.lookat = [0.8, 0.0, 0.5]
        self.decimation =2 
        self.episode_length_s = 5
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        # self.sim.physx.friction_correlation_distance = 0.00625

        

#-------------------------------------------------------------------------------------------

