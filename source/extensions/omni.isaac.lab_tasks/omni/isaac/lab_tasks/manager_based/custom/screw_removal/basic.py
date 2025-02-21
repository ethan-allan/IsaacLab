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
try:
    from . import mdp
except (ImportError, ModuleNotFoundError) as e:
    import mdp
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
    """Configuration for a cart-pole scene."""

    # Spawns the ground plane under the world prim
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
                            spawn=sim_utils.GroundPlaneCfg())

    # Spawns the assembled screw assembly under the assembly prim
    
   
        
    
    peg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/peg",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 5.3, 0.15], rot=[0, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/home/ethanallan175/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/custom/screw_removal/peg_v15.usd",
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
                scale=(0.001 , 0.001, 0.001),
            ),  
        )
    
    object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
    
 
    # peg: ArticulationCfg = ArticulationCfg(
    #     prim_path="/World/envs/env_.*/peg",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"/home/ethanallan175/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/custom/screw_removal/bolt.usd",
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             max_depenetration_velocity=5.0,
    #             linear_damping=0.0,
    #             angular_damping=0.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=3666.0,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=192,
    #             solver_velocity_iteration_count=1,
    #             max_contact_impulse=1e32,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
    #         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    #     ),
    #     actuators={},
    # )
    # cone_cfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Cone",
    #     spawn=sim_utils.ConeCfg(radius=0.1,
    #                             height=0.2,
    #                             rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #                             mass_props=sim_utils.MassPropertiesCfg(),
    #                             collision_props=sim_utils.CollisionPropertiesCfg(),
    #                             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
    #                             ),
    #                             init_state=RigidObjectCfg.InitialStateCfg(),
    
    #cone_object = RigidObject(cfg=cone_cfg)
    
    # object = RigidObjectCfg(
    #         prim_path="{ENV_REGEX_NS}/Object",
    #         init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
    #         spawn=sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #             scale=(0.8, 0.8, 0.8),
    #             rigid_props=RigidBodyPropertiesCfg(
    #                 solver_position_iteration_count=16,
    #                 solver_velocity_iteration_count=1,
    #                 max_angular_velocity=1000.0,
    #                 max_linear_velocity=1000.0,
    #                 max_depenetration_velocity=5.0,
    #                 disable_gravity=False,
    #             ),
    #         ),
    #     )
   # usd_path = f"{ASSET_DIR}/factory_bolt_m16.usd"

    # held_asset = ArticulationCfg(
    #     prim_path="/World/envs/env_.*/HeldAsset",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ASSET_DIR}/factory_bolt_m16.usd",
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=True,
    #             max_depenetration_velocity=5.0,
    #             linear_damping=0.0,
    #             angular_damping=0.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=3666.0,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=192,
    #             solver_velocity_iteration_count=1,
    #             max_contact_impulse=1e32,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
    #         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    #     ),
    #     actuators={},
    # )




    sample = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/sample",                                                                # ENV_REGEX_NS allows the part to be replicated for each environment instane
                        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.3, 0.3, 0.0], rot=[0.707, 0 ,0.0, 0.707]),           # Pose and rotation in format [x,y,z] and [w,a,b,c] respectively           # 
                        spawn=sim_utils.UsdFileCfg(usd_path=f"/home/ethanallan175/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/custom/screw_removal/sample.usd" ))

    
    # Spawns dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Spawns in the panda. Currently set to the low stiffness version.
    robot =FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    
   

#-------------------------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # Need to come up with basic reward function, will maybe steal this from franka reach
    # Distance to point makes the most sense, just need to work out how to get there
    
    pass

#-------------------------------------------------------------------------------------------

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass
#-------------------------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    # Environment observation specifications
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        #object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        #target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
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
    arm_action =DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
    
    gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )


#------------------------------------------------------------------------------------------- 
@configclass
class EventCfg:
#Used to define all events that can occur in the environment and what to do
#Currently just using reset but there are also startup and interval

    # Resets all the joints to some random position
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("peg", body_names="peg"),
    #     },
    # )

#-------------------------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out=DoneTerm(func=mdp.time_out,time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("peg")}
    )

#-------------------------------------------------------------------------------------------
# Manager-based env config is used to used to instantiate all of the 
# other managers and create the environment.
@configclass
class FrankaEnvCfg(ManagerBasedRLEnvCfg):

    # Scene configuration
    scene = FrankaTestSceneCfg(num_envs=22, env_spacing=2.0)



    
    
    # Manager instantiation
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()
    episode_length_s= 100
    # Decimation rate
    decimation = 1

    

    
    
    def __post__init__(self):
        self.viewer.eye = [4.5,0.0,6.0]
        self.viewer.lookat = [0.8, 0.0, 0.5]
        self.sim.render_interval = self.decimation

        self.sim.dt = 1.0 /6000.0

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        

#-------------------------------------------------------------------------------------------

