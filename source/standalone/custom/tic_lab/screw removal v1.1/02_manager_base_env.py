# This script utilises the IsaacLab manager-based base environment tutorial
# Instead of creating the cartpole environment it creates the environment for screw removal
# 
# In its current state the environment is created using 00_screw_env_cfg.py and an action 
# and observations manager are used to interact with the Franka




#----------------------------------------------------------------------------------------------
# Generic IsaacLab startup code.
# Reads cmd line arguments and creates a simulation app instance with them.
# Currently this demo won't do anything with cmd line arguments but normally used for env configs.
#----------------------------------------------------------------------------------------------
import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
#---------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------------------

#Standard python imports
import math
import torch

#Isaac specific imports
import omni.isaac.lab.envs.mdp as mdp                                   
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.utils import configclass 
from omni.isaac.lab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg
)
#Custom package imports
from screw_env_cfg import FrankaTestSceneCfg
#-------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------
# Configuration classes 
# Manager-based base env only supports the following configuration classes:
# 1. ActionsCfg
# 2. ObservationsCfg
# 3. EventCfg
#-------------------------------------------------------------------------------------------
@configclass
class ActionsCfg:

    # Assigns action as joint effort tensor
    # Provides the asset name (Franka), joint names (all joints except gripper)and scale
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", 
                                            joint_names=["panda_joint.*"], 
                                            scale=1.0)
    
#  
# @configclass
# class CommandsCfg:
#     """Command terms for the MDP."""
#     ee_pose = mdp.UniformPoseCommandCfg(
#         asset_name="robot",
#         body_name="panda_hand",
#         resampling_time_range=(4.0, 4.0),
#         debug_vis=True,
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             pos_x=(0.35, 0.65),
#             pos_y=(-0.2, 0.2),
#             pos_z=(0.15, 0.5),
#             roll=(0.0, 0.0),
#             pitch=(3.14, 3.14),  # depends on end-effector axis
#             yaw=(-3.14, 3.14),
#         ),
#     )



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
        #pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    #Create instance of policy
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
  
    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    #Reset robot and screw position
    pass

@configclass
class FrankaEnvCfg(ManagerBasedEnvCfg):
    scene = FrankaTestSceneCfg(num_envs=2, env_spacing=2.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    decimation = 4

    def __post__init__(self):
        self.viewer.eye = [4.5,0.0,6.0]
        self.viewer.lookat = [0.8, 0.0, 0.5]
        self.decimation = 1
        self.sim.dt = 0.005

def main():
    env_cfg=FrankaEnvCfg()
    env_cfg.scene.num_envs=2

    #env_cfg.scene.num_envs=1
    env = ManagerBasedEnv(cfg= env_cfg)
    count =0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                count=0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")


            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

