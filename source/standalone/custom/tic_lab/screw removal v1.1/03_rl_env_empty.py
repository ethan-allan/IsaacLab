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

#Custom package imports
from screw_env_cfg import FrankaTestSceneCfg
import mdp as mdp



#-------------------------------------------------------------------------------------------
def distance_calc():
    return 1


#-------------------------------------------------------------------------------------------
# Configuration classes 
# Manager-based base env only supports the following configuration classes:
# 1. ActionsCfg
# 2. ObservationsCfg
# 3. EventCfg
# 4. ManagerBasedEnvCfg


@configclass
class RewardsCfg:
    # Need to come up with basic reward function, will maybe steal this from franka reach
    # Distance to point makes the most sense, just need to work out how to get there
    
    # end_effector_position_tracking = RewTerm(
    #     func=,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    # )

    end_eff_null = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},)

    # Add contact forces at some point

    
    pass

#-------------------------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(3.14, 3.14),  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
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
            self.enable_corruption = True
            self.concatenate_terms = True
    
    #Create instance of policy
    policy: PolicyCfg = PolicyCfg()

#-------------------------------------------------------------------------------------------
@configclass
class ActionsCfg:

    # Assigns action as joint effort tensor
    # Provides the asset name (Franka), joint names (all joints except gripper)and scale
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", 
                                            joint_names=["panda_joint.*"], 
                                            scale=1.0)
    


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
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )
#-------------------------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out=DoneTerm(func=mdp.time_out,time_out=True)
    

#-------------------------------------------------------------------------------------------
# Manager-based env config is used to used to instantiate all of the 
# other managers and create the environment.
@configclass
class FrankaEnvCfg(ManagerBasedRLEnvCfg):

    # Scene configuration
    scene = FrankaTestSceneCfg(num_envs=16, env_spacing=2.0)

    # Manager instantiation
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards= RewardsCfg()
    commands= CommandsCfg()
    terminations= TerminationsCfg()
    episode_length_s= 12
    # Decimation rate
    decimation = 1

    def __post__init__(self):
        self.viewer.eye = [4.5,0.0,6.0]
        self.viewer.lookat = [0.8, 0.0, 0.5]
        self.decimation = 1
        self.sim.dt = 0.005


#-------------------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------------------
def main():

    # Create the environment
    env_cfg=FrankaEnvCfg()

    # Set the number of environments to spawn
    env_cfg.scene.num_envs=16

    env = ManagerBasedRLEnv(cfg= env_cfg)

    # Simulation loop
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
            obs, _  ,_,_,_= env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

