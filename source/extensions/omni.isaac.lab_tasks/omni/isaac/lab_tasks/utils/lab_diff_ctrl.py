# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""
import argparse
from math import sqrt
from omni.isaac.lab.app import AppLauncher
import torch 
import numpy as np

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.assets import ArticulationCfg, Articulation
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils import configclass

@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on startup
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
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

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
class lab_env():
    def __init__(self):
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        self.setup_lab()
        self.setup_robot()
        self.setup_camera()
        return
    def setup_lab(self):
        """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
        # Ground-plane
        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

        # spawn distant light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))      #Nice lighting, good for demos
        light_cfg.func("/World/Light", light_cfg)

        #Creates a prim to store everything in
        prim_utils.create_prim

        #Sets up the table as a static asset (fixed but collisions are enabled)
        cfg_table=sim_utils.CuboidCfg(
            size=(0.91,1.5,0.84),                                                           #Size: measurements taken from room
            collision_props=sim_utils.CollisionPropertiesCfg(),                             #Enables collisions
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)))     #Color: grey
        
 
    
    
        cfg_desk = sim_utils.CuboidCfg(
            size=(0.6,1.22,0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
            )
        cfg_pole = sim_utils.CylinderCfg(
            radius=0.0225,
            height=2.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
            )
        cfg_plinth = sim_utils.CylinderCfg(
            radius=0.2,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
            )
        cfg_cupboard1 = sim_utils.CuboidCfg(
            size=(0.6,0.72,1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        )
        cfg_cupboard2 = sim_utils.CuboidCfg(
            size=(1.2,0.72,2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ) 
        cfg_floor= sim_utils.CuboidCfg(
            size=(3.2,3.9,0.001),
            visual_material=sim_utils.PreviewSurfaceCfg( diffuse_color=(0.1, 0.1, 0.1))
            )
        cfg_baseplate=sim_utils.CuboidCfg(
            size=(0.24,0.24,0.02), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1))
            )
    
         # Spawning the objects
        cfg_table.func("/World/Objects/Table", cfg_table, translation=(cfg_table.size[0]/2+0.38, cfg_table.size[1]/2+0.42, cfg_table.size[2]/2))
        cfg_baseplate.func("/World/Objects/baseplate", cfg_baseplate, translation=(cfg_baseplate.size[0]/2+0.715, cfg_baseplate.size[1]/2+0.465, cfg_baseplate.size[2]/2+0.84))
        cfg_floor.func("/World/Objects/floor", cfg_floor, translation=(cfg_floor.size[0]/2, cfg_floor.size[1]/2,0))
        cfg_plinth.func("/World/Objects/Plinth", cfg_plinth, translation=(0.91+cfg_plinth.radius,2.21+cfg_plinth.radius , cfg_plinth.height/2))
        cfg_pole.func("/World/Objects/Pole", cfg_pole, translation=(cfg_pole.radius+0.34, cfg_pole.radius+0.27, cfg_pole.height/2))
        cfg_desk.func("/World/Objects/Desk", cfg_desk, translation=(cfg_desk.size[0]/2 + 2.6, cfg_desk.size[1]/2+0.4, cfg_desk.size[2]/2))
        cfg_cupboard1.func("/World/Objects/Cupboard1", cfg_cupboard1, translation=(1.4+cfg_cupboard1.size[0]/2, 3.17+cfg_cupboard1.size[1]/2, cfg_cupboard1.size[2]/2))
        cfg_cupboard2.func("/World/Objects/Cupboard2", cfg_cupboard2, translation=(2.0+cfg_cupboard2.size[0]/2, 3.17+cfg_cupboard2.size[1]/2, cfg_cupboard2.size[2]/2))


        


    def setup_camera(self):
        self.sim.set_camera_view([2.0, 0.0, 2.5], [0.71, 0.0, -0.71])

    def setup_robot(self):
        kr10_cfg = ArticulationCfg(
        prim_path="/World/Objects/KR10",
        spawn= sim_utils.UsdFileCfg(
            usd_path=f"source/extensions/omni.isaac.lab_assets/data/Robots/KR10/KR10.usd",
              activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 3.14,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
            },
        
        pos=(0.01+0.715, 0.01+0.465, 0.01+0.84), 
        rot=(0.71,0.0 , 0, 0.71),
    ),
    actuators={
        "KR10_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit=87,
            velocity_limit=124.6,
        ),
    },
    )
        self.kr10=Articulation(kr10_cfg)  

              # Create controller
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=self.sim.device) 

        robot_entity_cfg= SceneEntityCfg
        return 
    
    

  

    
lab=lab_env()

lab.sim.reset()

while simulation_app.is_running():
    lab.sim.step()
    pass

simulation_app.close()