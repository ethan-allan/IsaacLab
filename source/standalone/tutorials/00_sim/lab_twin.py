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

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


def setup_lab():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    prim_utils.create_prim("/World/Objects", "Xform")
 
 
    cfg_desk = sim_utils.CuboidCfg(
        size=(0.6,1.22,0.8),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)))
    cfg_pole = sim_utils.CylinderCfg(
        radius=0.0225,
        height=2.0,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)))
    cfg_plinth = sim_utils.CylinderCfg(
        radius=0.2,
        height=1.0,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)))
    cfg_cupboard1 = sim_utils.CuboidCfg(
        size=(0.6,0.72,1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),

    )
    cfg_cupboard2 = sim_utils.CuboidCfg(
        size=(1.2,0.72,2.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),

    ) #Cupboard configuration fgg
    cfg_floor= sim_utils.CuboidCfg(size=(3.2,3.9,0.001),
        visual_material=sim_utils.PreviewSurfaceCfg( diffuse_color=(0.1, 0.1, 0.1)))
    cfg_baseplate=sim_utils.CuboidCfg(size=(0.24,0.24,0.02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)))
    kr10_cfg = sim_utils.UsdFileCfg(usd_path=f"source/extensions/omni.isaac.lab_assets/data/Robots/KR10/KR10.usd")
# here we spawn the objects
    cfg_baseplate.func("/World/Objects/baseplate", cfg_baseplate, translation=(cfg_baseplate.size[0]/2+0.715, cfg_baseplate.size[1]/2+0.465, cfg_baseplate.size[2]/2+0.84))
    cfg_floor.func("/World/Objects/floor", cfg_floor, translation=(cfg_floor.size[0]/2, cfg_floor.size[1]/2,0))
    cfg_plinth.func("/World/Objects/Plinth", cfg_plinth, translation=(0.91+cfg_plinth.radius,2.21+cfg_plinth.radius , cfg_plinth.height/2))
    cfg_pole.func("/World/Objects/Pole", cfg_pole, translation=(cfg_pole.radius+0.34, cfg_pole.radius+0.27, cfg_pole.height/2))
    cfg_desk.func("/World/Objects/Desk", cfg_desk, translation=(cfg_desk.size[0]/2 + 2.6, cfg_desk.size[1]/2+0.4, cfg_desk.size[2]/2))
    cfg_cupboard1.func("/World/Objects/Cupboard1", cfg_cupboard1, translation=(1.4+cfg_cupboard1.size[0]/2, 3.17+cfg_cupboard1.size[1]/2, cfg_cupboard1.size[2]/2))
    cfg_cupboard2.func("/World/Objects/Cupboard2", cfg_cupboard2, translation=(2.0+cfg_cupboard2.size[0]/2, 3.17+cfg_cupboard2.size[1]/2, cfg_cupboard2.size[2]/2))

    kr10_cfg.func("/World/Objects/KR10", kr10_cfg, translation=(cfg_baseplate.size[0]/2+0.715, cfg_baseplate.size[1]/2+0.465, cfg_baseplate.size[2]/2+0.84), orientation=(0.71,0.0 , 0, 0.71))

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [0.71, 0.0, -0.71])

    # Design scene by adding assets to it
    setup_lab()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
