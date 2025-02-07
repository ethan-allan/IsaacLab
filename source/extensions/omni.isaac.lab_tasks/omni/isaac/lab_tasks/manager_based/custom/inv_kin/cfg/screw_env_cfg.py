# Screw Removal Envionment Configuration
# This script sets up the environment for any screw removal demos. It should spawn in the robot, sample, screw etc.
# Future work: Implement different env configs depending on the demo to run (i.e. screw removal, peg removal, different robots etc.)


import omni.isaac.lab.sim as sim_utils                           # Provides the ground plane and dome light
from omni.isaac.lab.assets import  AssetBaseCfg                  # Provides base asset configuration class  (generic structure for storing assets)       
from omni.isaac.lab.scene import  InteractiveSceneCfg            # Provides interactive scene cfg (generic structure for scene cfg)
from omni.isaac.lab.utils import configclass                     # Config class decorator to override some features of python 3.7 dataclasses
                                                                 # Dataclasses require type annotation for memebres, this stops that. More info here: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.configclass 
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG        # Provides the Franka Panda robot configuration   


# Path to the custom assets folder (screw, sample, peg etc.)
#assets_folder = f"/home/ethanallan175/IsaacLab/source/standalone/custom/assets/"

# Standard practice in IsaacLab: all managers should be a config class
@configclass
class FrankaTestSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

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

    # Spawns in the panda. Currently set to the low stiffness version.
    robot =FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    
    
    

    
