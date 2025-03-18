
from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.assets import Articulation, DeformableObject, RigidObject
from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

def reset_peg(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    sample_cfg: SceneEntityCfg = SceneEntityCfg("sample"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg")
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This funciton resets the peg position adter the sample has been reset.
    
    *Basically takes the sample position and adds a known offset to it to get the peg position*

    Args:
        env: The environment object.
        env_ids: The environment ids to reset.
        sample_cfg: The configuration of the sample asset.
        peg_cfg: The configuration of the peg asset.
    """
    # extract the used quantities (to enable type-hinting)
    sample: RigidObject = env.scene[sample_cfg.name]
    peg: RigidObject = env.scene[peg_cfg.name]

    # get default root state of sample
    sample_pos_w = sample.data.root_pos_w[:, :3]
    sample_quat_w = sample.data.root_quat_w[:, :4]
    print(sample_quat_w)
    pos_offset = torch.tensor([0.000, 0.00, 0.05], device=env.device)
    quat_offset = torch.tensor([-0.293, 0.7070,0, 0], device=env.device)
    length = sample_quat_w.shape[0]
    zero_rot = torch.zeros([length,4], device=env.device)
    #compute new position
    positions = torch.add(sample_pos_w , pos_offset) 
    orientations = torch.add(sample_quat_w, quat_offset)
    velocities = torch.zeros([1,6], device=env.device)

    # set into the physics simulation
        # pose = torch.cat([positions, orientations], dim=-1)
    peg.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    peg.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)