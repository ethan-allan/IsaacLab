import gymnasium as gym

from . import agents


#Simple model , only rewarded linearly for moving towards the target
gym.register(
    id="Isaac-Peg-Removal-Franka-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.peg_removal:FrankaEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
