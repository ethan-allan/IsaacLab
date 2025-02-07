import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-IK-Franka-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test_script:FrankaEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)