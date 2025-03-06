#Standalone environment run

import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from peg_removal import FrankaEnvCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv
import torch

def main():

    # Create the environment
    env_cfg=FrankaEnvCfg()

    # Set the number of environments to spawn
    env_cfg.scene.num_envs=2

    env = ManagerBasedRLEnv(cfg= env_cfg)

    # Simulation loop
    count =0
    while simulation_app.is_running():
        avg_rews=[]
        avg_rew=0
        with torch.inference_mode():
            # if count %  200== 0:
            #     count=0
            #     #env.reset()
                
            #     avg_rews.append(avg_rew)
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")
            #     avg_rew=0


            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rews, _, _, _= env.step(env.action_manager.action)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            avg_rew=torch.mean(rews)
            print("Average reward: ", avg_rew)
            # update counter
            # count += 1
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()