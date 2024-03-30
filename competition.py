import os
import yaml
import json
import argparse
from diambra.arena import Roles, SpaceTypes, load_settings_flat_dict
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from stable_baselines3 import PPO

"""This is an example agent based on stable baselines 3.

Usage:
diambra run python stable_baselines3/agent.py --cfgFile $PWD/stable_baselines3/cfg_files/doapp/sr6_128x4_das_nc.yaml --trainedModel "model_name"
"""

def main(cfg_file, trained_model, test=False):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])
    settings.role = Roles.P1
    settings.step_ratio = 3
    settings.stepRatio = 3
    settings.frame_shape = (128, 128, 1)
    settings.difficulty = 8
    settings.action_space = SpaceTypes.MULTI_DISCRETE

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])
    wrappers_settings.normalize_reward = False

    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, no_vec=True)
    print("Activated {} environment(s)".format(num_envs))

    # Policy param
    policy_kwargs = params["policy_kwargs"]

    # PPO settings
    ppo_settings = params["ppo_settings"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    clip_range_vf = clip_range
    batch_size = ppo_settings["batch_size"]
    n_epochs = ppo_settings["n_epochs"]
    n_steps = ppo_settings["n_steps"]

    # Load the trained agent
    model_path = os.path.join(model_folder, trained_model)
    agent = PPO.load("/sources/agent.zip", env=env,
                         gamma=gamma, learning_rate=learning_rate, clip_range=clip_range,
                         clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs)

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)
 
    obs, info = env.reset()

    for i in range(1):
      while True:
        action, _ = agent.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action.tolist())

        if terminated or truncated:
            obs, info = env.reset()
            if info["env_done"] or test is True:
                break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--trainedModel", type=str, default="model", help="Model checkpoint")
    parser.add_argument("--test", type=int, default=0, help="Test mode")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile, opt.trainedModel, bool(opt.test))

