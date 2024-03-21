#sudo /etc/init.d/docker start
#export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

import os
import time
import yaml
import json
import argparse
import gym
import math
import matplotlib.pyplot as plt
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.logger import configure


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
        
        
if __name__ == "__main__":
    log_dir = "/mnt/c/py/logs/"
    os.makedirs(log_dir, exist_ok=True)
    logger0 = configure(log_dir, ["stdout", "csv", "tensorboard"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True,
                        help="Configuration file")
    opt = parser.parse_args()
    print(opt)

    # Read the cfg file
    yaml_file = open(opt.cfgFile)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    time_dep_seed = int((time.time() - int(time.time() - 0.5)) * 1000)
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"],
                                params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")
    tensor_board_folder = os.path.join(base_path, params["folders"]["parent_dir"],
                                       params["settings"]["game_id"],
                                       params["folders"]["model_name"], "tb")

    os.makedirs(model_folder, exist_ok=True)

    # Settings
    settings = params["settings"]

    # Wrappers Settings
    wrappers_settings = params["wrappers_settings"]
    

    # Create environment
    env, num_envs = make_sb3_env(params["settings"]["game_id"],
                                 settings, wrappers_settings, seed=time_dep_seed)
    print("Activated {} environment(s)".format(num_envs))

    print("Observation space =", env.observation_space)
    print("Act_space =", env.action_space)

    # Custom Wrappers for Custom Reward Values
    #env = SuperbarWrapper(env)
    observation = env.reset()
    #env.show_obs(observation)
    # Policy param
    policy_kwargs = params["policy_kwargs"]

    # PPO settings
    ppo_settings = params["ppo_settings"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    learning_rate = linear_schedule(ppo_settings["learning_rate"][0],
                                    ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(ppo_settings["clip_range"][0],
                                 ppo_settings["clip_range"][1])
    clip_range_vf = clip_range
    batch_size = ppo_settings["batch_size"]
    n_epochs = ppo_settings["n_epochs"]
    n_steps = ppo_settings["n_steps"]

    if model_checkpoint == "0M":
        # Initialize the agent
        agent = PPO("MultiInputPolicy", env, verbose=1,
                    gamma=gamma, batch_size=batch_size,
                    n_epochs=n_epochs, n_steps=n_steps,
                    learning_rate=learning_rate, clip_range=clip_range,
                    clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                    tensorboard_log=tensor_board_folder)
    else:
        # Load the trained agent
        print(os.path.join(model_folder, model_checkpoint))
        agent = PPO.load(os.path.join(model_folder, model_checkpoint), env=env,
                         gamma=gamma, learning_rate=learning_rate,
                         clip_range=clip_range, clip_range_vf=clip_range_vf,
                         policy_kwargs=policy_kwargs,
                         tensorboard_log=tensor_board_folder)


    
    agent.set_logger(logger0)
    
    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Create the callback: autosave every USER DEF steps
    autosave_freq = ppo_settings["autosave_freq"]
    auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs,
                                  save_path=os.path.join(model_folder,
                                                         model_checkpoint + "_"))
    
    #callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # Train the agent
    time_steps = ppo_settings["time_steps"]
    agent.learn(total_timesteps=time_steps, callback=auto_save_callback)
    #agent.learn(total_timesteps=time_steps, callback=TensorboardCallback())
    # Helper from the library
    #results_plotter.plot_results(
    #    [log_dir], 1e5, results_plotter.X_TIMESTEPS, "TD3 LunarLander"
    #)
    
    # Save the agent
    new_model_checkpoint = str(int(model_checkpoint[:-1]) + time_steps) + "M"
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)
    
    total = 0
    total_damage_dealt = 0
    total_damage_taken = 0
    total_stage_complete = 0
    total_round_win = 0
    total_round_done = 0
    for i in range(10):
        observation = env.reset()
        
        cumulative_reward = 0
        damage_dealt = 0
        damage_taken = 0
        stage_complete = 0
        round_win = 0
        round_done = 0
        
        while True:
            env.render()
            
            action, _state = agent.predict(observation, deterministic=False)
            pre_observation = observation
            observation, reward, done, info = env.step(action)
            #print(action, _state)
            #env.show_obs(observation)
            cumulative_reward += reward
            #if (abs(reward) > 0.01):
                #print("Cumulative reward =", cumulative_reward)
            if info[0]['round_done'] or info[0]['stage_done']:
                print(info)
                print(pre_observation['P1_ownHealth'],pre_observation['P1_oppHealth'])
                damage_dealt += 1-pre_observation['P1_oppHealth']
                damage_taken += 1-pre_observation['P1_ownHealth']
                if pre_observation['P1_oppHealth']<=pre_observation['P1_ownHealth']:
                    damage_dealt += pre_observation['P1_oppHealth']
                    round_win += 1
                else:
                    damage_taken += pre_observation['P1_ownHealth']
                round_done += 1 
                if info[0]['stage_done']:
                    stage_complete += 1
            if done:
                #print(info,observation)
                observation = env.reset()
                total += cumulative_reward
                total_stage_complete += stage_complete
                total_round_win += round_win
                total_damage_dealt += damage_dealt
                total_damage_taken += damage_taken
                total_round_done += round_done
                break

    print("Total Reward =", total)
    print("Stages Completed =", total_stage_complete)
    print("Rounds Win =", total_round_win)
    print("Rounds Completed =", total_round_done)
    print("Damage Dealt =", total_damage_dealt)
    print("Damage Taken =", total_damage_taken)
    #print(action,_state,observation,info)
    # Close the environment
    env.close()
    
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()