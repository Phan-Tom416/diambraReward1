import math
import random
import numpy as np
import gym
import logging
from diambra.arena.env_settings import WrappersSettings

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, no_op_max=6):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be first action (0).
        :param env: (Gym Environment) the environment to wrap
        :param no_op_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.no_op_max = no_op_max
        self.override_num_no_ops = None

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_no_ops is not None:
            no_ops = self.override_num_no_ops
        else:
            no_ops = random.randint(1, self.no_op_max + 1)
        assert no_ops > 0
        obs = None
        no_op_action = [0, 0, 0, 0]
        if isinstance(self.action_space, gym.spaces.Discrete):
            no_op_action = 0
        for _ in range(no_ops):
            obs, _, done, _ = self.env.step(no_op_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class StickyActionsEnv(gym.Wrapper):
    def __init__(self, env, sticky_actions):
        """
        Apply sticky actions
        :param env: (Gym Environment) the environment to wrap
        :param sticky_actions: (int) number of steps
               during which the same action is sent
        """
        gym.Wrapper.__init__(self, env)
        self.sticky_actions = sticky_actions

        assert self.env.env_settings.step_ratio == 1, "sticky_actions can "\
                                                      "be activated only "\
                                                      "when stepRatio is "\
                                                      "set equal to 1"

    def step(self, action):

        rew = 0.0

        for _ in range(self.sticky_actions):

            obs, rew_step, done, info = self.env.step(action)
            rew += rew_step
            if info["round_done"] is True:
                break

        return obs, rew, done, info


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        clips the reward to {+1, 0, -1} by its sign.
        :param env: (Gym Environment) the environment
        """
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """
        Bin reward to {+1, 0, -1} by its sign.
        :param reward: (float)
        """
        return np.sign(reward)


class NormalizeRewardEnv(gym.RewardWrapper):
    def __init__(self, env, reward_normalization_factor):
        """
        Normalize the reward dividing it by the product of
        rewardNormalizationFactor multiplied by
        the maximum character health variadtion (max - min).
        :param env: (Gym Environment) the environment
        :param rewardNormalizationFactor: multiplication factor
        """
        gym.RewardWrapper.__init__(self, env)
        self.env.reward_normalization_value = reward_normalization_factor * self.env.max_delta_health

    def reward(self, reward):
        """
        Nomralize reward dividing by reward normalization factor*max_delta_health
        :param reward: (float)
        """
        return float(reward) / float(self.env.reward_normalization_value)

class SuperbarWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # Check if the stun bar key is available
        assert (
            "P1_ownSuperCount" in self.env.observation_space.keys()
        ), "The 'P1_ownSuperCount' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())

        # Initialization

        print("Applying Stunned Reward Wrapper")

    # Step the environment
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        supercount_bonus = 0
        # Penalize if super bar is at max
        current_supercount = obs['P1_ownSuperCount']
        if current_supercount >= 1:
            supercount_bouns = -0.003
        else:
            supercount_bonus = 0
            
        # Add the bonus to the reward
        reward += supercount_bonus

        return obs, reward, done, info

    # Reset the environment
    def reset(self):
        obs = self.env.reset()
        return obs
        
class BlockWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # Check if the player's health value is available
        assert (
            "P1_ownHealth" in self.env.observation_space.keys()
        ), "The 'P1_ownHealth' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())

        # Initialization
        self.previous_health = 0

        print("Applying Block Reward Wrapper")

    # Step the environment
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Bonus if block successful
        health_delta = obs['P1_ownHealth'] - self.previous_health
        if health_delta < 0 and health_delta >= -0.01:
            block_bonus = 0.5
            print("Block Successful: " + str(health_delta))
        else:
            block_bonus = 0
            
        # Add the bonus to the reward
        reward += block_bonus

        return obs, reward, done, info

    # Reset the environment
    def reset(self):
        obs = self.env.reset()
        self.previous_health = obs['P1_ownHealth']
        return obs        

class SuperHitWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # Check if the player's health value is available
        assert (
            "P1_oppHealth" in self.env.observation_space.keys()
        ), "The 'P1_oppHealth' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())
        #assert (
        #    "P1_ownSuperCount" in self.env.observation_space.keys()
        #), "The 'P1_ownSuperCount' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())
        # Initialization
        self.previous_opp_health = 0
        self.frames_after_super = -1
        self.previous_supercount = 0
        self.super_damage = 0
        print("Applying Super Move Reward Wrapper")

    # Step the environment
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Penalize missed super moves (not including blocked supers)
        opp_health_delta = obs['P1_oppHealth'] - self.previous_opp_health
        supercount_delta = obs['P1_ownSuperCount'] - self.previous_supercount
        super_bonus = 0
        
        if opp_health_delta == 0 and self.frames_after_super >= 0:
            self.frames_after_super += 1
        elif opp_health_delta < 0 and self.frames_after_super >= 0:
            #super_bonus = 5
            self.super_damage += -opp_health_delta
            print(supercount_delta,opp_health_delta,self.frames_after_super)
            
        if supercount_delta < 0:
            self.frames_after_super = 0
            print(supercount_delta,"Super Move Activated")
        
        if self.frames_after_super >= 22 or (obs['P1_oppHealth'] < 0.007 and self.super_damage >= 0.005):
            self.frames_after_super = -1
            print("Super Damage Dealt: " + str(self.super_damage))
            if self.super_damage >= 0.1: #Super Hit
                super_bonus = 5
                
            elif self.super_damage >= 0.005: #Super Blocked
                super_bonus = 2
            else:
                super_bonus = -3.5
            self.super_damage = 0
            print("Bonus RV: " + str(super_bonus))
        # Add the bonus to the reward
        reward += super_bonus
            
        
        self.previous_opp_health = obs['P1_oppHealth']
        self.previous_supercount = obs['P1_ownSuperCount']
        
        return obs, reward, done, info

class OppSuperHitWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # Check if the player's health value is available
        assert (
            "P1_ownHealth" in self.env.observation_space.keys()
        ), "The 'P1_ownHealth' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())
        #assert (
        #    "P1_ownSuperCount" in self.env.observation_space.keys()
        #), "The 'P1_ownSuperCount' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())
        # Initialization
        self.previous_own_health = 0
        self.frames_after_super = -1
        self.previous_supercount = 0
        self.super_damage = 0
        print("Applying Super Move Reward Wrapper")

    # Step the environment
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Penalize missed super moves (not including blocked supers)
        own_health_delta = obs['P1_ownHealth'] - self.previous_own_health
        supercount_delta = obs['P1_oppSuperCount'] - self.previous_supercount
        super_bonus = 0
        
        if own_health_delta == 0 and self.frames_after_super >= 0:
            self.frames_after_super += 1
        elif own_health_delta < 0 and self.frames_after_super >= 0:
            #super_bonus = 5
            self.super_damage += -own_health_delta
            print(supercount_delta,own_health_delta,self.frames_after_super)
            
        if supercount_delta < 0:
            self.frames_after_super = 0
            print(supercount_delta,"Opponent Super Move Activated")
        
        if self.frames_after_super >= 30 or (obs['P1_ownHealth'] < 0.007 and self.super_damage >= 0.005):
            self.frames_after_super = -1
            print("Super Damage Taken: " + str(self.super_damage))
            if self.super_damage >= 0.03: #Super Hit
                super_bonus = -3
            elif self.super_damage >= 0.005: #Super Blocked
                super_bonus = 10
            else:
                super_bonus = 10
            self.super_damage = 0
        
        # Add the bonus to the reward
        reward += super_bonus
            
        
        self.previous_own_health = obs['P1_ownHealth']
        self.previous_supercount = obs['P1_oppSuperCount']
        
        return obs, reward, done, info

    # Reset the environment
    def reset(self):
        obs = self.env.reset()
        self.previous_own_health = obs['P1_ownHealth']
        self.previous_supercount = obs['P1_oppSuperCount']
        return obs  
 

class NoHitWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # Check if the player's health value is available
        assert (
            "P1_oppHealth" in self.env.observation_space.keys()
        ), "The 'P1_oppHealth' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())
        assert (
            "P1_ownHealth" in self.env.observation_space.keys()
        ), "The 'P1_ownHealth' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())
        #assert (
        #    "P1_ownSuperCount" in self.env.observation_space.keys()
        #), "The 'P1_ownSuperCount' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())
        # Initialization
        self.previous_opp_health = 0
        self.previous_own_health = 0
        self.frames_after_hit = 0
        
        print("Applying Super Move Reward Wrapper")

    # Step the environment
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Penalize missed super moves (not including blocked supers)
        opp_health_delta = obs['P1_oppHealth'] - self.previous_opp_health
        own_health_delta = obs['P1_ownHealth'] - self.previous_own_health
        #supercount_delta = obs['P1_ownSuperCount'] - self.previous_supercount
        no_hit_bonus = 0
        
        if opp_health_delta >= 0.0:# and own_health_delta == 0 : Only non-blocked hits resets frame counter
            self.frames_after_hit += 1
            if opp_health_delta < 0:
                print("Blocked: " + str(-opp_health_delta))
        else:
            self.frames_after_hit = 0
        #else:
        #    self.frames_after_hit = min(self.frames_after_hit,30)
            
        if obs['P1_oppHealth'] <= 0 or obs['P1_ownHealth'] <= 0:
            self.frames_after_hit = 0
        if self.frames_after_hit == 20 or self.frames_after_hit == 30 or self.frames_after_hit == 50:
            print("Frames no hit = " + str(self.frames_after_hit))
            
        if self.frames_after_hit >= 50:
            no_hit_bonus = -0.025
        elif self.frames_after_hit >= 30:
            no_hit_bonus = -0.015
        elif self.frames_after_hit >= 20:
            no_hit_bonus = -0.01
            
        
        # Add the bonus to the reward
        reward += no_hit_bonus
            
        
        self.previous_opp_health = obs['P1_oppHealth']
        self.previous_own_health = obs['P1_ownHealth']
        #self.previous_supercount = obs['P1_ownSuperCount']
        
        return obs, reward, done, info

    # Reset the environment
    def reset(self):
        obs = self.env.reset()
        self.previous_opp_health = obs['P1_oppHealth']
        self.previous_supercount = obs['P1_ownSuperCount']
        return obs   
class CustomStunWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # Check if the stun bar key is available
        assert (
            "P1_oppStunBar" in self.env.observation_space.keys()
        ), "The 'P1_oppStunBar' key must be present in add_obs to use this wrapper. Keys: {}".format(self.env.observation_space.keys())

        # Initialization
        self.previous_stunbar = 0

        print("Applying Stunned Reward Wrapper")

    # Step the environment
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Calculate bonus based on opponent's stun bar increase
        current_stunbar = obs['P1_oppStunBar']
        stunbar_increase = current_stunbar - self.previous_stunbar
        stunbar_bonus = 0 if stunbar_increase <= 0 else 0.1
        

        # Add the bonus to the reward
        reward += stunbar_bonus
        
        self.previous_stunbar = obs['P1_oppStunBar']
        
        return obs, reward, done, info

    # Reset the environment
    def reset(self):
        obs = self.env.reset()
        self.previous_stunbar = obs['P1_oppStunBar']
        return obs

class TimePenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        print("Applying Stunned Reward Wrapper")

    # Step the environment
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Add the bonus to the reward
        reward -= 0.008

        return obs, reward, done, info

    # Reset the environment
    def reset(self):
        obs = self.env.reset()
        return obs

class WinWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.cooldown = 0
        print("Applying Win Reward Wrapper")

    # Step the environment
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        current_opp_health = obs['P1_oppHealth']
        current_own_health = obs['P1_ownHealth']
        if current_opp_health <= 0 and self.cooldown <= 0:
            win_bonus = 10
            self.cooldown = 2400
        elif current_own_health <= 0 and self.cooldown <= 0:
            win_bonus = -10
            self.cooldown = 2400
        else:
            win_bonus = 0
        
        if self.cooldown > 0:
            self.cooldown -= 1
        # Add the bonus to the reward
        reward += win_bonus

        return obs, reward, done, info

    # Reset the environment
    def reset(self):
        obs = self.env.reset()
        return obs
                  
# Environment Wrapping (rewards normalization, resizing, grayscaling, etc)
def env_wrapping(env, wrappers_settings: WrappersSettings, hardcore: bool=False):
    """
    Typical standard environment wrappers
    :param env: (Gym Environment) the diambra environment
    :param no_op_max: (int) wrap the environment to perform
                    no_op_max no action steps at reset
    :param clipRewards: (bool) wrap the reward clipping wrapper
    :param rewardNormalization: (bool) if to activate reward noramlization
    :param rewardNormalizationFactor: (double) noramlization factor
                                      for reward normalization wrapper
    :param frameStack: (int) wrap the frame stacking wrapper
                       using #frameStack frames
    :param dilation (frame stacking): (int) stack one frame every
                                      #dilation frames, useful to assure
                                      action every step considering
                                      a dilated subset of previous frames
    :param actionsStack: (int) wrap the frame stacking wrapper
                         using #frameStack frames
    :param scale: (bool) wrap the scaling observation wrapper
    :param scaleMod: (int) them scaling method: 0->[0,1] 1->[-1,1]
    :return: (Gym Environment) the wrapped diambra environment
    """
    logger = logging.getLogger(__name__)

    if wrappers_settings.no_op_max > 0:
        env = NoopResetEnv(env, no_op_max=wrappers_settings.no_op_max)

    if wrappers_settings.sticky_actions > 1:
        env = StickyActionsEnv(env, sticky_actions=wrappers_settings.sticky_actions)

    if hardcore is True:
        from diambra.arena.wrappers.obs_wrapper_hardcore import WarpFrame,\
            WarpFrame3C, FrameStack, FrameStackDilated,\
            ScaledFloatObsNeg, ScaledFloatObs
    else:
        from diambra.arena.wrappers.obs_wrapper import WarpFrame, \
            WarpFrame3C, FrameStack, FrameStackDilated,\
            ActionsStack, ScaledFloatObsNeg, ScaledFloatObs, FlattenFilterDictObs

    if wrappers_settings.hwc_obs_resize[2] == 1:
        # Resizing observation from H x W x 3 to
        # hwObsResize[0] x hwObsResize[1] x 1
        env = WarpFrame(env, wrappers_settings.hwc_obs_resize)
    elif wrappers_settings.hwc_obs_resize[2] == 3:
        # Resizing observation from H x W x 3 to
        # hwObsResize[0] x hwObsResize[1] x hwObsResize[2]
        env = WarpFrame3C(env, wrappers_settings.hwc_obs_resize)

    # Normalize rewards
    if wrappers_settings.reward_normalization is True:
        env = NormalizeRewardEnv(env, wrappers_settings.reward_normalization_factor)

    # Clip rewards using sign function
    if wrappers_settings.clip_rewards is True:
        env = ClipRewardEnv(env)

    # Stack #frameStack frames together
    if wrappers_settings.frame_stack > 1:
        if wrappers_settings.dilation == 1:
            env = FrameStack(env, wrappers_settings.frame_stack)
        else:
            logger.debug("Using frame stacking with dilation = {}".format(wrappers_settings.dilation))
            env = FrameStackDilated(env, wrappers_settings.frame_stack, wrappers_settings.dilation)

    # Stack #actionsStack actions together
    if wrappers_settings.actions_stack > 1 and not hardcore:
        env = ActionsStack(env, wrappers_settings.actions_stack)

    # Scales observations normalizing them
    if wrappers_settings.scale is True:
        if wrappers_settings.scale_mod == 0:
            # Between 0.0 and 1.0
            if hardcore is False:
                env = ScaledFloatObs(env, wrappers_settings.exclude_image_scaling, wrappers_settings.process_discrete_binary)
            else:
                env = ScaledFloatObs(env)
        elif wrappers_settings.scale_mod == -1:
            # Between -1.0 and 1.0
            raise RuntimeError("Scaling between -1.0 and 1.0 currently not implemented")
            env = ScaledFloatObsNeg(env)
        else:
            raise ValueError("Scale mod must be either 0 or -1")

    if wrappers_settings.flatten is True:
        if hardcore is True:
            logger.warning("Dictionary observation flattening is valid only for not hardcore mode, skipping it.")
        else:
            env = FlattenFilterDictObs(env, wrappers_settings.filter_keys)
    
    if wrappers_settings.superbar_wrapper is True:
        env = SuperbarWrapper(env)
        
    if wrappers_settings.custom_stun_wrapper is True:
        env = CustomStunWrapper(env)
    
    if wrappers_settings.block_wrapper is True:
        env = BlockWrapper(env)
    
    if wrappers_settings.super_hit_wrapper is True:
        env = SuperHitWrapper(env)
    
    if wrappers_settings.opp_super_hit_wrapper is True:
        env = OppSuperHitWrapper(env)
        
    if wrappers_settings.time_penalty_wrapper is True:
        env = TimePenaltyWrapper(env)
        
    if wrappers_settings.win_wrapper is True:
        env = WinWrapper(env)
    
    if wrappers_settings.no_hit_wrapper is True:
        env = NoHitWrapper(env)
    return env
