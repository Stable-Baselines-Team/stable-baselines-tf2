import time
from abc import ABC, abstractmethod
from collections import deque
import os
import io
import zipfile

import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common import logger
from stable_baselines.common.policies import get_policy_from_name
from stable_baselines.common.utils import set_random_seed, get_schedule_fn
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv, unwrap_vec_normalize, sync_envs_normalization
from stable_baselines.common.monitor import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.save_util import data_to_json, json_to_data


class BaseRLModel(ABC):
    """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: (BasePolicy) the base policy used by this method
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 debug
    :param support_multi_env: (bool) Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: (bool) When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: (int) Seed for the pseudo random generators
    """
    def __init__(self, policy, env, policy_base, policy_kwargs=None,
                 verbose=0, device='auto', support_multi_env=False,
                 create_eval_env=False, monitor_wrapper=True, seed=None):
        if isinstance(policy, str) and policy_base is not None:
            self.policy_class = get_policy_from_name(policy_base, policy)
        else:
            self.policy_class = policy

        self.env = env
        # get VecNormalize object if needed
        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self.num_timesteps = 0
        self.eval_env = None
        self.replay_buffer = None
        self.seed = seed
        self.action_noise = None

        # Track the training progress (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress = 1

        # Create and wrap the env if needed
        if env is not None:
            if isinstance(env, str):
                if create_eval_env:
                    eval_env = gym.make(env)
                    if monitor_wrapper:
                        eval_env = Monitor(eval_env, filename=None)
                    self.eval_env = DummyVecEnv([lambda: eval_env])
                if self.verbose >= 1:
                    print("Creating environment from the given name, wrapped in a DummyVecEnv.")

                env = gym.make(env)
                if monitor_wrapper:
                    env = Monitor(env, filename=None)
                env = DummyVecEnv([lambda: env])

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            if not isinstance(env, VecEnv):
                if self.verbose >= 1:
                    print("Wrapping the env in a DummyVecEnv.")
                env = DummyVecEnv([lambda: env])
            self.n_envs = env.num_envs
            self.env = env

            if not support_multi_env and self.n_envs > 1:
                raise ValueError("Error: the model does not support multiple envs requires a single vectorized"
                                 " environment.")

    def _get_eval_env(self, eval_env):
        """
        Return the environment that will be used for evaluation.

        :param eval_env: (gym.Env or VecEnv)
        :return: (VecEnv)
        """
        if eval_env is None:
            eval_env = self.eval_env

        if eval_env is not None:
            if not isinstance(eval_env, VecEnv):
                eval_env = DummyVecEnv([lambda: eval_env])
            assert eval_env.num_envs == 1
        return eval_env

    def scale_action(self, action):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: (np.ndarray)
        :return: (np.ndarray)
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _setup_learning_rate(self):
        """Transform to callable if needed."""
        self.learning_rate = get_schedule_fn(self.learning_rate)

    def _update_current_progress(self, num_timesteps, total_timesteps):
        """
        Compute current progress (from 1 to 0)

        :param num_timesteps: (int) current number of timesteps
        :param total_timesteps: (int)
        """
        self._current_progress = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers):
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress (from 1 to 0).

        :param optimizers: ([th.optim.Optimizer] or Optimizer) An optimizer
            or a list of optimizer.
        """
        # Log the current learning rate
        logger.logkv("learning_rate", self.learning_rate(self._current_progress))

        # if not isinstance(optimizers, list):
        #     optimizers = [optimizers]
        # for optimizer in optimizers:
        #     update_learning_rate(optimizer, self.learning_rate(self._current_progress))

    @staticmethod
    def safe_mean(arr):
        """
        Compute the mean of an array if there is at least one element.
        For empty array, return nan. It is used for logging only.

        :param arr: (np.ndarray)
        :return: (float)
        """
        return np.nan if len(arr) == 0 else np.mean(arr)

    def get_env(self):
        """
        returns the current environment (can be None if not defined)

        :return: (gym.Env) The current environment
        """
        return self.env

    @staticmethod
    def check_env(env, observation_space, action_space):
        """
        Checks the validity of the environment and returns if it is coherent
        Checked parameters:
         - observation_space
         - action_space
        :return: (bool) True if environment seems to be coherent
        """
        if observation_space != env.observation_space:
            return False
        if action_space != env.action_space:
            return False
        # return true if no check failed
        return True

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
         - observation_space
         - action_space

        :param env: (gym.Env) The environment for learning a policy
        """
        if self.check_env(env, self.observation_space, self.action_space) is False:
            raise ValueError("The given environment is not compatible with model: observation and action spaces do not match")
        # it must be coherent now
        # if it is not a VecEnv, make it a VecEnv
        if not isinstance(env, VecEnv):
            if self.verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])
        self.n_envs = env.num_envs
        self.env = env

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, val_interval=None):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """
        raise NotImplementedError()

    @abstractmethod
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run",
              eval_env=None, eval_freq=-1, n_eval_episodes=5, reset_num_timesteps=True):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param callback: (function (dict, dict)) -> boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :param eval_env: (gym.Env) Environment that will be used to evaluate the agent
        :param eval_freq: (int) Evaluate the agent every `eval_freq` timesteps (this may vary a little)
        :param n_eval_episodes: (int) Number of episode to evaluate the agent
        :return: (BaseRLModel) the trained model
        """
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        pass

    def set_random_seed(self, seed=None):
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed: (int)
        """
        if seed is None:
            return
        set_random_seed(seed)
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)
        if self.eval_env is not None:
            self.eval_env.seed(seed)

    def _setup_learn(self, eval_env):
        """
        Initialize different variables needed for training.

        :param eval_env: (gym.Env or VecEnv)
        :return: (int, int, [float], np.ndarray, VecEnv)
        """
        self.start_time = time.time()
        self.ep_info_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        timesteps_since_eval, episode_num = 0, 0
        evaluations = []

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)
        obs = self.env.reset()
        return timesteps_since_eval, episode_num, evaluations, obs, eval_env

    def _update_info_buffer(self, infos):
        """
        Retrieve reward and episode length and update the buffer
        if using Monitor wrapper.

        :param infos: ([dict])
        """
        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])

    def _eval_policy(self, eval_freq, eval_env, n_eval_episodes,
                     timesteps_since_eval, deterministic=True):
        """
        Evaluate the current policy on a test environment.

        :param eval_env: (gym.Env) Environment that will be used to evaluate the agent
        :param eval_freq: (int) Evaluate the agent every `eval_freq` timesteps (this may vary a little)
        :param n_eval_episodes: (int) Number of episode to evaluate the agent
        :parma timesteps_since_eval: (int) Number of timesteps since last evaluation
        :param deterministic: (bool) Whether to use deterministic or stochastic actions
        :return: (int) Number of timesteps since last evaluation
        """
        if 0 < eval_freq <= timesteps_since_eval and eval_env is not None:
            timesteps_since_eval %= eval_freq
            # Synchronise the normalization stats if needed
            sync_envs_normalization(self.env, eval_env)
            mean_reward, std_reward = evaluate_policy(self, eval_env, n_eval_episodes, deterministic=deterministic)
            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("FPS: {:.2f}".format(self.num_timesteps / (time.time() - self.start_time)))
        return timesteps_since_eval
