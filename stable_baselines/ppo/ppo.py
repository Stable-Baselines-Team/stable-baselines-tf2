import os
import time

import gym
from gym import spaces
import tensorflow as tf
import numpy as np

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.buffers import RolloutBuffer
from stable_baselines.common.utils import explained_variance, get_schedule_fn
from stable_baselines.common import logger
from stable_baselines.ppo.policies import PPOPolicy


class PPO(BaseRLModel):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI spinningup (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: (int) Minibatch size
    :param n_epochs: (int) Number of epoch when optimizing the surrogate loss
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: (float or callable) Clipping parameter, it can be a function of the current progress
        (from 1 to 0).
    :param clip_range_vf: (float or callable) Clipping parameter for the value function,
        it can be a function of the current progress (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param target_kl: (float) Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param seed: (int) Seed for the pseudo random generators
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """
    def __init__(self, policy, env, learning_rate=3e-4,
                 n_steps=2048, batch_size=64, n_epochs=10,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 target_kl=None, tensorboard_log=None, create_eval_env=False,
                 policy_kwargs=None, verbose=0, seed=0,
                 _init_setup_model=True):

        super(PPO, self).__init__(policy, env, PPOPolicy, policy_kwargs=policy_kwargs,
                                  verbose=verbose, create_eval_env=create_eval_env, support_multi_env=True, seed=seed)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.tb_writer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self._setup_learning_rate()
        # TODO: preprocessing: one hot vector for obs discrete
        state_dim = self.observation_space.shape[0]
        if isinstance(self.action_space, spaces.Box):
            # Action is a 1D vector
            action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, spaces.Discrete):
            # Action is a scalar
            action_dim = 1

        # TODO: different seed for each env when n_envs > 1
        if self.n_envs == 1:
            self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, state_dim, action_dim,
                                            gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.n_envs)
        self.policy = self.policy_class(self.observation_space, self.action_space,
                                        self.learning_rate, **self.policy_kwargs)

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        clipped_actions = self.policy.actor_forward(np.array(observation).reshape(1, -1), deterministic=deterministic)
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)
        return clipped_actions

    def collect_rollouts(self, env, rollout_buffer, n_rollout_steps=256, callback=None,
                         obs=None):

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:
            actions, values, log_probs = self.policy.call(obs)
            actions = actions.numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self._update_info_buffer(infos)
            n_steps += 1
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(obs, actions, rewards, dones, values, log_probs)
            obs = new_obs

        rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        return obs

    def policy_loss(self, advantage, log_prob, old_log_prob, clip_range):
        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = tf.exp(log_prob - old_log_prob)
        # clipped surrogate loss
        policy_loss_1 = advantage * ratio
        policy_loss_2 = advantage * tf.clip_by_value(ratio, 1 - clip_range, 1 + clip_range)
        return - tf.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))

    def value_loss(self, values, old_values, return_batch, clip_range_vf):
        if clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = old_values + tf.clip_by_value(values - old_values, -clip_range_vf, clip_range_vf)
        # Value loss using the TD(gae_lambda) target
        return tf.keras.losses.MSE(return_batch, values_pred)

    def train(self, gradient_steps, batch_size=64):
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress)
        else:
            clip_range_vf = None

        for gradient_step in range(gradient_steps):
            approx_kl_divs = []
            # Sample replay buffer
            for replay_data in self.rollout_buffer.get(batch_size):
                # Unpack
                obs, action, old_values, old_log_prob, advantage, return_batch = replay_data

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action for float to long
                    action = action.astype(np.int64).flatten()

                with tf.GradientTape() as tape:
                    tape.watch(self.policy.trainable_variables)
                    values, log_prob, entropy = self.policy.evaluate_actions(obs, action)
                    # Flatten
                    values = tf.reshape(values, [-1])

                    policy_loss = self.policy_loss(advantage, log_prob, old_log_prob, clip_range)

                    value_loss = self.value_loss(values, old_values, return_batch, clip_range_vf)

                    # Entropy loss favor exploration
                    entropy_loss = -tf.reduce_mean(entropy)

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                # Clip grad norm
                # gradients = tf.clip_by_norm(gradients, self.max_grad_norm)
                self.policy.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
                approx_kl_divs.append(tf.reduce_mean(old_log_prob - log_prob).numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print("Early stopping at step {} due to reaching max kl: {:.2f}".format(gradient_step,
                                                                                        np.mean(approx_kl_divs)))
                break

        explained_var = explained_variance(self.rollout_buffer.returns.flatten(),
                                           self.rollout_buffer.values.flatten())

        logger.logkv("clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.logkv("clip_range_vf", clip_range_vf)

        logger.logkv("explained_variance", explained_var)
        # TODO: gather stats for the entropy and other losses?
        logger.logkv("entropy", entropy.numpy().mean())
        logger.logkv("policy_loss", policy_loss.numpy())
        logger.logkv("value_loss", value_loss.numpy())
        if hasattr(self.policy, 'log_std'):
            logger.logkv("std", tf.exp(self.policy.log_std).numpy().mean())

    def learn(self, total_timesteps, callback=None, log_interval=1,
              eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="PPO", reset_num_timesteps=True):

        timesteps_since_eval, iteration, evaluations, obs, eval_env = self._setup_learn(eval_env)

        if self.tensorboard_log is not None:
            self.tb_writer = tf.summary.create_file_writer(os.path.join(self.tensorboard_log, tb_log_name))

        while self.num_timesteps < total_timesteps:

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            obs = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps,
                                        obs=obs)
            iteration += 1
            self.num_timesteps += self.n_steps * self.n_envs
            timesteps_since_eval += self.n_steps * self.n_envs
            self._update_current_progress(self.num_timesteps, total_timesteps)

            # Display training infos
            if self.verbose >= 1 and log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.logkv("iterations", iteration)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.logkv('ep_rew_mean', self.safe_mean([ep_info['r'] for ep_info in self.ep_info_buffer]))
                    logger.logkv('ep_len_mean', self.safe_mean([ep_info['l'] for ep_info in self.ep_info_buffer]))
                logger.logkv("fps", fps)
                logger.logkv('time_elapsed', int(time.time() - self.start_time))
                logger.logkv("total timesteps", self.num_timesteps)
                logger.dumpkvs()

            self.train(self.n_epochs, batch_size=self.batch_size)

            # Evaluate the agent
            timesteps_since_eval = self._eval_policy(eval_freq, eval_env, n_eval_episodes,
                                                     timesteps_since_eval, deterministic=True)
            # For tensorboard integration
            # if self.tb_writer is not None:
            #     with self.tb_writer.as_default():
            #         tf.summary.scalar('Eval/reward', mean_reward, self.num_timesteps)


        return self
