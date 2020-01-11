import time

import tensorflow as tf
import numpy as np

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common import logger
from stable_baselines.td3.policies import TD3Policy


class TD3(BaseRLModel):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param buffer_size: (int) size of the replay buffer
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gamma: (float) the discount factor
    :param batch_size: (int) Minibatch size for each gradient update
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param action_noise: (ActionNoise) the action noise type. Cf common.noise for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param seed: (int) Seed for the pseudo random generators
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, buffer_size=int(1e6), learning_rate=1e-3,
                 policy_delay=2, learning_starts=100, gamma=0.99, batch_size=100,
                 train_freq=-1, gradient_steps=-1, n_episodes_rollout=1,
                 tau=0.005, action_noise=None, target_policy_noise=0.2, target_noise_clip=0.5,
                 tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0,
                 seed=None, _init_setup_model=True):

        super(TD3, self).__init__(policy, env, TD3Policy, policy_kwargs, verbose,
                                  create_eval_env=create_eval_env, seed=seed)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.n_episodes_rollout = n_episodes_rollout
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.action_noise = action_noise
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self._setup_learning_rate()
        obs_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(self.buffer_size, obs_dim, action_dim)
        self.policy = self.policy_class(self.observation_space, self.action_space,
                                        self.learning_rate, **self.policy_kwargs)
        self._create_aliases()

    def _create_aliases(self):
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def predict(self, observation, state=None, mask=None, deterministic=True):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        return self.unscale_action(self.actor(np.array(observation).reshape(1, -1)).numpy())

    @tf.function
    def critic_loss(self, obs, action, next_obs, done, reward):
        # Select action according to policy and add clipped noise
        noise = tf.random.normal(shape=action.shape) * self.target_policy_noise
        noise = tf.clip_by_value(noise, -self.target_noise_clip, self.target_noise_clip)
        next_action = tf.clip_by_value(self.actor_target(next_obs) + noise, -1., 1.)

        # Compute the target Q value
        target_q1, target_q2 = self.critic_target(next_obs, next_action)
        target_q = tf.minimum(target_q1, target_q2)
        target_q = reward + tf.stop_gradient((1 - done) * self.gamma * target_q)

        # Get current Q estimates
        current_q1, current_q2 = self.critic(obs, action)

        # Compute critic loss
        return tf.keras.losses.MSE(current_q1, target_q) + tf.keras.losses.MSE(current_q2, target_q)

    @tf.function
    def actor_loss(self, obs):
        return - tf.reduce_mean(self.critic.q1_forward(obs, self.actor(obs)))

    @tf.function
    def update_targets(self):
        self.critic_target.soft_update(self.critic, self.tau)
        self.actor_target.soft_update(self.actor, self.tau)

    def train(self, gradient_steps, batch_size=100, policy_delay=2):
        # self._update_learning_rate()

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            obs, action, next_obs, done, reward = self.replay_buffer.sample(batch_size)

            with tf.GradientTape() as critic_tape:
                critic_tape.watch(self.critic.trainable_variables)
                critic_loss = self.critic_loss(obs, action, next_obs, done, reward)

            # Optimize the critic
            grads_critic = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))


            # Delayed policy updates
            if gradient_step % policy_delay == 0:
                with tf.GradientTape() as actor_tape:
                    actor_tape.watch(self.actor.trainable_variables)
                    # Compute actor loss
                    actor_loss = self.actor_loss(obs)

                # Optimize the actor
                grads_actor = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))

                # Update the frozen target models
                self.update_targets()


    def learn(self, total_timesteps, callback=None, log_interval=4,
              eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="TD3", reset_num_timesteps=True):

        timesteps_since_eval, episode_num, evaluations, obs, eval_env = self._setup_learn(eval_env)

        while self.num_timesteps < total_timesteps:

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            rollout = self.collect_rollouts(self.env, n_episodes=self.n_episodes_rollout,
                                            n_steps=self.train_freq, action_noise=self.action_noise,
                                            deterministic=False, callback=None,
                                            learning_starts=self.learning_starts,
                                            num_timesteps=self.num_timesteps,
                                            replay_buffer=self.replay_buffer,
                                            obs=obs, episode_num=episode_num,
                                            log_interval=log_interval)
            # Unpack
            episode_reward, episode_timesteps, n_episodes, obs = rollout

            episode_num += n_episodes
            self.num_timesteps += episode_timesteps
            timesteps_since_eval += episode_timesteps
            self._update_current_progress(self.num_timesteps, total_timesteps)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:

                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else episode_timesteps
                self.train(gradient_steps, batch_size=self.batch_size, policy_delay=self.policy_delay)

            # Evaluate the agent
            timesteps_since_eval = self._eval_policy(eval_freq, eval_env, n_eval_episodes,
                                                     timesteps_since_eval, deterministic=True)

        return self


    def collect_rollouts(self, env, n_episodes=1, n_steps=-1, action_noise=None,
                         deterministic=False, callback=None,
                         learning_starts=0, num_timesteps=0,
                         replay_buffer=None, obs=None,
                         episode_num=0, log_interval=None):
        """
        Collect rollout using the current policy (and possibly fill the replay buffer)
        TODO: move this method to off-policy base class.

        :param env: (VecEnv)
        :param n_episodes: (int)
        :param n_steps: (int)
        :param action_noise: (ActionNoise)
        :param deterministic: (bool)
        :param callback: (callable)
        :param learning_starts: (int)
        :param num_timesteps: (int)
        :param replay_buffer: (ReplayBuffer)
        :param obs: (np.ndarray)
        :param episode_num: (int)
        :param log_interval: (int)
        """
        episode_rewards = []
        total_timesteps = []
        total_steps, total_episodes = 0, 0
        assert isinstance(env, VecEnv)
        assert env.num_envs == 1

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            # Reset environment: not needed for VecEnv
            # obs = env.reset()
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                # Select action randomly or according to policy
                if num_timesteps < learning_starts:
                    # Warmup phase
                    unscaled_action = np.array([self.action_space.sample()])
                else:
                    unscaled_action = self.predict(obs)

                # Rescale the action from [low, high] to [-1, 1]
                scaled_action = self.scale_action(unscaled_action)

                # Add noise to the action (improve exploration)
                if action_noise is not None:
                    scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(self.unscale_action(scaled_action))

                done_bool = [float(done[0])]
                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos)

                # Store data in replay buffer
                if replay_buffer is not None:
                    replay_buffer.add(obs, new_obs, scaled_action, reward, done_bool)

                obs = new_obs

                num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1
                if 0 < n_steps <= total_steps:
                    break

            if done:
                total_episodes += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)
                if action_noise is not None:
                    action_noise.reset()

                # Display training infos
                if self.verbose >= 1 and log_interval is not None and (
                            episode_num + total_episodes) % log_interval == 0:
                    fps = int(num_timesteps / (time.time() - self.start_time))
                    logger.logkv("episodes", episode_num + total_episodes)
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        logger.logkv('ep_rew_mean', self.safe_mean([ep_info['r'] for ep_info in self.ep_info_buffer]))
                        logger.logkv('ep_len_mean', self.safe_mean([ep_info['l'] for ep_info in self.ep_info_buffer]))
                    # logger.logkv("n_updates", n_updates)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - self.start_time))
                    logger.logkv("total timesteps", num_timesteps)
                    logger.dumpkvs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        return mean_reward, total_steps, total_episodes, obs
