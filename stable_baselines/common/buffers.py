import numpy as np


class BaseBuffer(object):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: (int) Max number of element in the buffer
    :param obs_dim: (int) Dimension of the observation
    :param action_dim: (int) Dimension of the action space
    :param n_envs: (int) Number of parallel environments
    """
    def __init__(self, buffer_size, obs_dim, action_dim, n_envs=1):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
        self.n_envs = n_envs

    def size(self):
        """
        :return: (int) The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs):
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size):
        """
        :param batch_size: (int) Number of element to sample
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        """
        :param batch_inds: (np.ndarray)
        :return: ([np.ndarray])
        """
        raise NotImplementedError()


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: (int) Max number of element in the buffer
    :param obs_dim: (int) Dimension of the observation
    :param action_dim: (int) Dimension of the action space
    :param n_envs: (int) Number of parallel environments
    """
    def __init__(self, buffer_size, obs_dim, action_dim, n_envs=1):
        super(ReplayBuffer, self).__init__(buffer_size, obs_dim, action_dim, n_envs=n_envs)

        assert n_envs == 1
        self.observations = np.zeros((self.buffer_size, self.n_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, self.obs_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds):
        return (self.observations[batch_inds, 0, :],
                self.actions[batch_inds, 0, :],
                self.next_observations[batch_inds, 0, :],
                self.dones[batch_inds],
                self.rewards[batch_inds])


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: (int) Max number of element in the buffer
    :param obs_dim: (int) Dimension of the observation
    :param action_dim: (int) Dimension of the action space
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """
    def __init__(self, buffer_size, obs_dim, action_dim,
                 gae_lambda=1, gamma=0.99, n_envs=1):
        super(RolloutBuffer, self).__init__(buffer_size, obs_dim, action_dim, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_value, dones=False, use_gae=True):
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and advantage (A(s) = R - V(S)).

        :param last_value: (tf.Tensor)
        :param dones: ([bool])
        :param use_gae: (bool) Whether to use Generalized Advantage Estimation
            or normal advantage for advantage computation.
        """
        if use_gae:
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = np.array(1.0 - dones)
                    next_value = last_value.numpy().flatten()
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.values[step + 1]
                delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            self.returns = self.advantages + self.values
        else:
            # Discounted return with value bootstrap
            # Note: this is equivalent to GAE computation
            # with gae_lambda = 1.0
            last_return = 0.0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = np.array(1.0 - dones)
                    next_value = last_value.numpy().flatten()
                    last_return = self.rewards[step] + next_non_terminal * next_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    last_return = self.rewards[step] + self.gamma * last_return * next_non_terminal
                self.returns[step] = last_return
            self.advantages = self.returns - self.values

    def add(self, obs, action, reward, done, value, log_prob):
        """
        :param obs: (np.ndarray) Observation
        :param action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param value: (np.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (np.Tensor) log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.numpy().flatten().copy()
        self.log_probs[self.pos] = log_prob.numpy().copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size=None):
        assert self.full
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['observations', 'actions', 'values',
                           'log_probs', 'advantages', 'returns']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        return (self.observations[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten())

    @staticmethod
    def swap_and_flatten(tensor):
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param tensor: (np.ndarray)
        :return: (np.ndarray)
        """
        shape = tensor.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return tensor.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
