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
        self.observations = np.zeros((self.buffer_size, self.n_envs, self.obs_dim))
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim))
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, self.obs_dim))
        self.rewards = np.zeros((self.buffer_size, self.n_envs))
        self.dones = np.zeros((self.buffer_size, self.n_envs))

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
        # TODO: remove casting, it won't work with discrete actions
        return (self.observations[batch_inds, 0, :].astype(np.float32),
                self.actions[batch_inds, 0, :].astype(np.float32),
                self.next_observations[batch_inds, 0, :].astype(np.float32),
                self.dones[batch_inds].astype(np.float32),
                self.rewards[batch_inds].astype(np.float32))
