import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
import tensorflow_probability as tfp
from gym import spaces


class Distribution(object):
    def __init__(self):
        super(Distribution, self).__init__()

    def log_prob(self, x):
        """
        returns the log likelihood

        :param x: (object) the taken action
        :return: (tf.Tensor) The log likelihood of the distribution
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns shannon's entropy of the probability

        :return: (tf.Tensor) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        returns a sample from the probabilty distribution

        :return: (tf.Tensor) the stochastic action
        """
        raise NotImplementedError


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix,
    for continuous actions.

    :param action_dim: (int)  Number of continuous actions
    """

    def __init__(self, action_dim):
        super(DiagGaussianDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim, log_std_init=0.0):
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: (int) Dimension og the last layer of the policy (before the action layer)
        :param log_std_init: (float) Initial value for the log standard deviation
        :return: (tf.keras.models.Sequential, tf.Variable)
        """
        mean_actions = Sequential(layers.Dense(self.action_dim, input_shape=(latent_dim,), activation=None))
        log_std = tf.Variable(tf.ones(self.action_dim) * log_std_init)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions, log_std, deterministic=False):
        """
        Create and sample for the distribution given its parameters (mean, std)

        :param mean_actions: (tf.Tensor)
        :param log_std: (tf.Tensor)
        :param deterministic: (bool)
        :return: (tf.Tensor)
        """
        action_std = tf.ones_like(mean_actions) * tf.exp(log_std)
        self.distribution = tfp.distributions.Normal(mean_actions, action_std)
        if deterministic:
            action = self.mode()
        else:
            action = self.sample()
        return action, self

    def mode(self):
        return self.distribution.mode()

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy()

    def log_prob_from_params(self, mean_actions, log_std):
        """
        Compute the log probabilty of taking an action
        given the distribution parameters.

        :param mean_actions: (tf.Tensor)
        :param log_std: (tf.Tensor)
        :return: (tf.Tensor, tf.Tensor)
        """
        action, _ = self.proba_distribution(mean_actions, log_std)
        log_prob = self.log_prob(action)
        return action, log_prob

    def log_prob(self, action):
        """
        Get the log probabilty of an action given a distribution.
        Note that you must call `proba_distribution()` method
        before.

        :param action: (tf.Tensor)
        :return: (tf.Tensor)
        """
        log_prob = self.distribution.log_prob(action)
        if len(log_prob.shape) > 1:
            log_prob = tf.reduce_sum(log_prob, axis=1)
        else:
            log_prob = tf.reduce_sum(log_prob)
        return log_prob


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: (int) Number of discrete actions
    """
    def __init__(self, action_dim):
        super(CategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim):
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilties using a softmax.

        :param latent_dim: (int) Dimension og the last layer of the policy (before the action layer)
        :return: (tf.keras.models.Sequential)
        """
        action_logits = layers.Dense(self.action_dim, input_shape=(latent_dim,), activation=None)
        return Sequential([action_logits])

    def proba_distribution(self, action_logits, deterministic=False):
        self.distribution = tfp.distributions.Categorical(logits=action_logits)
        if deterministic:
            action = self.mode()
        else:
            action = self.sample()
        return action, self

    def mode(self):
        return self.distribution.mode()

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy()

    def log_prob_from_params(self, action_logits):
        action, _ = self.proba_distribution(action_logits)
        log_prob = self.log_prob(action)
        return action, log_prob

    def log_prob(self, action):
        log_prob = self.distribution.log_prob(action)
        return log_prob


def make_proba_distribution(action_space, dist_kwargs=None):
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: (Gym Space) the input action space
    :param dist_kwargs: (dict) Keyword arguments to pass to the probabilty distribution
    :return: (Distribution) the approriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        return DiagGaussianDistribution(action_space.shape[0], **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n, **dist_kwargs)
    # elif isinstance(action_space, spaces.MultiDiscrete):
    #     return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    # elif isinstance(action_space, spaces.MultiBinary):
    #     return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError("Error: probability distribution, not implemented for action space of type {}."
                                  .format(type(action_space)) +
                                  " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary.")
