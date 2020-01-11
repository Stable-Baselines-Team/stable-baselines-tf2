from itertools import zip_longest

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential


class BasePolicy(Model):
    """
    The base policy object

    :param observation_space: (Gym Space) The observation space of the environment
    :param action_space: (Gym Space) The action space of the environment
    """

    def __init__(self, observation_space, action_space):
        super(BasePolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def save(self, path):
        """
        Save model to a given location.

        :param path: (str)
        """
        raise NotImplementedError()

    def load(self, path):
        """
        Load saved model from path.

        :param path: (str)
        """
        raise NotImplementedError()

    @tf.function
    def soft_update(self, other_network, tau):
        other_variables = other_network.trainable_variables
        current_variables = self.trainable_variables

        for (current_var, other_var) in zip(current_variables, other_variables):
            current_var.assign((1. - tau) * current_var + tau * other_var)

    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)

    def call(self, x):
        raise NotImplementedError()


def create_mlp(input_dim, output_dim, net_arch,
               activation_fn=tf.nn.relu, squash_out=False):
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: (int) Dimension of the input vector
    :param output_dim: (int)
    :param net_arch: ([int]) Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: (tf.activations or str) The activation function
        to use after each layer.
    :param squash_out: (bool) Whether to squash the output using a Tanh
        activation function
    """
    modules = [layers.Flatten(input_shape=(input_dim,), dtype=tf.float32)]

    if len(net_arch) > 0:
        modules.append(layers.Dense(net_arch[0], activation=activation_fn))

    for idx in range(len(net_arch) - 1):
        modules.append(layers.Dense(net_arch[idx + 1], activation=activation_fn))

    if output_dim > 0:
        modules.append(layers.Dense(output_dim, activation=None))
    if squash_out:
        modules.append(layers.Activation(activation='tanh'))
    return modules


_policy_registry = dict()


def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]



def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]


def register_policy(name, policy):
    """
    returns the registed policy from the base type and name

    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError("Error: the policy {} is not of any known subclasses of BasePolicy!".format(policy))

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError("Error: the name {} is alreay registered for a different policy, will not override."
                         .format(name))
    _policy_registry[sub_class][name] = policy
