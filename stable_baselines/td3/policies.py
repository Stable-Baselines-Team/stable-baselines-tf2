import tensorflow as tf
from tensorflow.keras.models import Sequential

from stable_baselines.common.policies import BasePolicy, register_policy, create_mlp


class Actor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param obs_dim: (int) Dimension of the observation
    :param action_dim: (int) Dimension of the action space
    :param net_arch: ([int]) Network architecture
    :param activation_fn: (str or tf.activation) Activation function
    """
    def __init__(self, obs_dim, action_dim, net_arch, activation_fn=tf.nn.relu):
        super(Actor, self).__init__(None, None)

        actor_net = create_mlp(obs_dim, action_dim, net_arch, activation_fn, squash_out=True)
        self.mu = Sequential(actor_net)
        self.mu.build()

    @tf.function
    def call(self, obs):
        return self.mu(obs)


class Critic(BasePolicy):
    """
    Critic network for TD3,
    in fact it represents the action-state value function (Q-value function)

    :param obs_dim: (int) Dimension of the observation
    :param action_dim: (int) Dimension of the action space
    :param net_arch: ([int]) Network architecture
    :param activation_fn: (nn.Module) Activation function
    """
    def __init__(self, obs_dim, action_dim,
                 net_arch, activation_fn=tf.nn.relu):
        super(Critic, self).__init__(None, None)

        q1_net = create_mlp(obs_dim + action_dim, 1, net_arch, activation_fn)
        self.q1_net = Sequential(q1_net)

        q2_net = create_mlp(obs_dim + action_dim, 1, net_arch, activation_fn)
        self.q2_net = Sequential(q2_net)

        self.q_networks = [self.q1_net, self.q2_net]

        for q_net in self.q_networks:
            q_net.build()

    @tf.function
    def call(self, obs, action):
        qvalue_input = tf.concat([obs, action], axis=1)
        return [q_net(qvalue_input) for q_net in self.q_networks]

    @tf.function
    def q1_forward(self, obs, action):
        return self.q_networks[0](tf.concat([obs, action], axis=1))


class TD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param learning_rate: (callable) Learning rate schedule (could be constant)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param activation_fn: (str or tf.nn.activation) Activation function
    """
    def __init__(self, observation_space, action_space,
                 learning_rate, net_arch=None,
                 activation_fn=tf.nn.relu):
        super(TD3Policy, self).__init__(observation_space, action_space)

        # Default network architecture, from the original paper
        if net_arch is None:
            net_arch = [400, 300]

        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn
        }
        self.actor_kwargs = self.net_args.copy()

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self._build(learning_rate)

    def _build(self, learning_rate):
        self.actor = self.make_actor()
        self.actor_target = self.make_actor()
        self.actor_target.hard_update(self.actor)
        self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate(1))

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.hard_update(self.critic)
        self.critic.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate(1))

    def make_actor(self):
        return Actor(**self.actor_kwargs)

    def make_critic(self):
        return Critic(**self.net_args)

    @tf.function
    def call(self, obs):
        return self.actor(obs)


MlpPolicy = TD3Policy

register_policy("MlpPolicy", MlpPolicy)
