import pytest
import numpy as np

from stable_baselines import TD3, PPO
from stable_baselines.common.noise import NormalActionNoise

action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))


def test_td3():
    model = TD3('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]), seed=0,
                learning_starts=100, verbose=1, create_eval_env=True, action_noise=action_noise)
    model.learn(total_timesteps=10000, eval_freq=5000)
    # model.save("test_save")
    # model.load("test_save")
    # os.remove("test_save.zip")

@pytest.mark.parametrize("model_class", [PPO])
@pytest.mark.parametrize("env_id", ['CartPole-v1', 'Pendulum-v0'])
def test_onpolicy(model_class, env_id):
    model = model_class('MlpPolicy', env_id, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)
    # model.save("test_save")
    # model.load("test_save")
    # os.remove("test_save.zip")
