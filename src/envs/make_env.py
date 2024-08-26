import gymnasium as gym


def make_env(env_config):
    env = gym.make('Pendulum-v1', g=9.81)
    return env 