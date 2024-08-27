import gymnasium as gym


def make_env(env_config):
    env = gym.make('CarRacing-v2')
    return env 