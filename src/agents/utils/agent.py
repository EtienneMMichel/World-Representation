from gymnasium.spaces import Discrete


class Agent:

    def __init__(self, observation_space, action_space, agent_config):
        
        self.obs_space_dims = observation_space.shape[0]

        if isinstance(action_space,Discrete):
            self.action_space_dims = 1
        else:
            self.action_space_dims = action_space.shape[0]
        

    