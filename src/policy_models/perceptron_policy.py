import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import os

class Perceptron_Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims, action_space_dims, config):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()
        self.eps = 1e-6  # small number for mathematical stability
        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def model_action_2_world_action(self, action):
        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action["means"][0] + self.eps, action["stddevs"][0] + self.eps)
        action = distrib.sample()
        # action = (torch.tensor(1) if action > .5 else torch.tensor(0)) # IF SCALAR (Discrete)
        prob = distrib.log_prob(action)
        action = action.numpy()
        
        return action, prob

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        action, prob = self.model_action_2_world_action({"means": action_means, "stddevs": action_stddevs})
        return  action, prob
    

    def save(self, saving_path):
        path = f"{saving_path}/models"
        os.mkdir(path)
        torch.save(self.shared_net.state_dict(), f"{path}/shared_net.pt")
        torch.save(self.policy_mean_net.state_dict(), f"{path}/policy_mean_net.pt")
        torch.save(self.policy_stddev_net.state_dict(), f"{path}/policy_stddev_net.pt")