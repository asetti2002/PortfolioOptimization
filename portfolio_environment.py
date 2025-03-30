import gymnasium as gym
import pandas as pd
import numpy as np
import torch


def PortfolioEnv(gymnasium.Env):
    def __init__(self,
                 data, 
                 agent_type,
                 short_positions=False,
                 ):
        self.data = data
        self.agent_type = agent_type
        self.action_space = None
        # define action size
        self.current_weights = np.zeros(self.action_size)
        self.short_positions = short_positions

    def get_action_space(self):
        if self.agent_type == 'discrete':
            pass
        elif self.agent_type == 'continuous':
            pass

    def step(self, action):
        # Implement the step function
        self.current_weights = self.new_weights

        if self.agent_type == 'discrete':
            self.new_weights = np.zeros(self.action_size)
            self.new_weights[action] = 1
        elif self.agent_type == 'continuous':
            if self.short_positions:
                self.new_weights = torch.tensor(action) - torch.tensor(action).mean()
            else:
                self.new_weights = torch.tensor(action) / torch.tensor(action).sum()

        
            
    def reset(self):
        # Implement the reset function
        pass
        
        


def load_data(fp):
    

if __name__ == '__main__':
    