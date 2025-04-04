from os import truncate
import random
import gymnasium as gym
import pandas as pd
import numpy as np
import torch
from .utils import get_weights_asTensors
from gymnasium.spaces import Box, Discrete


class PortfolioEnv(gym.Env):
    def __init__(self,
                 data, 
                 agent_type,
                 short_positions=False,
                 continuous_weights=False,
                 allow_short_positions=True,
                 rebalance_every=1,
                 slippage=0.0,
                 transaction_cost=0.0,
                 ):
        self.data = data
        self.agent_type = agent_type

        self.allow_short_positions = allow_short_positions


        self.action_space = self.get_action_space()
        # define action size
        self.action_size = 0 #TODO

        self.current_weights = np.zeros(self.action_size)
        self.short_positions = short_positions

        self.available_dates = None # Need to fix this later

        self.rebalancing_dates = None
        self.current_rebalancing_date = None
        self.next_rebalancing_date = None

        self.rebalance_every = rebalance_every
        self.returns_sell = None
        self.returns_buy = None
        self.returns_hold = None
        self.continuous_weights = continuous_weights

        self.trajectory_bootstrapping = True # param
        self.trajectory_returns = []

        # need to set the values for these
        self.slippage = slippage
        self.transaction_cost = transaction_cost

        self.reset()

    def get_action_space(self):
        n_instruments = 1 # change this #TODO
        if self.agent_type == 'discrete':
            return Discrete(n_instruments)
        elif self.agent_type == 'continuous':
            return Box(-np.ones(n_instruments) * self.allow_short_positions, np.ones(n_instruments))
        else:
            raise Exception('This is not a valid agent type')
    

    def get_observation_space(self):
        if self.render_mode == 'vector':
            

    def return_obs_frame_as_tensor(self, obs_frame) -> torch.Tensor:
        # Take repeated indicators and reshape that
        # Tensors are in the shape of Channels x Height x Width -> instruments x lookback x indicators
        n_channels = len(self.indicator_instrument_names)
        global_tensor = torch.tile(torch.Tensor(obs_frame.loc[:, self.global_columns].values), [n_channels, 1, 1])

        indicators_tensor = torch.Tensor(
            [obs_frame[i].loc[:, self.indicator_names].values for i in self.indicator_instrument_names])

        if not 0 in global_tensor.size():
            return torch.concat([indicators_tensor, global_tensor], 1)
        else:
            return indicators_tensor

    def calculate_reward(self, returns):
        # sumf of the log returns
        return torch.sum(torch.log(1+returns))
    

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

        effective_rebalancing_date = self.available_dates[self.available_dates.index(self.current_rebalancing_date) + 1]

        r_sell = torch.Tensor(self.returns_sell.loc[[effective_rebalancing_date]].values).squeeze()
        r_buy = torch.Tensor(self.returns_buy.loc[[effective_rebalancing_date]].values).squeeze()
        r_hold = torch.Tensor(self.returns_hold.loc[[effective_rebalancing_date]].values).squeeze()
        
        return_frame = self.returns_hold.loc[effective_rebalancing_date:self.next_rebalancing_date, :]
        R_hold = torch.Tensor(return_frame.values)
        

        idx_lookback = max(0, self.available_dates.index(self.next_rebalancing_date) - self.observation_frame_lookback)
        observation_frame = self.df_observations[self.available_dates[idx_lookback]:self.next_rebalancing_date]
        observation_frame = self.expand_observation_frame(observation_frame)

        info = dict()
        info['indices'] = observation_frame.index.tolist()
        info['features'] = observation_frame.columns.tolist()


        # observations = None 
        if self.render_mode == 'tile':
            observations = torch.Tensor(observation_frame.values)
        elif self.render_mode == 'vector':
            observations = torch.Tensor(observation_frame.values.squeeze())
        else: # this a tensor
            observations = self.return_obs_frame_as_tensor(observation_frame)

        self.current_trajectory_len += self.rebalance_every
        truncated = False
        terminated = False


        if self.next_rebalancing_date == self.available_dates[-1]:
            terminated = True
        elif self.current_trajectory_len >= self.max_trajectory_len:
            truncated = True
        else:
            self.current_rebalancing_date = self.next_rebalancing_date
            self.next_rebalancing_date = self.available_dates[self.available_dates.index(self.current_rebalancing_date) + 1]


        # get the split weights and returns

        if self.continuous_weights:
            ret = torch.matmul(self.new_weight, R_hold.T)
            # from the first day return we ned to subract the transaction cost and slippage
            ret[0] -= (self.transaction_cost + self.slippage)
        else:
            # split the weight vector into different components (hold, buy, sell)
            # and then calculate the returns

            # imeplement this function later #TODO
            hold_weight, buy_weight, sell_weight = get_weights_asTensors(self.new_weights, self.current_weights)


            hold_return = torch.dot(hold_weight.squeeze(), r_hold)
            buy_return = torch.dot(buy_weight.squeeze(), r_buy)
            sell_return = torch.dot(sell_weight.squeeze(), r_sell)

            ret = torch.matmul(self.new_weight, R_hold.T)

            # fix the first day retunrns
            ret[0] += (sell_return - self.transaction_cost - self.slippage)
            ret[0] += (buy_return - self.transaction_cost - self.slippage)
            ret[0] += hold_return
        
        self.last_returns = ret

        df_r = pd.DataFrame(ret.numpy().squeeze(), index=pd.to_datetime(return_frame.index))
        self.trajectory_returns.append(df_r)

        if truncated or terminated:
            self.trajectory_returns = pd.concat(self.trajectory_returns)
            # self.trajectory_returns = self.trajectory_returns.reset_index(drop=True)
        r = self.calculate_reward(ret)

        return observations, r, terminated, truncated, info
    
            


    def reset(self):
        # Implement the reset function

        # takes a random date and prepares rebalancing dates based on the rebalance_every parameter

        self.current_rebalancing_date = random.choice(self.available_dates[:-self.rebalance_every-1])
        self.current_trajectory_len = 0
        self.trajectory_returns = []

        # TODO
        self.process_indicator_types()
        self.process_returns()


        if self.trajectory_bootstrapping:
            # randomly samples rebalancing dates from the remaining dates
            # then it sorts them 
            n_samples = 1 + self.max_trajectory_len // self.rebalance_every
            self.rebalancing_dates = np.random.choice(self.available_dates[self.available_dates.index(self.current_rebalancing_date):], n_samples)
            self.rebalancing_dates.sort()
        else:
            self.rebalancing_dates = self.rebalancing_dates[self.rebalancing_dates.index(self.current_rebalancing_date)::self.rebalance_every]
        
        self.next_rebalancing_date = self.rebalancing_dates[self.rebalancing_dates.index(self.current_rebalancing_date) + 1]


        observation_frame = self.df_observations[self.available_dates[idx_lookback]:self.current_rebalancing_date]
        observation_frame = self.expand_observation_frame(observation_frame)


        info = dict()
        info['indices'] = observation_frame.index.tolist()
        info['features'] = observation_frame.columns.tolist()


        # observations = None 
        if self.render_mode == 'tile':
            observations = torch.Tensor(observation_frame.values)
        elif self.render_mode == 'vector':
            observations = torch.Tensor(observation_frame.values.squeeze())
        else: # this a tensor
            observations = self.return_obs_frame_as_tensor(observation_frame)

        return observations, info


