import random
import gymnasium
import pandas as pd
import numpy as np
import torch
from .utils import compute_OHLC_returns, create_return_matrices, get_weights_asTensors
from gymnasium.spaces import Box, Discrete

class PortfolioEnv(gymnasium.Env):
    def __init__(self,
                 data_ohlc, 
                 agent_type,
                 short_positions=False,
                 continuous_weights=False,
                 allow_short_positions=True,
                 rebalance_every=1,
                 slippage=0.0001,
                 transaction_cost=0.0001,
                 render_mode='tile',
                 max_trajectory_len=252,
                 observation_frame_lookback=0,
                 trajectory_bootstrapping=False,
                 ):
        # Set parameters first
        self.max_trajectory_len = max_trajectory_len
        self.observation_frame_lookback = observation_frame_lookback  # Set early!
        self.agent_type = agent_type
        self.render_mode = render_mode
        self.allow_short_positions = allow_short_positions

        # Process your OHLC data
        self.data_ohlc = data_ohlc.dropna()
        # Define available dates from the OHLC data index
        self.available_dates = [pd.to_datetime(x) for x in self.data_ohlc.sort_index().index.tolist()]


        # Determine action size based on the number of instruments (using 'Close' columns)
        close_cols = [col for col in self.data_ohlc.columns if col[1] == 'Close']
        self.action_size = len(close_cols)

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.current_weights = np.zeros(self.action_size)
        self.short_positions = short_positions

        self.rebalancing_dates = None
        self.current_rebalancing_date = None
        self.next_rebalancing_date = None
        self.rebalance_every = rebalance_every

        # Pre-define returns placeholders.
        self.returns_sell = None
        self.returns_buy = None
        self.returns_hold = None
        self.continuous_weights = continuous_weights

        self.trajectory_bootstrapping = trajectory_bootstrapping
        self.trajectory_returns = []

        self.slippage = slippage
        self.transaction_cost = transaction_cost

        # Initialize new_weights (for step())
        self.new_weights = torch.zeros(self.action_size, dtype=torch.float32)

        self.preprocess_returns()
        self.reset()

    def get_action_space(self):
        low_bound = -np.ones(self.action_size) if self.allow_short_positions else np.zeros(self.action_size)
        return Box(low=low_bound, high=np.ones(self.action_size))

    def get_observation_space(self):
        lower_bound = np.tile(self.data_ohlc.min(axis=0).values,
                                (1 + self.observation_frame_lookback, 1))
        upper_bound = np.tile(self.data_ohlc.max(axis=0).values,
                                (1 + self.observation_frame_lookback, 1))
        return Box(low=lower_bound,
                    high=upper_bound,
                    shape=[1 + self.observation_frame_lookback, self.data_ohlc.shape[1]])

    def calculate_reward(self, returns):
        # def calculate_reward(self, returns):
        safe_returns = torch.clamp(returns, min=-0.999)  # ensure 1 + returns > 0
        return torch.sum(torch.log1p(safe_returns))

    
    def preprocess_returns(self):
        instruments = set([col[0] for col in self.data_ohlc.columns])
        r_h_dict, r_b_dict, r_s_dict = {}, {}, {}
        for inst in instruments:
            df_inst = self.data_ohlc[inst]
            ret_df = compute_OHLC_returns(df_inst)
            r_h_dict[inst] = ret_df['r_h'].values
            r_b_dict[inst] = ret_df['r_b'].values
            r_s_dict[inst] = ret_df['r_s'].values
        r_h = pd.DataFrame(r_h_dict, index=self.data_ohlc.index).dropna()
        r_b = pd.DataFrame(r_b_dict, index=self.data_ohlc.index).dropna()
        r_s = pd.DataFrame(r_s_dict, index=self.data_ohlc.index).dropna()
        self.returns_hold = r_h
        self.returns_buy = r_b
        self.returns_sell = r_s

    def step(self, action):
        self.current_weights = self.new_weights

        # elif self.agent_type == 'continuous':
        if self.short_positions:
            self.new_weights = torch.tensor(action, dtype=torch.float32) - torch.tensor(action, dtype=torch.float32).mean()
        else:
            # Use softmax to ensure positive weights summing to 1
            action_tensor = torch.tensor(action, dtype=torch.float32)
            self.new_weights = torch.nn.functional.softmax(action_tensor, dim=0)
        # elif self.agent_type == 'continuous':
        #     if self.short_positions:
        #         self.new_weights = torch.tensor(action, dtype=torch.float32) - torch.tensor(action, dtype=torch.float32).mean()
        #     else:
        #         self.new_weights = torch.tensor(action, dtype=torch.float32) / torch.tensor(action, dtype=torch.float32).sum()

        effective_rebalancing_date = self.available_dates[
            self.available_dates.index(self.current_rebalancing_date) + 1]
        
        r_sell = torch.tensor(self.returns_sell.loc[[effective_rebalancing_date]].values, dtype=torch.float32).squeeze()
        r_buy = torch.tensor(self.returns_buy.loc[[effective_rebalancing_date]].values, dtype=torch.float32).squeeze()
        r_hold = torch.tensor(self.returns_hold.loc[[effective_rebalancing_date]].values, dtype=torch.float32).squeeze()

        return_frame = self.returns_hold.loc[effective_rebalancing_date:self.next_rebalancing_date, :]
        R_hold = torch.tensor(return_frame.values, dtype=torch.float32)

        idx_lookback = max(0, self.available_dates.index(self.next_rebalancing_date) - self.observation_frame_lookback)
        observation_frame = self.data_ohlc[self.available_dates[idx_lookback]:self.next_rebalancing_date]
        if observation_frame.shape[0] < self.observation_frame_lookback + 1:
            n_needed = self.observation_frame_lookback + 1 - observation_frame.shape[0]
            observation_frame = pd.concat([observation_frame,
                                        pd.concat([observation_frame.iloc[[-1]]]*n_needed)])
        info = {'indices': observation_frame.index.tolist(),
                'features': observation_frame.columns.tolist()}

        # Instead of converting to torch tensor, return the observation as a NumPy array
        if self.render_mode == 'tile':
            observations = observation_frame.values.astype(np.float32)
        elif self.render_mode == 'vector':
            observations = observation_frame.values.squeeze().astype(np.float32)
        else:
            observations = observation_frame.values.astype(np.float32)

        self.current_trajectory_len += self.rebalance_every
        truncated = False
        terminated = False
        if self.next_rebalancing_date == self.available_dates[-1]:
            terminated = True
        elif self.current_trajectory_len >= self.max_trajectory_len:
            truncated = True
        else:
            self.current_rebalancing_date = self.next_rebalancing_date
            self.next_rebalancing_date = self.available_dates[
                self.available_dates.index(self.current_rebalancing_date) + 1]

        # if self.continuous_weights:
        #     ret = torch.matmul(self.new_weights, R_hold.T)
        #     ret[0] -= (self.transaction_cost + self.slippage)


        if self.continuous_weights:
            ret = torch.matmul(self.new_weights, R_hold.T)
            r = self.calculate_reward(ret)
            cost = self.transaction_cost + self.slippage
            # Ensure cost is within valid range to avoid log(0)
            if cost >= 1.0:
                cost = 0.999  # Prevent invalid log
            r += torch.log(torch.tensor(1.0 - cost, dtype=torch.float32))
            reward = r.item()
        else:
            hold_weight, buy_weight, sell_weight = get_weights_asTensors(self.new_weights, self.current_weights)
            hold_return = torch.dot(hold_weight.squeeze(), r_hold)
            buy_return = torch.dot(buy_weight.squeeze(), r_buy)
            sell_return = torch.dot(sell_weight.squeeze(), r_sell)
            ret = torch.matmul(self.new_weights, R_hold.T)
            ret[0] += (sell_return - self.transaction_cost - self.slippage)
            ret[0] += (buy_return - self.transaction_cost - self.slippage)
            ret[0] += hold_return

        self.last_returns = ret
        df_r = pd.Series(ret.numpy().squeeze(), index=pd.to_datetime(return_frame.index))
        self.trajectory_returns.append(df_r)
        if truncated or terminated:
            self.trajectory_returns = pd.concat(self.trajectory_returns)
        r = self.calculate_reward(ret)
        reward = r.item()  # Convert to scalar
        return observations, reward, terminated, truncated, info


    def reset(self, seed: int = 5106, options: dict = None) -> tuple:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        # self.current_rebalancing_date = random.choice(self.available_dates[:-self.rebalance_every - 1])
        self.current_rebalancing_date = self.available_dates[0]
        self.current_trajectory_len = 0
        self.trajectory_returns = []
        self.preprocess_returns()
        if self.trajectory_bootstrapping:
            n_samples = 1 + self.max_trajectory_len // self.rebalance_every
            self.rebalancing_dates = np.random.choice(
                self.available_dates[self.available_dates.index(self.current_rebalancing_date):], n_samples)
            self.rebalancing_dates.sort()
        else:
            start_idx = self.available_dates.index(self.current_rebalancing_date)
            self.rebalancing_dates = self.available_dates[start_idx::self.rebalance_every]
        self.next_rebalancing_date = self.rebalancing_dates[
            self.rebalancing_dates.index(self.current_rebalancing_date) + 1]
        idx_lookback = max(0, self.available_dates.index(self.current_rebalancing_date) - self.observation_frame_lookback)
        start_date = pd.to_datetime(self.available_dates[idx_lookback])
        end_date = pd.to_datetime(self.current_rebalancing_date)
        observation_frame = self.data_ohlc.loc[start_date:end_date]
        if observation_frame.empty:
            observation_frame = self.data_ohlc.tail(1)
        if observation_frame.shape[0] < self.observation_frame_lookback + 1:
            n_needed = self.observation_frame_lookback + 1 - observation_frame.shape[0]
            last_row = observation_frame.iloc[[-1]]
            observation_frame = pd.concat([observation_frame, pd.concat([last_row] * n_needed)])
        info = {'indices': observation_frame.index.tolist(),
                'features': observation_frame.columns.tolist()}
        if self.render_mode == 'tile':
            observations = observation_frame.values.astype(np.float32)
        elif self.render_mode == 'vector':
            observations = observation_frame.values.squeeze().astype(np.float32)
        else:
            observations = observation_frame.values.astype(np.float32)
        return (observations, info)
