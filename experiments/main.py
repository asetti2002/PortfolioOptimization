import sys
import os
# Append the parent directory so that the "environments" module can be found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import your custom environment.
from environments.portfolio_environment import PortfolioEnv

def main():
    # --- Load OHLC Data from an H5 File ---
    # Replace 'data/data.h5' and 'instruments' with your actual file path and H5 key.
    df_ohlc = pd.read_hdf('data/data.h5', key='instruments')
    df_ohlc = df_ohlc.dropna()

    # --- Create the Portfolio Environment ---
    env = PortfolioEnv(
        data_ohlc=df_ohlc,
        agent_type='continuous',         # Use 'continuous' or 'discrete'
        short_positions=False,
        continuous_weights=True,           # Continuous weight updates.
        allow_short_positions=True,
        rebalance_every=1,
        slippage=0.001,
        transaction_cost=0.001,
        render_mode='tile',                # Change to 'vector' if desired.
        max_trajectory_len=252,            # One trading year (example).
        observation_frame_lookback=5,
        trajectory_bootstrapping=False,
        episodic_instrument_shiftin=False,
        verbose=0
    )

    # (Optional) Check that the environment follows the Gym API.
    check_env(env)

    # --- Train a PPO Agent ---
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)

    # --- Run a Simulation with the Trained Policy ---
    obs, info = env.reset()
    done = False
    cumulative_log_return = 0.0
    portfolio_values = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_log_return += reward.item()
        portfolio_value = np.exp(cumulative_log_return)
        portfolio_values.append(portfolio_value)

    # --- Plot the Cumulative Portfolio Value ---
    plt.figure(figsize=(12, 8))
    plt.plot(portfolio_values, marker='o', label="Portfolio Value")
    plt.xlabel("Rebalancing Periods")
    plt.ylabel("Portfolio Value (Relative)")
    plt.title("Cumulative Portfolio Value Over Time (Optimal Policy)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
