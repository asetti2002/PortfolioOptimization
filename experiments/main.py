import sys
import os
# Append the parent directory so that the "environments" module can be found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random

from stable_baselines3 import PPO, DDPG, A2C, SAC
from stable_baselines3.common.env_checker import check_env

# Import your custom environment.
from environments.portfolio_environment import PortfolioEnv

def run_simulation(model, env):
    """Runs one simulation episode using a trained model on the given environment.
    Returns a tuple of (date_list, portfolio_values)."""
    obs, info = env.reset()
    done = False
    cumulative_log_return = 0.0
    portfolio_values = []
    date_list = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Since reward is a float, add it directly.
        cumulative_log_return += reward
        portfolio_value = np.exp(cumulative_log_return)
        portfolio_values.append(portfolio_value)
        # Using the last date from info as the current simulation date.
        current_date = info['indices'][-1]
        date_list.append(current_date)

    # Convert the date_list to DatetimeIndex if needed.
    date_list = pd.to_datetime(date_list)
    return date_list, portfolio_values

def main():
    # --- Load OHLC Data from an H5 File ---
    df_ohlc = pd.read_hdf('data/data.h5', key='instruments')
    # Ensure we use data from 1990 onward (your dataset may extend further)
    df_ohlc = df_ohlc.loc["1990-01-01":].dropna()

    # Define common environment parameters.
    env_params = dict(
        data_ohlc = df_ohlc,
        agent_type = 'continuous',         # Using continuous action space
        short_positions = True,
        continuous_weights = True,
        allow_short_positions = True,
        rebalance_every = 1,
        slippage = 0.001,
        transaction_cost = 0.001,
        render_mode = 'tile',               # or 'vector'
        max_trajectory_len = 252 * 32,
        observation_frame_lookback = 5,
        trajectory_bootstrapping = False,
        episodic_instrument_shiftin = False,
        verbose = 1
    )

    # We will train three agents on separate environment instances
    algorithms = {
        # "PPO": PPO,
        "DDPG": DDPG,
        # "A2C": A2C,
        # "SAC": SAC
    }
    
    results = {}  # To store (date_list, portfolio_values) for each algorithm
    
    for algo_name, AlgoClass in algorithms.items():
        print(f"\n--- Training {algo_name} ---\n")
        # Create a fresh environment instance for this algorithm.
        env_instance = PortfolioEnv(**env_params)
        check_env(env_instance)  # Optionally validate the environment.
        
        # Create and train the model.
        model = AlgoClass("MlpPolicy", env_instance, verbose=1)
        # Adjust total_timesteps as needed.
        model.learn(total_timesteps=200_000)
        
        # Run simulation (simulate one episode).
        date_list, portfolio_values = run_simulation(model, env_instance)
        results[algo_name] = (date_list, portfolio_values)
    
    # --- Plot the Cumulative Portfolio Value Over Time for each agent ---
    plt.figure(figsize=(14, 8))
    for algo_name, (date_list, portfolio_values) in results.items():
        plt.plot(date_list, portfolio_values, marker='o', label=algo_name)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Relative)")
    plt.title("Cumulative Portfolio Value Over Time (Comparison of RL Algorithms)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
