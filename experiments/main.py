import sys
import os

from zmq import device
# Append the parent directory so that the "environments" module can be found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random

from stable_baselines3 import PPO, DDPG, A2C, SAC
from stable_baselines3.common.env_checker import check_env
from ewp import ewp
# Import your custom environment.
from environments.portfolio_environment import PortfolioEnv
from sp500 import sp500


# SEED = 
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

def run_simulation(model, env):
    obs, info = env.reset()
    done = False
    cumulative_log_return = 0.0
    portfolio_values = []
    date_list = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Reward is daily log return, so we accumulate it:
        cumulative_log_return += reward
        # Exponentiate to get the current portfolio value relative to initial_balance.
        portfolio_value = np.exp(cumulative_log_return)
        portfolio_values.append(portfolio_value)
        # Assuming 'indices' key in info to track the date.
        current_date = info.get('indices', [None])[-1]
        date_list.append(current_date)

    # Convert the date_list to DatetimeIndex if needed.
    date_list = pd.to_datetime(date_list)
    return date_list, portfolio_values

def compute_metrics(portfolio_values, trading_days_per_year=252):
    pv = np.array(portfolio_values, dtype=float)
    n = len(pv)
    if n < 2:
        return {
            'Annualized Return': np.nan,
            'Annualized Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Sortino Ratio': np.nan,
            'Max Drawdown': np.nan
        }
    
    # 1) Annualized Return
    # Assume initial_value = portfolio_values[0], final_value = portfolio_values[-1]
    annualized_return = (pv[-1] / pv[0])**(trading_days_per_year / (n - 1)) - 1
    
    # 2) Daily returns (simple returns)
    # r_i = (pv[i] - pv[i-1]) / pv[i-1], or equivalently pv[i] / pv[i-1] - 1
    daily_returns = (pv[1:] / pv[:-1]) - 1

    # 3) Annualized Volatility (using daily_returns)
    daily_vol = np.std(daily_returns, ddof=1)
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)

    # 4) Sharpe Ratio (risk-free rate = 0)
    # Sharpe = (annualized_return - 0) / annualized_vol
    if annualized_vol != 0:
        sharpe_ratio = annualized_return / annualized_vol
    else:
        sharpe_ratio = np.nan

    # 5) Sortino Ratio
    # We need downside volatility => std of negative daily returns only
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) > 0:
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(trading_days_per_year)
    else:
        downside_vol = np.nan
    
    if downside_vol == 0 or np.isnan(downside_vol):
        sortino_ratio = np.nan
    else:
        sortino_ratio = annualized_return / downside_vol

    # 6) Maximum Drawdown
    # MDD = max_{t} [1 - (pv[t] / max_{s in [0, t]} pv[s])]
    rolling_max = np.maximum.accumulate(pv)
    drawdowns = (pv - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)  # this is negative
    max_drawdown = abs(max_drawdown)  # convert to positive fraction
    
    metrics = {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }
    return metrics

def main():
    # --- Load OHLC Data from an H5 File ---
    df_ohlc = pd.read_hdf('data/data.h5', key='instruments')
    # Ensure we use data from 1990 onward (your dataset may extend further)
    df_ohlc = df_ohlc.loc["2017-01-01":].dropna()

    plot_ewp = True
    plot_sandp500 = True
    # Common environment parameters (initial_balance defaults to 1e6 in your env)
    env_params = dict(
        data_ohlc = df_ohlc,
        agent_type = 'continuous',
        short_positions = False,
        continuous_weights = True,
        allow_short_positions = False,
        rebalance_every = 1,
        slippage = 0.001,
        transaction_cost = 0.001,
        render_mode = 'tile',
        max_trajectory_len = 252 * 8,
        observation_frame_lookback = 5,
        trajectory_bootstrapping = False,
        # episodic_instrument_shiftin = False,
        # verbose = 1
    )

    # We will train multiple agents on separate environment instances.
    algorithms = {
        # "PPO": PPO,
        "DDPG": DDPG,
        # "A2C": A2C,
        # "SAC": SAC
    }
    
    # Store date_list and portfolio_values for each algorithm
    results = {}
    # Store performance metrics for each algorithm
    performance_metrics = {}

    for algo_name, AlgoClass in algorithms.items():
        print(f"\n--- Training {algo_name} ---\n")
        # Create a fresh environment instance for this algorithm.
        env_instance = PortfolioEnv(**env_params)
        check_env(env_instance)  # Optionally validate the environment.
        
        # Create and train the model (adjust total_timesteps as needed).
        model = AlgoClass("MlpPolicy", env_instance, verbose=1, device="cuda" if torch.cuda.is_available() else "cpu")
        model.learn(total_timesteps=1500)

        # Run simulation (simulate one episode)
        date_list, portfolio_values = run_simulation(model, env_instance)
        results[algo_name] = (date_list, portfolio_values)
        
        # Compute metrics
        metrics = compute_metrics(portfolio_values)
        performance_metrics[algo_name] = metrics
    
    # --- Print performance metrics ---
    print("\nPerformance Metrics:")
    for algo_name, metrics in performance_metrics.items():
        print(f"\n[{algo_name} Metrics]")
        print(f"Annualized Return:    {metrics['Annualized Return'] * 100:.2f}%")
        print(f"Annualized Volatility:{metrics['Annualized Volatility'] * 100:.2f}%")
        print(f"Sharpe Ratio:         {metrics['Sharpe Ratio']:.4f}")
        print(f"Sortino Ratio:        {metrics['Sortino Ratio']:.4f}")
        print(f"Max Drawdown:         {metrics['Max Drawdown'] * 100:.2f}%")
    
    # --- Plot the Cumulative Portfolio Value Over Time for each agent ---
    plt.figure(figsize=(14, 8))
    for algo_name, (date_list, portfolio_values) in results.items():
        plt.plot(date_list, portfolio_values, marker='o', markersize=0.5, label=algo_name)
    
    if plot_ewp:
        # ewp() presumably returns a Series of the equally-weighted portfolio over time
        cumulative_value = ewp()
        plt.plot(cumulative_value.index, cumulative_value.values, label='Equally Weighted Portfolio')
    
    if plot_sandp500:
        # sp500() presumably returns a Series of the S&P 500 over time
        cumulative_value, _ = sp500()
        plt.plot(cumulative_value.index, cumulative_value.values, label='S&P 500')

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Relative)")
    plt.title("Cumulative Portfolio Value Over Time (Comparison of RL Algorithms)")
    plt.grid(True)
    plt.legend()

    # Save the plot to a specific folder instead of showing it
    save_folder = "plots"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # save_path = os.path.join(save_folder, "cumulative_portfolio_value_for_RL_agents_and_EWP.png")
    save_path = os.path.join(save_folder, "S&P returns with DDPG.png")
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    main()
