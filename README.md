# PortfolioOptimization

This project presents a reinforcement learning framework for portfolio optimization
that leverages historical OHLC data to simulate realistic trading conditions.


I have developed a custom Gymnasium-compatible environment that incorporates key market dynamics such as transaction costs, slippage, and inherent
trading frictions, enabling the agent to experience realistic risk and return sce-
narios during training. The environment offers both discrete and continuous
action spaces along with flexible observation modes (vector or tile), ensuring
adaptability to different portfolio strategies. 

To mitigate risk, the framework employs techniques like weight normalization via softmax, clamping of returns
to avoid extreme losses, and the inclusion of costs that simulate real-world trad-
ing constraints. 

The optimal policy is learned using PPO, A2C, and DDPG algorithms from Stable-Baselines3, where
the episodic setup (with clear termination and truncation conditions) aids in robust policy convergence over finite trading horizons. Preliminary experiments
indicate a promising improvement in cumulative returns against baseline strategies while maintaining controlled risk exposure, suggesting a viable pathway for
further research and live application in dynamic financial markets.
