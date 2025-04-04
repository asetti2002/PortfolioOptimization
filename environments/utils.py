import pandas as pd
import numpy as np
import torch

def process_ohlc_data(data):
    pass

def get_weights_asTensors(new_weights, current_weights) -> torch.Tensor:
    return torch.tensor(new_weights) - torch.tensor(current_weights).mean()