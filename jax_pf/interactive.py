import numpy as np
import pandas as pd
import os

import jax.numpy as jnp

from typing import Callable, Tuple


def get_path_function(file_path: str, x_s: float, t_s: float) -> Callable:
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None, skiprows=1, names=['timestamp', 'x_position', 'y_position'])

    # Extract time, x, and y as NumPy arrays
    time_values = df['timestamp'].values / t_s
    x_values = df['x_position'].values / x_s
    y_values = df['y_position'].values / x_s
    
    # Define a function for linear interpolation
    def interpolate_coordinates(time_point: float) -> Tuple[float, float]:
        x_interp = jnp.interp(time_point, time_values, x_values)
        y_interp = jnp.interp(time_point, time_values, y_values)
        return x_interp, y_interp
    
    return interpolate_coordinates