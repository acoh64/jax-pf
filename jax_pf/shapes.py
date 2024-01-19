import dataclasses

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from typing import Tuple, Optional
import dataclasses

from jax_pf import shapes

import numpy as np

Array = jax.Array

@dataclasses.dataclass
class Shape:
    """Sets up a geometry/shape for solving PDE on with smoothed boundary method.

    The user creates a shape by providing a binary representation and an optional smoothing parameter.
    """
    
    binary: Array
    method: str
    dx: Optional[float] = 1.0
    smooth_params: Optional[Tuple[float, float, float]] = (1.0, 0.1, 1.0)
    
    def __post_init__(self):
        if self.method == "r":
            self.smooth = self.smooth_shape_rxndiff()
        else:
            self.smooth = self.smooth_shape_phasefield()
        
    def smooth_shape_rxndiff(self) -> Array:
        zeta, dt, tf = self.smooth_params
        lap = lambda arr: (jnp.roll(arr, -1, axis=1)+jnp.roll(arr, 1, axis=1)+jnp.roll(arr, -1, axis=0)+jnp.roll(arr, 1, axis=0)-4*arr)/(self.dx**2)
        deriv = lambda x: zeta**2 * lap(x) - (2.0*x - 1.0)*(-1.0 + (2.0*x - 1.0)**2)
        step = lambda x: x + dt * deriv(x)
        g = lambda x, t: (step(x), None)
        smoothed_shape, _ = jax.lax.scan(g, self.binary, None, length=int(tf/dt))
        return smoothed_shape
        
    def smooth_shape_phasefield(self):
        
        eps, dt, tf = self.smooth_params
        timesteps = int(tf/dt)

        nx, ny = self.binary.shape
        zero = 1e-15
        
        field = np.array(self.binary)

        for it in range(timesteps):
            # Laplacian calculation
            laplace = field[2:nx, 1:ny-1] - 2 * field[1:nx-1, 1:ny-1] + field[0:nx-2, 1:ny-1] + field[1:nx-1, 2:ny] - 2 * field[1:nx-1, 1:ny-1] + field[1:nx-1, 0:ny-2]

            # Calculate gradients at cell corners
            norm_R = np.sqrt((field[1:nx, 1:ny-1] - field[0:nx-1, 1:ny-1])**2 + 0.25 * 0.25 * (field[1:nx, 2:ny] + field[0:nx-1, 2:ny] - field[1:nx, 0:ny-2] - field[0:nx-1, 0:ny-2])**2)
            bulk = np.where(norm_R <= zero)
            norm_R[bulk] = 1.0

            norm_T = np.sqrt(0.25 * 0.25 * (field[2:nx, 1:ny] + field[2:nx, 0:ny-1] - field[0:nx-2, 1:ny] - field[0:nx-2, 0:ny-1])**2 + (field[1:nx-1, 1:ny] - field[1:nx-1, 0:ny-1])**2)
            bulk = np.where(norm_T <= zero)
            norm_T[bulk] = 1.0

            curvature = 0.5 * np.sqrt((field[2:nx, 1:ny-1] - field[0:nx-2, 1:ny-1])**2 + (field[1:nx-1, 2:ny] - field[1:nx-1, 0:ny-2])**2) * (
                    (field[2:nx, 1:ny-1] - field[1:nx-1, 1:ny-1]) / norm_R[1:nx-1, :] - (field[1:nx-1, 1:ny-1] - field[0:nx-2, 1:ny-1]) / norm_R[0:nx-2, :] +
                    (field[1:nx-1, 2:ny] - field[1:nx-1, 1:ny-1]) / norm_T[:, 1:ny-1] - (field[1:nx-1, 1:ny-1] - field[1:nx-1, 0:ny-2]) / norm_T[:, 0:ny-2])

            # Solve phase-field evolution with well potential
            field[1:nx-1, 1:ny-1] = field[1:nx-1, 1:ny-1] + dt * 2 * (eps * (laplace - 0.9 * curvature) - 9 / eps * field[1:nx-1, 1:ny-1] * (1 - field[1:nx-1, 1:ny-1]) * (1 - 2 * field[1:nx-1, 1:ny-1]))

            # Boundary conditions
            # No-flux BC at left and right sides
            field[0, :] = field[1, :]
            field[-1, :] = field[-2, :]
            # No-flux BC at bottom and top
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
    
        field[field < 0.00001] = 0.00001
        field[field > 0.99] = 1.0
        
        return jnp.array(field)