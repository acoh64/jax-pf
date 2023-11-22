import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from flax import linen as nn

from typing import Sequence

Array = jax.Array    

class EqxFunc(eqx.Module):
    """MLP for use inside simulation."""
    
    mlp: eqx.nn.MLP
    
    def __init__(self, state_size, width_size, depth, key):
        self.mlp = eqx.nn.MLP(
            in_size = state_size,
            out_size = state_size,
            width_size = width_size,
            depth = depth,
            activation = jnn.softplus,
            key = key,
        )
        
    def __call__(self, u):
        return self.mlp(u)
    
class FlxFunc(nn.Module):
    """MLP for use inside simulation."""
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x
    
class FlxFuncAbs(nn.Module):
    """MLP with `abs` final layer activation."""
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return jnp.abs(x)
    
class FlxChemPot(nn.Module):
    """MLP to use with chemical potential with zeros at -1, 1."""
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x*(inputs-1)*(inputs+1)