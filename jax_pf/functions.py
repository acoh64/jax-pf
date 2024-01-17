import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from flax import linen as nn
import dataclasses

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
    
class FlxFuncDiff(nn.Module):
    """MLP with `abs` final layer activation."""
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            x = nn.activation.softplus(x)
            #if i != len(self.features) - 1:
               # x = nn.relu(x)
        #return jnp.abs(x)
        return x

class FlxFuncChemPot(nn.Module):
    
    features: Sequence[int]
    ks: int
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Conv(features=feat, kernel_size=(self.ks, self.ks), strides=1, padding='CIRCULAR')(x)
            if i != len(self.features)-1:
                x = nn.activation.elu(x)
        return x
    
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
    
    
@dataclasses.dataclass
class ChemPotSimple():
    
    def apply(self, state, params):
        return state**3 - state

@dataclasses.dataclass
class LegendrePolynomials():
    
    max_degree: int
    
    def __post_init__(self):
        if self.max_degree == 0:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x))
        elif self.max_degree == 1:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x))
        elif self.max_degree == 2:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x) + p[2]*self.T2(x))
        elif self.max_degree == 3:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x) + p[2]*self.T2(x) + p[3]*self.T3(x))
        elif self.max_degree == 4:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x) + p[2]*self.T2(x) + p[3]*self.T3(x) + p[4]*self.T4(x))
        elif self.max_degree == 5:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x) + p[2]*self.T2(x) + p[3]*self.T3(x) + p[4]*self.T4(x) + p[5]*self.T5(x))
        elif self.max_degree == 6:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x) + p[2]*self.T2(x) + p[3]*self.T3(x) + p[4]*self.T4(x) + p[5]*self.T5(x) + p[6]*self.T6(x))
        elif self.max_degree == 7:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x) + p[2]*self.T2(x) + p[3]*self.T3(x) + p[4]*self.T4(x) + p[5]*self.T5(x) + p[6]*self.T6(x) + p[7]*self.T7(x))
        elif self.max_degree == 8:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x) + p[2]*self.T2(x) + p[3]*self.T3(x) + p[4]*self.T4(x) + p[5]*self.T5(x) + p[6]*self.T6(x) + p[7]*self.T7(x) + p[8]*self.T8(x))
        elif self.max_degree == 9:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x) + p[2]*self.T2(x) + p[3]*self.T3(x) + p[4]*self.T4(x) + p[5]*self.T5(x) + p[6]*self.T6(x) + p[7]*self.T7(x) + p[8]*self.T8(x) + p[9]*self.T9(x))
        elif self.max_degree == 10:
            self.func = jax.jit(lambda p,x: p[0]*self.T0(x) + p[1]*self.T1(x) + p[2]*self.T2(x) + p[3]*self.T3(x) + p[4]*self.T4(x) + p[5]*self.T5(x) + p[6]*self.T6(x) + p[7]*self.T7(x) + p[8]*self.T8(x) + p[9]*self.T9(x) + p[10]*self.T10(x))
    
    def apply(self, params, inputs):
        return self.func(params, inputs)
        
    def T0(self, x):
        return 1.0
    def T1(self, x):
        return x
    def T2(self, x):
        return 0.5*(3*x**2 - 1.0)
    def T3(self, x):
        return 0.5*(5*x**3 - 3*x)
    def T4(self, x):
        return 0.125*(35*x**4 - 30*x**2 + 3)
    def T5(self, x):
        return 0.125*(63*x**5 - 70*x**3 + 15*x)
    def T6(self, x):
        return 0.0625*(231*x**6 - 315*x**4 + 105*x**2 - 5)
    def T7(self, x):
        return 0.0625*(429*x**7 - 693*x**5 + 315*x**3 - 35*x)
    def T8(self, x):
        return 0.0078125*(6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35)
    def T9(self, x):
        return 0.0078125*(12155*x**9 - 25740*x**7 + 18018*x**5 - 4620*x**3 + 315*x)
    def T10(self, x):
        return 0.00390625*(46189*x**10 - 109395*x**8 + 90090*x**6 - 30030*x**4 + 3465*x**2 - 63)
    
    