import dataclasses

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from typing import Tuple, Callable, TypeVar, Sequence
import dataclasses

from jax_pf import equations
from jax_pf.simutils import inner_sim

Array = jax.Array
State = TypeVar("State")

class Environment:
    
    def step(self, action) -> Tuple[State, float]:
        raise NotImplementedError

@dataclasses.dataclass
class GPLights(Environment):
    """Sets up a reinforcement learning environment.

    The following information is stored in an Environment:
    -- `state`          the current state of the system
    -- `agent`          the agent interacting with the environment
    -- `timestepper`    function that takes in the state and agent and returns a new state
    """
    
    domain: float
    k: float
    epsilon: float
    dt: float
    cooling: float
    reaction_time: float
    num_lights: float
    state: Sequence[Array]
    
    time_stepper: Callable
    reward_fn: Callable
        
    def __post_init__(self):
        self.steps = int(self.reaction_time // self.dt)
        
        # change this into using scan function
        def light(t, p, xs, ys):
            res = jnp.zeros(xs.shape, dtype=xs.dtype)
            for i in range(0, p.shape[0], 4):
                res += p[i]*jnp.exp(-((xs-p[i+2])**2 + (ys-p[i+3])**2)/(2.0*p[i+1]))
            return res
        
        self.lights = light
        
        
    def step(self, action):
        self.state[1] += action
        _light = lambda t, xmesh, ymesh: self.lights(t, self.state[1], xmesh, ymesh)
        eq = equations.GPE2DTSControl(self.domain, self.k, self.epsilon, _light, False)
        stepfn = self.time_stepper(eq, self.dt*(1.0 - 1j*self.cooling))
        advance_fn = inner_sim(stepfn, self.dt, self.steps, [self.steps])
        self.state[0] = advance_fn(self.state[0], 0.0)
        return self.state, self.reward_fn(self.state)
    
    