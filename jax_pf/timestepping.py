import equations
from typing import TypeVar, Callable, Sequence
import jax.numpy as jnp

State = TypeVar("State")
TimeStepFn = Callable[[State], State]

def forward_euler(equation: equations.ExplicitODE, dt: float) -> TimeStepFn:
    
    def step_fn(u, t):
        return u + dt * equation.explicit_terms(u, t)
    
    return step_fn

def runge_kutta_4(equation: equations.ExplicitODE, dt: float) -> TimeStepFn:
    
    def step_fn(u, t):
        k1 = equation.explicit_terms(u, t)
        k2 = equation.explicit_terms(u + dt * k1 / 2, t)
        k3 = equation.explicit_terms(u + dt * k2 / 2, t)
        k4 = equation.explicit_terms(u + dt * k3, t)
        return u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return step_fn

def strange_splitting(equation: equations.TimeSplittingODE, dt: float) -> TimeStepFn:
    
    def step_fn(u, t):
        tmp = u*jnp.exp(equation.A_terms(u, t)*0.5*dt)
        tmp = equation.ifft(tmp)
        tmp = tmp*jnp.exp(equation.B_terms(tmp, t)*dt)
        tmp /= jnp.sqrt(jnp.sum(jnp.abs(tmp)**2) * equation.domain.dx[0]**2)
        tmp = equation.fft(tmp)
        return tmp*jnp.exp(equation.A_terms(u, t)*0.5*dt)
        
    return step_fn

def strange_splitting_adi(equation: equations.TimeSplittingODE, dt: float) -> TimeStepFn:
    
    def step_fn(u, t):
        A1, A2 = equation.A_terms(u, t)
        ux_hat = equation.fft(u, axis=0)
        tmp = ux_hat*jnp.exp(A1*0.5*dt)
        tmp = equation.ifft(tmp, axis=0)
        uy_hat = equation.fft(tmp, axis=1)
        tmp = uy_hat*jnp.exp(A2*0.5*dt)
        tmp = equation.ifft(tmp, axis=1)
        tmp = tmp*jnp.exp(equation.B_terms(tmp, t)*dt)
        uy_hat = equation.fft(tmp, axis=1)
        tmp = uy_hat*jnp.exp(A2*0.5*dt)
        tmp = equation.ifft(tmp, axis=1)
        ux_hat = equation.fft(tmp, axis=0)
        tmp = ux_hat*jnp.exp(A1*0.5*dt)
        tmp = equation.ifft(tmp, axis=0)
        tmp /= jnp.sqrt(jnp.sum(jnp.abs(tmp)**2) * equation.domain.dx[0]**2)
        return tmp
        
    return step_fn
