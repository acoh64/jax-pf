import jax
import jax.numpy as jnp
import numpy as np
import math
import typing
from typing import Callable, Any, Optional, Sequence, Tuple, TypeVar, Union

from jax_pf import equations

Array = jax.Array

Carry = TypeVar('Carry')
Input = TypeVar('Input')
Output = TypeVar('Output')
Func = TypeVar('Func', bound=Callable)

def inner_sim(step_fn: Callable, 
                dt: float, 
                steps: int, 
                nested_lengths: Sequence[int]
) -> Callable:
    """Inner loop for running simulations.

    Args:
        step_fn: function called at each iteration
        dt: timestep of simulation
        steps: number of steps in inner loop (between consecutive output states)
        nested_lengths: used for checkpointing, see below
                        to use without checkpointing, set `nested_lengths = [steps]`
    Returns:
        function that runs an a number of iterations between two saved output states
    """
    def f_repeated(x_initial, t_initial):
        g = lambda x, t: (step_fn(x, t), None)
        x_final, _ = nested_checkpoint_scan(g, x_initial, xs=jnp.linspace(t_initial, t_initial+(steps-1)*dt, num=steps), nested_lengths=nested_lengths)
        return x_final
    return f_repeated

def outer_sim(inner_fn: Callable, 
                dt: float, 
                tstart: float, 
                tfinal: float, 
                save_at: float, 
                num_saves: int
) -> Callable:
    """Outer loop for running simulations.

    Args:
        inner_fn: function that runs inner loop
        dt: timestep of simulation
        tstart: starting time of simulation
        tfinal: final time of simulation
        save_at: states are outputted at multiples of `save_at`
        num_saves: number of output states to save
    Returns:
        function that runs a simulation
    """
    def step(carry_in, t):
        carry_out = inner_fn(carry_in, t)
        frame = carry_out
        return carry_out, frame
    
    def multistep(values):
        return jax.lax.scan(step, values, xs=jnp.linspace(tstart, tfinal-save_at*dt, num=num_saves))
    
    return multistep

def traj_scan(
    u0: Array, 
    dt: float, 
    step_fn: Callable, 
    steps: Array, 
    write_every: int, 
    checkpoint=False, 
    nested_lengths: Optional[Sequence[int]] = None
) -> Array:
    """Function for running a simulation. Simpler than `inner/outer_sim_t` but slower.

    Args:
        u0: initial condition
        dt: timestep of simulation
        step_fn: function that takes one simulation step
        steps: total number of simulation steps to take
        write_every: states are outputted at multiples of `write_every`
        checkpoint: whether or not to checkpoint
        nested_lengths: if checkpointing, see below
    Returns:
        result of running the simulation
    """
    res = jnp.zeros((steps.shape[0] // write_every,) + u0.shape, dtype=u0.dtype)
        
    def stepper(u_res, t):
        u, res = u_res
        u = step_fn(u, t*dt)
        res = jax.lax.cond(t % write_every == 0, lambda p: p.at[t // write_every].set(u), lambda p: p, res)
        return (u, res), None

    if checkpoint:
        (uf, res), _ = nested_checkpoint_scan(stepper, (u0, res), xs=steps, nested_lengths=nested_lengths)
        return res
    else:
        (uf, res), _ = jax.lax.scan(stepper, (u0, res), steps)
        return res    
    
    
def set_up_timestepping(
    tstart: float, 
    tfinal:float, 
    dt: float, 
    save_at: float, 
    nested_lengths: Sequence[int]
) -> Tuple[int, int, int, Array]:
    """Function to set up timestepping.
       Care must be taken to make sure number 
       of iteractions between saves is consistent
       with checkpointing.

    Args:
        tstart: starting time of simulation
        tfinal: final time of simulation
        dt: timestep of simulation
        save_at: states are outputted at multiples of `save_at`
        nested_lengths: used for checkpointing, see below
                        to use without checkpointing, set `nested_lengths = [steps]`
    Returns:
        tf: final time of simulation
            may be slightly adjusted to be a multiple of `save_at`
        num_steps: number of simulation steps to run
        num_out_states: number of states to output
        ts: times at which output states occur
    """
    num_steps = int(tfinal // dt)
    save_every = int(save_at // dt)
    num_out_states = int(num_steps // save_every)
    
    tf = save_at * (tfinal//save_at)
    
    cond1 = round((save_every)**(1.0/len(nested_lengths)))**len(nested_lengths) != save_every
    cond2 = jnp.prod(jnp.array(nested_lengths)) != save_every
    
    if cond1 or cond2:
        raise ValueError(f'inconsistent {save_every=} and {nested_lengths=}')
   
    return tf, num_steps, save_every, num_out_states, jnp.linspace(tstart+save_at, tf, num=num_out_states)


def set_up_simulation(
    equation: equations.ODE,
    step_fn: Callable,
    tstart: float,
    tfinal: float,
    dt: float,
    save_at: float,
    save_every: int,
    out_states: int,
    nested_lengths: Sequence[int]
) -> Callable:
    """Function to set up the simulation function.
       Care must be taken to make sure number 
       of iteractions between saves is consistent
       with checkpointing.

    Args:
        step_fn: function that runs one step of simulation
        tstart: starting time of simulation
        tfinal: final time of simulation
        dt: timestep of simulation
        save_at: states are outputted at multiples of `save_at` in time units
        save_every: states are outputted at multiples of `save_every` iterations
        out_states: number of states to output
        nested_lengths: used for checkpointing, see below
                        to use without checkpointing, set `nested_lengths = [steps]`
    Returns:
        function that runs a simulation for a given initial state
    """
    inner_fn = inner_sim(step_fn, dt, save_every, nested_lengths)
    
    if equation.space == "R":
        return outer_sim(inner_fn, dt, tstart, tfinal, save_at, out_states)
    else:
        def sim_func(u0):
            u0_hat = equation.fft(u0)
            _, res = outer_sim(inner_fn, dt, tstart, tfinal, save_at, out_states)(u0_hat)
            return jax.vmap(equation.ifft)(res)
        return sim_func
    
# implementation from
# https://github.com/google/jax/issues/2139
def nested_checkpoint_scan(
    f: Callable[[Carry, Input], Tuple[Carry, Output]],
    init: Carry,
    xs: Input,
    length: Optional[int] = None,
    *,
    nested_lengths: Sequence[int],
    scan_fn = jax.lax.scan, #this line maybe should changes
    checkpoint_fn: Callable[[Func], Func] = jax.checkpoint,
) -> Tuple[Carry, Output]:
    """A version of lax.scan that supports recursive gradient checkpointing.

    The interface of `nested_checkpoint_scan` exactly matches lax.scan, except for
    the required `nested_lengths` argument.

    The key feature of `nested_checkpoint_scan` is that gradient calculations
    require O(max(nested_lengths)) memory, vs O(prod(nested_lengths)) for unnested
    scans, which it achieves by re-evaluating the forward pass
    `len(nested_lengths) - 1` times.

    `nested_checkpoint_scan` reduces to `lax.scan` when `nested_lengths` has a
    single element.

    Args:
    f: function to scan over.
    init: initial value.
    xs: scanned over values.
    length: leading length of all dimensions
    nested_lengths: required list of lengths to scan over for each level of
      checkpointing. The product of nested_lengths must match length (if
      provided) and the size of the leading axis for all arrays in ``xs``.
    scan_fn: function matching the API of lax.scan
    checkpoint_fn: function matching the API of jax.checkpoint.

    Returns:
    Carry and output values.
    """
    if length is not None and length != math.prod(nested_lengths):
        raise ValueError(f'inconsistent {length=} and {nested_lengths=}')

    def nested_reshape(x):
        x = jnp.asarray(x)
        new_shape = tuple(nested_lengths) + x.shape[1:]
        return x.reshape(new_shape)

    sub_xs = jax.tree_map(nested_reshape, xs)
    return _inner_nested_scan(f, init, sub_xs, nested_lengths, scan_fn,
                              checkpoint_fn)


def _inner_nested_scan(f, init, xs, lengths, scan_fn, checkpoint_fn):
    """Recursively applied scan function."""
    if len(lengths) == 1:
        return scan_fn(f, init, xs, lengths[0])

    @checkpoint_fn
    def sub_scans(carry, xs):
        return _inner_nested_scan(f, carry, xs, lengths[1:], scan_fn, checkpoint_fn)

    carry, out = scan_fn(sub_scans, init, xs, lengths[0])
    stacked_out = jax.tree_map(jnp.concatenate, out)
    return carry, stacked_out

# borrowed from 
# https://github.com/biswaroopmukherjee/condensate/tree/master
def initialize_Psi(N, width=100, vortexnumber=0):
    """Function for creating an initial condition for GPE
    
    Args:
        N: number of grid points
        width: width of blob
        vortextnumber: number of vortices to intialize
        
    Returns:
        initial condition for GPE simulation
    """
    psi = np.zeros((N,N), dtype=complex)
    for i in range(N):
        for j in range(N):
            phase = 1
            if vortexnumber:
                phi = vortexnumber * np.arctan2((i-N//2), (j-N//2))
                phase = np.exp(1.j * np.mod(phi,2*np.pi))
            psi[i,j] = np.exp(-( (i-N//2)/width)** 2.  -  ((j-N//2)/width)** 2. )
            psi[i,j] *= phase     
    psi = psi.astype(complex)
    return jnp.array(psi)