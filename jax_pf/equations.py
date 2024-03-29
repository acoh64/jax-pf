import jax
import jax.numpy as jnp
from typing import TypeVar, Callable, Any
import dataclasses

from jax_pf import domains
from jax_pf import fftutils
from jax_pf import functions

State = TypeVar("State")

Array = jax.Array

PyTree = Any

class ODE:
    pass

class ExplicitODE(ODE):
    
    def explicit_terms(self, state: State, t: float) -> State:
        raise NotImplementedError

class ImplicitExplicitODE(ODE):
    
    def explicit_terms(self, state: State, t: float) -> State:
        raise NotImplementedError
        
    def implicit_terms(self, state: State, t: float) -> State:
        raise NotImplementedError
        
    def implicit_solve(self, state: State, t: float, step_size: float) -> State:
        raise NotImplementedError
        
class TimeSplittingODE(ODE):
    
    def A_terms(self, state: State, t: float) -> State:
        raise NotImplementedError
        
    def B_terms(self, state: State, t:float) -> State:
        raise NotImplementedError

@dataclasses.dataclass
class Exponential(ExplicitODE):
    """For testing."""
    
    p: float
    
    def explicit_terms(self, state, t):
        return self.p * state

@dataclasses.dataclass
class CumSum(ExplicitODE):
    """For testing."""
        
    def explicit_terms(self, state, t):
        return t


@dataclasses.dataclass
class Advection1D(ExplicitODE):
    """For testing."""
    
    domain: domains.Domain
    
    def __post_init__(self):
        self.kx, = self.domain.fft_axes()
        self.two_pi_i_k = (2j * jnp.pi * self.kx)
        self.fft = jnp.fft.fft
        self.ifft = jnp.fft.ifft
        
    def explicit_terms(self, state, t):
        return -self.ifft(self.two_pi_i_k*self.fft(state)).real

@dataclasses.dataclass
class Advection2D(ExplicitODE):
    """For testing."""
    
    domain: domains.Domain
    v: Array
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = (2j * jnp.pi * self.kx)
        self.two_pi_i_ky = (2j * jnp.pi * self.ky)
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn
        
    def explicit_terms(self, state, t):
        return -self.v[0]*self.ifft(self.two_pi_i_kx*self.fft(state)).real - self.v[1]*self.ifft(self.two_pi_i_ky*self.fft(state)).real

@dataclasses.dataclass
class CahnHilliard2DIMEX(ImplicitExplicitODE):
    
    domain: domains.Domain
    gamma: float
    mu: Callable
    smooth: bool = True
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx_2 = (2j * jnp.pi * self.kx)**2
        self.two_pi_i_ky_2 = (2j * jnp.pi * self.ky)**2
        self.two_pi_i_k_4 = (self.two_pi_i_kx_2 + self.two_pi_i_ky_2)**2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        
    def explicit_terms(self, state_hat, t):
        state = self.ifft(state_hat)
        return (self.two_pi_i_kx_2 + self.two_pi_i_ky_2)*self.fft(self.mu(state))
    
    def implicit_terms(self, state_hat, t):
        return -self.gamma*(self.two_pi_i_k_4)*state_hat
    
    def implicit_solve(self, state_hat, t, time_step):
        return state_hat/(1.0 + time_step*self.gamma*(self.two_pi_i_k_4))

@dataclasses.dataclass
class CahnHilliard2DHatEX(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    mu: Callable
    smooth: bool = True
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx_2 = (2j * jnp.pi * self.kx)**2
        self.two_pi_i_ky_2 = (2j * jnp.pi * self.ky)**2
        self.two_pi_i_k_4 = (self.two_pi_i_kx_2 + self.two_pi_i_ky_2)**2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        
    def explicit_terms(self, state_hat, t):
        state = self.ifft(state_hat)
        return (self.two_pi_i_kx_2 + self.two_pi_i_ky_2)*self.fft(self.mu(state)) - self.gamma*(self.two_pi_i_k_4)*state_hat

@dataclasses.dataclass
class CahnHilliard2DEXNN(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    chem_pot: functions.FlxFunc
    params: Any
    smooth: bool = True
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx_2 = (2j * jnp.pi * self.kx)**2
        self.two_pi_i_ky_2 = (2j * jnp.pi * self.ky)**2
        self.two_pi_i_k_4 = (self.two_pi_i_kx_2 + self.two_pi_i_ky_2)**2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        
    def explicit_terms(self, state_hat, t):
        state = self.ifft(state_hat)
        return (self.two_pi_i_kx_2 + self.two_pi_i_ky_2)*self.fft(jnp.reshape(self.chem_pot.apply(self.params, jnp.reshape(state.real, (self.domain.points[0]*self.domain.points[1],1))), state.shape) - self.gamma*self.ifft((self.two_pi_i_kx_2 + self.two_pi_i_ky_2)*state_hat))

@dataclasses.dataclass
class CahnHilliard3DIMEX(ImplicitExplicitODE):
    
    domain: domains.Domain
    gamma: float
    mu: Callable
    smooth: bool = True
    
    def __post_init__(self):
        self.kx, self.ky, self.kz = self.domain.fft_mesh()
        self.two_pi_i_kx_2 = (2j * jnp.pi * self.kx)**2
        self.two_pi_i_ky_2 = (2j * jnp.pi * self.ky)**2
        self.two_pi_i_kz_2 = (2j * jnp.pi * self.kz)**2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2 + self.two_pi_i_kz_2
        self.two_pi_i_k_4 = (self.two_pi_i_k_2)**2
        self.fft = fftutils.truncated_fft_2x_3D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_3D if self.smooth else jnp.fft.ifftn
        
    def explicit_terms(self, state_hat):
        state = self.ifft(state_hat)
        return self.two_pi_i_k_2*self.fft(self.mu(state)) 
    
    def implicit_terms(self, state_hat):
        return -self.gamma*(self.two_pi_i_k_4)*state_hat
    
    def implicit_solve(self, state_hat, t, time_step):
        return state_hat/(1.0 + time_step*self.gamma*(self.two_pi_i_k_4))


@dataclasses.dataclass
class CahnHilliard3DEX(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    mu: Callable
    D: Callable
    smooth: bool = True
    
    def __post_init__(self):
        self.kx, self.ky, self.kz = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kz = 2j * jnp.pi * self.kz
        self.two_pi_i_kx_2 = (self.two_pi_i_kx)**2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky)**2
        self.two_pi_i_kz_2 = (self.two_pi_i_kz)**2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2 + self.two_pi_i_kz_2
        self.two_pi_i_k_4 = (self.two_pi_i_k_2)**2
        self.fft = fftutils.truncated_fft_2x_3D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_3D if self.smooth else jnp.fft.ifftn
        
    def explicit_terms(self, state_hat, t):
        state = self.ifft(state_hat)
        tmp = self.fft(self.mu(state)) - self.gamma*(self.two_pi_i_k_2)*state_hat
        tmpx = self.fft(self.D(state) * self.ifft(self.two_pi_i_kx * tmp))
        tmpy = self.fft(self.D(state) * self.ifft(self.two_pi_i_ky * tmp))
        tmpz = self.fft(self.D(state) * self.ifft(self.two_pi_i_kz * tmp))
        
        return self.two_pi_i_kx * tmpx + self.two_pi_i_ky * tmpy + self.two_pi_i_kz * tmpz

@dataclasses.dataclass
class CahnHilliard3DEXNN(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    mu: functions.FlxChemPot
    D: functions.FlxFuncDiff
    params_mu: Any
    params_D: Any
    smooth: bool = True
    
    def __post_init__(self):
        self.kx, self.ky, self.kz = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kz = 2j * jnp.pi * self.kz
        self.two_pi_i_kx_2 = (self.two_pi_i_kx)**2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky)**2
        self.two_pi_i_kz_2 = (self.two_pi_i_kz)**2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2 + self.two_pi_i_kz_2
        self.two_pi_i_k_4 = (self.two_pi_i_k_2)**2
        self.fft = fftutils.truncated_fft_2x_3D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_3D if self.smooth else jnp.fft.ifftn
        
    def explicit_terms(self, state_hat, t):
        state = self.ifft(state_hat)
        tmpmu = jnp.reshape(
            self.mu.apply(self.params_mu, 
                                jnp.reshape(
                                    state.real, 
                                    (self.domain.points[0]*self.domain.points[1]*self.domain.points[2],1))), 
            state.shape)
        #tmpmu = state**3 - state
        tmp = self.fft(tmpmu) - self.gamma*(self.two_pi_i_k_2)*state_hat
        tmpD = jnp.reshape(
            self.D.apply(self.params_D, 
                                jnp.reshape(
                                    state.real, 
                                    (self.domain.points[0]*self.domain.points[1]*self.domain.points[2],1))), 
            state.shape)
        #tmpD = 1.0 - state
        tmpx = self.fft(tmpD * self.ifft(self.two_pi_i_kx * tmp))
        tmpy = self.fft(tmpD * self.ifft(self.two_pi_i_ky * tmp))
        tmpz = self.fft(tmpD * self.ifft(self.two_pi_i_kz * tmp))
        
        return self.two_pi_i_kx * tmpx + self.two_pi_i_ky * tmpy + self.two_pi_i_kz * tmpz


@dataclasses.dataclass
class GPE2DTS(TimeSplittingODE):
    
    domain: domains.Domain
    k: float
    e: float
    smooth: bool = True
    space: str = "F"
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx)**2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky)**2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        self.xmesh, self.ymesh = self.domain.mesh()
        
    def A_terms(self, state_hat, t):
        return 0.5j*(self.two_pi_i_k_2) 
    
    def B_terms(self, state, t):
        return -0.5j*((1+self.e)*self.xmesh**2 + (1-self.e)*self.ymesh**2) - self.k*1j*(jnp.abs(state)**2)

@dataclasses.dataclass
class GPE2DTSControl(TimeSplittingODE):
    
    domain: domains.Domain
    k: float
    e: float
    lights: Callable
    smooth: bool = True
    space: str = "F"
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx)**2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky)**2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        self.xmesh, self.ymesh = self.domain.mesh()
        self.control = lambda t: self.lights(t, self.xmesh, self.ymesh)
        
    def A_terms(self, state_hat, t):
        return 0.5j*(self.two_pi_i_k_2) 
    
    def B_terms(self, state, t):
        return -0.5j*((1+self.e)*self.xmesh**2 + (1-self.e)*self.ymesh**2) - 1j*self.control(t) - self.k*1j*(jnp.abs(state)**2)

@dataclasses.dataclass
class GPE2DTSRot(TimeSplittingODE):
    
    domain: domains.Domain
    k: float
    e: float
    omega: float
    smooth: bool = True
    space: str = "R"
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx)**2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky)**2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = fftutils.truncated_fft_2x if self.smooth else jnp.fft.fft
        self.ifft = fftutils.padded_ifft_2x if self.smooth else jnp.fft.ifft
        self.xmesh, self.ymesh = self.domain.mesh()
        
    def A_terms(self, state_hat, t):
        return 0.5j*self.two_pi_i_kx_2 - self.omega*self.ymesh*self.two_pi_i_kx, 0.5j*self.two_pi_i_ky_2 + self.omega*self.xmesh*self.two_pi_i_ky

    def B_terms(self, state, t):
        return -0.5j*((1+self.e)*self.xmesh**2 + (1-self.e)*self.ymesh**2) - self.k*1j*(jnp.abs(state)**2)

    

@dataclasses.dataclass
class Diffusion2DFDDirichlet(ExplicitODE):
    
    domain: domains.Domain
    D: float
    space: str = "R"
    
    def explicit_terms(self, state, t):
        df_dx, df_dy = jnp.gradient(state, *self.domain.dx, axis=(1,0))
        d2f_dx2, d2f_dy2 = jnp.gradient(self.D*df_dx, self.domain.dx[0], axis=(1)), jnp.gradient(self.D*df_dy, self.domain.dx[1], axis=(0))
        return d2f_dx2 + d2f_dy2
        
        
    def dirichlet_bc_x(self, t):
        return 1.0
    
    def dirichlet_bc_y(self, t):
        return 1.0
    

@dataclasses.dataclass
class Diffusion2D(ExplicitODE):
    
    domain: domains.Domain
    D: float
    smooth: bool = False
    space: str = "F"
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
    
    def explicit_terms(self, state_hat, t):
        tmpx = self.D * self.two_pi_i_kx * state_hat
        tmpy = self.D * self.two_pi_i_ky * state_hat
        
        return self.two_pi_i_kx * tmpx + self.two_pi_i_ky * tmpy
    
    
@dataclasses.dataclass
class CahnHilliardPS2DDirichlet(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    mu: Callable
    bc_x: Callable
    bc_y: Callable
    smooth: bool = True
    space: str = "R"
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx_2 = (2j * jnp.pi * self.kx)**2
        self.two_pi_i_ky_2 = (2j * jnp.pi * self.ky)**2
        self.two_pi_i_k_4 = (self.two_pi_i_kx_2 + self.two_pi_i_ky_2)**2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        
    def explicit_terms(self, state, t):
        state_hat = self.fft(state)
        return self.ifft((self.two_pi_i_kx_2 + self.two_pi_i_ky_2)*self.fft(self.mu(state)) - self.gamma*(self.two_pi_i_k_4)*state_hat).real
    
    def dirichlet_bc_x(self, t):
        return self.bc_x(t)
    
    def dirichlet_bc_y(self, t):
        return self.bc_y(t)
    
@dataclasses.dataclass
class CahnHilliardPS2D(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    mu: Callable
    smooth: bool = True
    space: str = "R"
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx_2 = (2j * jnp.pi * self.kx)**2
        self.two_pi_i_ky_2 = (2j * jnp.pi * self.ky)**2
        self.two_pi_i_k_4 = (self.two_pi_i_kx_2 + self.two_pi_i_ky_2)**2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        
    def explicit_terms(self, state, t):
        state_hat = self.fft(state)
        return self.ifft((self.two_pi_i_kx_2 + self.two_pi_i_ky_2)*self.fft(self.mu(state)) - self.gamma*(self.two_pi_i_k_4)*state_hat).real
    
@dataclasses.dataclass
class CahnHilliardFD2D(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    mu: Callable
    #D: Callable
    params_mu: PyTree = None
    #params_D: PyTree
    space: str = "R"
    
    def __post_init__(self):
        self.gradx = lambda arr: (jnp.roll(arr, -1, axis=1)-jnp.roll(arr, 1, axis=1))/(2*self.domain.dx[0])
        self.grady = lambda arr: (jnp.roll(arr, -1, axis=0)-jnp.roll(arr, 1, axis=0))/(2*self.domain.dx[1])
        self.laplacian = lambda arr: (jnp.roll(arr, -1, axis=1)+jnp.roll(arr, 1, axis=1)+jnp.roll(arr, -1, axis=0)+jnp.roll(arr, 1, axis=0)-4*arr)/(self.domain.dx[0]**2)
        
    def explicit_terms(self, state, t):
        mu = self.mu.apply(state, self.params_mu) - self.gamma*self.laplacian(state)
        return self.laplacian(mu)
    
    
@dataclasses.dataclass
class CahnHilliardFD2DDirichlet(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    mu: Callable
    #D: Callable
    params_mu: PyTree
    #params_D: PyTree
    bc_x: Callable
    bc_y: Callable
    space: str = "R"
    
    def __post_init__(self):
        self.gradx = lambda arr: (jnp.roll(arr, -1, axis=1)-jnp.roll(arr, 1, axis=1))/(2*self.domain.dx[0])
        self.grady = lambda arr: (jnp.roll(arr, -1, axis=0)-jnp.roll(arr, 1, axis=0))/(2*self.domain.dx[1])
        self.laplacian = lambda arr: (jnp.roll(arr, -1, axis=1)+jnp.roll(arr, 1, axis=1)+jnp.roll(arr, -1, axis=0)+jnp.roll(arr, 1, axis=0)-4*arr)/(self.domain.dx[0]**2)
        
    def explicit_terms(self, state, t):
        #mu = self.mu.apply(self.params_mu,state[...,None])[...,0] - self.gamma*self.laplacian(state)
        mu = self.mu.apply(self.params_mu,state[...,None])[...,0]
        return self.laplacian(mu)
    
    def dirichlet_bc_x(self, t):
        return self.bc_x(t)
    
    def dirichlet_bc_y(self, t):
        return self.bc_y(t)
    
    
@dataclasses.dataclass
class CahnHilliardSmoothedBoundary(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    chem_pot: Callable
    D: Callable
    space: str = "R"
    
    def __post_init__(self):
        self.psi = self.domain.geometry.smooth
    
    def explicit_terms(self, state, t):
        
        flux_left = 0.5 * (self.psi[:-2, 1:-1] + self.psi[1:-1, 1:-1]) * (state[1:-1, 1:-1] - state[:-2, 1:-1]) / self.domain.dx[0]
        flux_right = 0.5 * (self.psi[1:-1, 1:-1] + self.psi[2:, 1:-1]) * (state[2:, 1:-1] - state[1:-1, 1:-1]) / self.domain.dx[0]
        flux_bottom = 0.5 * (self.psi[1:-1, :-2] + self.psi[1:-1, 1:-1]) * (state[1:-1, 1:-1] - state[1:-1, :-2]) / self.domain.dx[1]
        flux_top = 0.5 * (self.psi[1:-1, 1:-1] + self.psi[1:-1, 2:]) * (state[1:-1, 2:] - state[1:-1, 1:-1]) / self.domain.dx[1]
        
        mu = -self.gamma * (((flux_right - flux_left) / self.domain.dx[0] + (flux_top - flux_bottom) / self.domain.dx[1]) / self.psi[1:-1, 1:-1]) + self.chem_pot(0.5 * (state[:-2, 1:-1] + state[1:-1, 1:-1]))
        
        mu = jnp.vstack([mu[0, :], mu, mu[-3, :]])
        mu = jnp.hstack([mu[:, [0]], mu, mu[:, [-3]]])
        
        diffusivity = 0.5 * (state[:-2, 1:-1] + state[1:-1, 1:-1]) * self.D(0.5 * (state[:-2, 1:-1] + state[1:-1, 1:-1]))
        
        flux_left = 0.5 * (self.psi[:-2, 1:-1] + self.psi[1:-1, 1:-1]) * diffusivity * (mu[1:-1, 1:-1] - mu[:-2, 1:-1]) / self.domain.dx[0]
        flux_right = 0.5 * (self.psi[1:-1, 1:-1] + self.psi[2:, 1:-1]) * diffusivity * (mu[2:, 1:-1] - mu[1:-1, 1:-1]) / self.domain.dx[0]
        flux_bottom = 0.5 * (self.psi[1:-1, :-2] + self.psi[1:-1, 1:-1]) * diffusivity * (mu[1:-1, 1:-1] - mu[1:-1, :-2]) / self.domain.dx[1]
        flux_top = 0.5 * (self.psi[1:-1, 1:-1] + self.psi[1:-1, 2:]) * diffusivity * (mu[1:-1, 2:] - mu[1:-1, 1:-1]) / self.domain.dx[1]

        dcdt = ((flux_right - flux_left) / self.domain.dx[0] + (flux_top - flux_bottom) / self.domain.dx[1]) / self.psi[1:-1, 1:-1]

        dcdt = jnp.vstack([dcdt[0, :], dcdt, dcdt[-3, :]])
        dcdt = jnp.hstack([dcdt[:, [0]], dcdt, dcdt[:, [-3]]])
        return dcdt
    
    
@dataclasses.dataclass
class CahnHilliardSmoothedBoundary2(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    chem_pot: Callable
    D: Callable
    space: str = "R"
    
    def __post_init__(self):
        self.psi = self.domain.geometry.smooth
        self.gradxf = lambda arr: (jnp.roll(arr, -1, axis=1)-arr)/(self.domain.dx[0])
        self.gradyf = lambda arr: (jnp.roll(arr, -1, axis=0)-arr)/(self.domain.dx[1])
        self.gradxb = lambda arr: (arr - jnp.roll(arr, 1, axis=1))/(self.domain.dx[0])
        self.gradyb = lambda arr: (arr - jnp.roll(arr, 1, axis=0))/(self.domain.dx[0])
        
    def explicit_terms(self, state, t):
        tmp1 = self.psi * self.gradxf(state)
        tmp2 = self.psi * self.gradyf(state)
        tmp3 = (self.gamma/self.psi) * (self.gradxb(tmp1) + self.gradyb(tmp2))
        tmp4 = self.chem_pot(state) - tmp3
        tmp5 = self.gradxf(tmp4)
        tmp6 = self.gradyf(tmp4)
        tmp7 = self.psi * self.D(state) * state
        return (self.gradxb(tmp7 * tmp5) + self.gradyb(tmp7 * tmp6)) / self.psi
    
@dataclasses.dataclass
class CahnHilliardSmoothedBoundary3(ExplicitODE):
    
    domain: domains.Domain
    gamma: float
    chem_pot: Callable
    D: Callable
    space: str = "R"
    smooth: bool = False
    
    def __post_init__(self):
        self.psi = self.domain.geometry.smooth
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        
    def explicit_terms(self, state, t):
        state_hat = self.fft(state)
        tmp1 = self.psi * self.ifft(self.two_pi_i_kx*state_hat)
        tmp2 = self.psi * self.ifft(self.two_pi_i_ky*state_hat)
        tmp1_hat = self.fft(tmp1)
        tmp2_hat = self.fft(tmp2)
        tmp3 = (self.gamma/self.psi) * self.ifft(self.two_pi_i_kx*tmp1_hat + self.two_pi_i_ky*tmp2_hat)
        tmp4_hat = self.fft(self.chem_pot(state) - tmp3)
        tmp5 = self.ifft(self.two_pi_i_kx*tmp4_hat)
        tmp6 = self.ifft(self.two_pi_i_ky*tmp4_hat)
        tmp7 = self.psi * self.D(state) * state
        return self.ifft(self.two_pi_i_kx*self.fft(tmp7 * tmp5) + self.two_pi_i_ky*self.fft(tmp7 * tmp6)).real / self.psi