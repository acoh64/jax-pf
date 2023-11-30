import jax
import jax.numpy as jnp
from typing import TypeVar, Callable, Any
import dataclasses

from jax_pf import domains
from jax_pf import fftutils
from jax_pf import functions

State = TypeVar("State")

Array = jax.Array

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
    D: functions.FlxFuncAbs
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

    
