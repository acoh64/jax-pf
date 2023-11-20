import jax
import jax.numpy as jnp
import math

Array = jax.Array

def truncated_fft_2x(u: Array) -> Array:
    uhat = jnp.fft.fftshift(jnp.fft.fft(u))
    k, = uhat.shape
    final_size = (k + 1) // 2
    return jnp.fft.ifftshift(uhat[final_size // 2:(-final_size + 1) // 2]) / 2


def padded_ifft_2x(uhat: Array) -> Array:
    n, = uhat.shape
    final_size = n + 2 * (n // 2)
    added = n // 2
    smoothed = jnp.pad(jnp.fft.fftshift(uhat), (added, added))
    assert smoothed.shape == (final_size,), "incorrect padded shape"
    return 2 * jnp.fft.ifft(jnp.fft.ifftshift(smoothed))


def truncated_fft_2x_2D(u: Array) -> Array:
    uhat = jnp.fft.fftshift(jnp.fft.fftn(u))
    kx, ky = uhat.shape
    final_size_x, final_size_y = (kx + 1) // 2, (ky + 1) // 2
    return jnp.fft.ifftshift(uhat[final_size_x // 2:(-final_size_x + 1) // 2, final_size_y // 2:(-final_size_y + 1) // 2]) / 4


def padded_ifft_2x_2D(uhat: Array) -> Array:
    nx, ny = uhat.shape
    final_size_x, final_size_y = nx + 2 * (nx // 2), ny + 2 * (ny // 2)
    added_x, added_y = nx // 2, ny // 2
    smoothed = jnp.pad(jnp.fft.fftshift(uhat), ((added_x, added_x), (added_y, added_y)))
    assert smoothed.shape == (final_size_x, final_size_y), "incorrect padded shape"
    return jnp.fft.ifftn(jnp.fft.ifftshift(smoothed)) * 4


def truncated_fft_2x_3D(u: Array) -> Array:
    uhat = jnp.fft.fftshift(jnp.fft.fftn(u))
    kx, ky, kz = uhat.shape
    final_size_x, final_size_y, final_size_z = (kx + 1) // 2, (ky + 1) // 2, (kz + 1) // 2
    return jnp.fft.ifftshift(uhat[final_size_x // 2:(-final_size_x + 1) // 2, final_size_y // 2:(-final_size_y + 1) // 2, final_size_z // 2:(-final_size_z + 1) // 2]) / 8

def padded_ifft_2x_3D(uhat: Array) -> Array:
    nx, ny, nz = uhat.shape
    final_size_x, final_size_y, final_size_z = nx + 2 * (nx // 2), ny + 2 * (ny // 2), nz + 2 * (nz // 2)
    added_x, added_y, added_z = nx // 2, ny // 2, nz // 2
    smoothed = jnp.pad(jnp.fft.fftshift(uhat), ((added_x, added_x), (added_y, added_y), (added_z, added_z)))
    assert smoothed.shape == (final_size_x, final_size_y, final_size_z), "incorrect padded shape"
    return jnp.fft.ifftn(jnp.fft.ifftshift(smoothed)) * 8
