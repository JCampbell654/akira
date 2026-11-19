"""Detector imaging, PSF convolution, and validated observations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.signal import convolve2d
import numpy as np


@dataclass(frozen=True, slots=True)
class PSF:
    """A normalized, odd-sized point-spread function kernel."""

    kernel: Array

    def __post_init__(self) -> None:
        kernel = np.asarray(self.kernel, dtype=np.float64)
        if kernel.ndim != 2 or any(size <= 0 for size in kernel.shape):
            raise ValueError("PSF kernel must be a non-empty two-dimensional array.")
        if any(size % 2 == 0 for size in kernel.shape):
            raise ValueError("PSF kernel dimensions must be odd.")
        if not np.all(np.isfinite(kernel)):
            raise ValueError("PSF kernel must contain only finite values.")
        if np.any(kernel < 0):
            raise ValueError("PSF kernel values must be non-negative.")
        if not np.isclose(kernel.sum(), 1.0, rtol=1e-12, atol=1e-14):
            raise ValueError("PSF kernel must sum to one.")
        object.__setattr__(self, "kernel", jnp.asarray(kernel, dtype=jnp.float64))

    @classmethod
    def delta(cls) -> "PSF":
        """Return a one-pixel identity PSF."""

        return cls(jnp.ones((1, 1), dtype=jnp.float64))


def gaussian_psf(sigma_pixels: float, size: int | None = None) -> PSF:
    """Construct a normalized circular Gaussian PSF.

    Args:
        sigma_pixels: Gaussian standard deviation in detector pixels.
        size: Odd kernel width. By default, extends at least four sigma from
            the centre on every side.
    """

    if not math.isfinite(sigma_pixels) or sigma_pixels <= 0:
        raise ValueError("sigma_pixels must be a positive finite value.")
    if size is None:
        radius = max(1, math.ceil(4.0 * sigma_pixels))
        size = 2 * radius + 1
    if size <= 0 or size % 2 == 0:
        raise ValueError("Gaussian PSF size must be a positive odd integer.")

    axis = jnp.arange(size, dtype=jnp.float64) - size // 2
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    kernel = jnp.exp(-0.5 * (xx**2 + yy**2) / sigma_pixels**2)
    kernel = kernel / kernel.sum()
    return PSF(kernel)


@jax.jit
def convolve_psf(image: Array, kernel: Array) -> Array:
    """Convolve an image with a PSF using zero-padded linear convolution."""

    return convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0.0)


@dataclass(frozen=True, slots=True)
class ImagingObservation:
    """A validated detector image and its statistical description.

    ``mask=True`` means that a pixel is excluded from likelihood evaluation.
    The full image is retained for rendering and diagnostics.
    """

    data: Array
    noise_map: Array
    psf: PSF
    pixel_scale: float
    mask: Array | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        noise_map = np.asarray(self.noise_map, dtype=np.float64)
        if data.ndim != 2 or data.size == 0:
            raise ValueError("Observation data must be a non-empty 2D image.")
        if noise_map.shape != data.shape:
            raise ValueError("noise_map must have the same shape as data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Observation data must contain only finite values.")
        if not np.all(np.isfinite(noise_map)) or np.any(noise_map <= 0):
            raise ValueError("noise_map must contain positive finite values.")
        if not math.isfinite(self.pixel_scale) or self.pixel_scale <= 0:
            raise ValueError("pixel_scale must be a positive finite value.")

        if self.mask is None:
            mask = np.zeros(data.shape, dtype=bool)
        else:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != data.shape:
                raise ValueError("mask must have the same shape as data.")
        if np.all(mask):
            raise ValueError("At least one observation pixel must be unmasked.")

        object.__setattr__(self, "data", jnp.asarray(data, dtype=jnp.float64))
        object.__setattr__(self, "noise_map", jnp.asarray(noise_map, dtype=jnp.float64))
        object.__setattr__(self, "mask", jnp.asarray(mask, dtype=bool))

