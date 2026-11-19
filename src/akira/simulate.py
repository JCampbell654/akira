"""Reproducible detector-image simulation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from akira.imaging import ImagingObservation, PSF, convolve_psf


def simulate_imaging(
    unconvolved_image: Array,
    psf: PSF,
    noise_sigma: float | Array,
    pixel_scale: float,
    seed: int,
    mask: Array | None = None,
) -> ImagingObservation:
    """Convolve a model image and add seeded independent Gaussian noise."""

    model = convolve_psf(
        jnp.asarray(unconvolved_image, dtype=jnp.float64), psf.kernel
    )
    noise_map = jnp.broadcast_to(
        jnp.asarray(noise_sigma, dtype=jnp.float64), model.shape
    )
    key = jax.random.PRNGKey(seed)
    noise = noise_map * jax.random.normal(key, shape=model.shape, dtype=jnp.float64)
    return ImagingObservation(
        data=model + noise,
        noise_map=noise_map,
        psf=psf,
        pixel_scale=pixel_scale,
        mask=mask,
    )

