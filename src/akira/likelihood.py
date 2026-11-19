"""Image-plane likelihoods and diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def normalized_residuals(data: Array, model: Array, noise_map: Array) -> Array:
    """Return ``(data - model) / noise`` for every image pixel."""

    return (data - model) / noise_map


@jax.jit
def gaussian_log_likelihood(
    data: Array,
    model: Array,
    noise_map: Array,
    mask: Array,
) -> Array:
    """Return the normalized independent-Gaussian image log likelihood.

    Pixels for which ``mask`` is true contribute exactly zero.
    """

    residual = normalized_residuals(data, model, noise_map)
    per_pixel = residual**2 + jnp.log(2.0 * jnp.pi * noise_map**2)
    return -0.5 * jnp.where(mask, 0.0, per_pixel).sum()


@jax.jit
def chi_squared(data: Array, model: Array, noise_map: Array, mask: Array) -> Array:
    """Return image-plane chi-squared over unmasked pixels."""

    residual = normalized_residuals(data, model, noise_map)
    return jnp.where(mask, 0.0, residual**2).sum()

