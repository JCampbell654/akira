"""Image rendering for the verified SIS-plus-Gaussian vertical slice."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from akira.coordinates import CartesianGrid
from akira.lensing import ray_trace
from akira.profiles import Gaussian, SIS, gaussian_brightness, sis_deflection


@jax.jit
def render_sis_gaussian_from_coordinates(
    subpixel_coordinates: Array,
    einstein_radius: float,
    lens_centre: Array,
    source_intensity: float,
    source_sigma: float,
    source_centre: Array,
    source_q: float,
    source_phi: float,
) -> Array:
    """JIT-compatible SIS-plus-Gaussian renderer.

    ``subpixel_coordinates`` has shape ``(ny, nx, n_subpixels, 2)``. The
    returned image is the mean surface brightness within each image pixel.
    """

    deflections = sis_deflection(
        subpixel_coordinates, einstein_radius, lens_centre
    )
    source_coordinates = ray_trace(subpixel_coordinates, deflections)
    subpixel_brightness = gaussian_brightness(
        source_coordinates,
        source_intensity,
        source_sigma,
        source_centre,
        source_q,
        source_phi,
    )
    return subpixel_brightness.mean(axis=2)


def render_sis_gaussian(grid: CartesianGrid, lens: SIS, source: Gaussian) -> Array:
    """Render a lensed Gaussian source through an SIS mass profile."""

    return render_sis_gaussian_from_coordinates(
        grid.subpixel_coordinates(),
        lens.einstein_radius,
        lens_centre=jnp.asarray(lens.centre, dtype=jnp.float64),
        source_intensity=source.intensity,
        source_sigma=source.sigma,
        source_centre=jnp.asarray(source.centre, dtype=jnp.float64),
        source_q=source.q,
        source_phi=source.phi,
    )
