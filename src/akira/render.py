"""Image rendering for the verified SIS-plus-Gaussian vertical slice."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from akira.coordinates import CartesianGrid
from akira.lensing import ray_trace
from akira.model import CompositeLight, LensPlane
from akira.profiles import (
    Gaussian,
    SIS,
    epl_deflection,
    external_shear_deflection,
    gaussian_brightness,
    sersic_brightness,
    sis_deflection,
)


@jax.jit
def render_from_coordinates(
    subpixel_coordinates: Array,
    lens: LensPlane,
    source: CompositeLight,
) -> Array:
    """Render any composable single-plane lens and source model."""

    deflections = lens.deflection(subpixel_coordinates)
    source_coordinates = ray_trace(subpixel_coordinates, deflections)
    return source.brightness(source_coordinates).mean(axis=2)


def render(
    grid: CartesianGrid,
    lens: LensPlane,
    source: CompositeLight,
) -> Array:
    """Render a composable lens system on a Cartesian detector grid."""

    return render_from_coordinates(grid.subpixel_coordinates(), lens, source)


@jax.jit
def render_epl_sersic_from_coordinates(
    subpixel_coordinates: Array,
    einstein_radius: float,
    slope: float,
    lens_centre: Array,
    lens_q: float,
    lens_phi: float,
    shear_strength: float,
    shear_phi: float,
    source_intensity: float,
    source_effective_radius: float,
    source_sersic_index: float,
    source_centre: Array,
    source_q: float,
    source_phi: float,
) -> Array:
    """Differentiable EPL + external-shear + Sérsic rendering kernel."""

    deflections = epl_deflection(
        subpixel_coordinates,
        einstein_radius,
        slope,
        lens_centre,
        lens_q,
        lens_phi,
    )
    deflections = deflections + external_shear_deflection(
        subpixel_coordinates, shear_strength, shear_phi, lens_centre
    )
    source_coordinates = ray_trace(subpixel_coordinates, deflections)
    brightness = sersic_brightness(
        source_coordinates,
        source_intensity,
        source_effective_radius,
        source_sersic_index,
        source_centre,
        source_q,
        source_phi,
    )
    return brightness.mean(axis=2)


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
