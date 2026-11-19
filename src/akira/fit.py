"""Initial parameter-fitting interface for the verified vertical slice."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy.optimize import minimize

from akira.coordinates import CartesianGrid
from akira.imaging import ImagingObservation, convolve_psf
from akira.likelihood import gaussian_log_likelihood
from akira.profiles import Gaussian
from akira.render import render_sis_gaussian_from_coordinates


@dataclass(frozen=True, slots=True)
class FitResult:
    """Result of fitting an SIS Einstein radius with a fixed Gaussian source."""

    einstein_radius: float
    log_likelihood: float
    model_image: Array
    success: bool
    message: str
    evaluations: int


def fit_sis_einstein_radius(
    observation: ImagingObservation,
    grid: CartesianGrid,
    source: Gaussian,
    initial_einstein_radius: float,
    lens_centre: tuple[float, float] = (0.0, 0.0),
    bounds: tuple[float, float] = (1e-6, 10.0),
) -> FitResult:
    """Fit an SIS Einstein radius while holding source parameters fixed."""

    if observation.data.shape != grid.shape:
        raise ValueError("Observation and grid shapes must match.")
    if not bounds[0] < initial_einstein_radius < bounds[1]:
        raise ValueError("initial_einstein_radius must lie strictly within bounds.")

    coordinates = grid.subpixel_coordinates()
    lens_centre_array = jnp.asarray(lens_centre, dtype=jnp.float64)
    source_centre = jnp.asarray(source.centre, dtype=jnp.float64)

    def model_image(einstein_radius: Array) -> Array:
        unconvolved = render_sis_gaussian_from_coordinates(
            coordinates,
            einstein_radius,
            lens_centre_array,
            source.intensity,
            source.sigma,
            source_centre,
            source.q,
            source.phi,
        )
        return convolve_psf(unconvolved, observation.psf.kernel)

    def negative_log_likelihood(einstein_radius: Array) -> Array:
        model = model_image(einstein_radius)
        return -gaussian_log_likelihood(
            observation.data, model, observation.noise_map, observation.mask
        )

    value_and_gradient = jax.jit(jax.value_and_grad(negative_log_likelihood))

    def scipy_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_gradient(jnp.asarray(values[0], dtype=jnp.float64))
        return float(value), np.asarray([gradient], dtype=np.float64)

    optimized = minimize(
        scipy_objective,
        x0=np.asarray([initial_einstein_radius], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[bounds],
    )
    fitted_radius = float(optimized.x[0])
    fitted_model = model_image(jnp.asarray(fitted_radius, dtype=jnp.float64))
    fitted_log_likelihood = gaussian_log_likelihood(
        observation.data,
        fitted_model,
        observation.noise_map,
        observation.mask,
    )
    return FitResult(
        einstein_radius=fitted_radius,
        log_likelihood=float(fitted_log_likelihood),
        model_image=fitted_model,
        success=bool(optimized.success),
        message=str(optimized.message),
        evaluations=int(optimized.nfev),
    )

