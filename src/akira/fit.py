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
from akira.profiles import EllipticalPowerLaw, ExternalShear, Gaussian, Sersic
from akira.render import (
    render_epl_sersic_from_coordinates,
    render_sis_gaussian_from_coordinates,
)


@dataclass(frozen=True, slots=True)
class FitResult:
    """Result of fitting an SIS Einstein radius with a fixed Gaussian source."""

    einstein_radius: float
    log_likelihood: float
    model_image: Array
    success: bool
    message: str
    evaluations: int


@dataclass(frozen=True, slots=True)
class EPLSersicParameters:
    """The four free parameters in the controlled analytic recovery problem."""

    einstein_radius: float
    slope: float
    source_x: float
    source_y: float


@dataclass(frozen=True, slots=True)
class EPLSersicFitResult:
    """Result of the controlled four-parameter EPL/Sérsic fit."""

    parameters: EPLSersicParameters
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


def fit_epl_sersic_parameters(
    observation: ImagingObservation,
    grid: CartesianGrid,
    lens_template: EllipticalPowerLaw,
    source_template: Sersic,
    initial: EPLSersicParameters,
    shear: ExternalShear | None = None,
    bounds: tuple[tuple[float, float], ...] = (
        (1e-4, 10.0),
        (1.01, 2.99),
        (-5.0, 5.0),
        (-5.0, 5.0),
    ),
) -> EPLSersicFitResult:
    """Fit Einstein radius, 3D slope, and two source-position coordinates.

    All lens shape, shear, and Sérsic structure parameters are held fixed. This
    deliberately constrained fit validates multi-parameter gradients and the
    composable analytic architecture before a general parameter system is added.
    """

    if observation.data.shape != grid.shape:
        raise ValueError("Observation and grid shapes must match.")
    if len(bounds) != 4:
        raise ValueError("bounds must contain four (lower, upper) pairs.")

    initial_values = np.asarray(
        [initial.einstein_radius, initial.slope, initial.source_x, initial.source_y],
        dtype=np.float64,
    )
    if any(not lower < value < upper for value, (lower, upper) in zip(initial_values, bounds)):
        raise ValueError("Every initial parameter must lie strictly within its bounds.")

    coordinates = grid.subpixel_coordinates()
    lens_centre = jnp.asarray(lens_template.centre, dtype=jnp.float64)
    shear = shear if shear is not None else ExternalShear(0.0)

    def model_image(values: Array) -> Array:
        unconvolved = render_epl_sersic_from_coordinates(
            coordinates,
            values[0],
            values[1],
            lens_centre,
            lens_template.q,
            lens_template.phi,
            shear.strength,
            shear.phi,
            source_template.intensity,
            source_template.effective_radius,
            source_template.sersic_index,
            values[2:4],
            source_template.q,
            source_template.phi,
        )
        return convolve_psf(unconvolved, observation.psf.kernel)

    def negative_log_likelihood(values: Array) -> Array:
        model = model_image(values)
        return -gaussian_log_likelihood(
            observation.data, model, observation.noise_map, observation.mask
        )

    value_and_gradient = jax.jit(jax.value_and_grad(negative_log_likelihood))

    def scipy_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_gradient(jnp.asarray(values, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=np.float64)

    optimized = minimize(
        scipy_objective,
        x0=initial_values,
        jac=True,
        method="L-BFGS-B",
        bounds=list(bounds),
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    fitted_values = jnp.asarray(optimized.x, dtype=jnp.float64)
    fitted_model = model_image(fitted_values)
    fitted_log_likelihood = gaussian_log_likelihood(
        observation.data,
        fitted_model,
        observation.noise_map,
        observation.mask,
    )
    parameters = EPLSersicParameters(*map(float, optimized.x))
    return EPLSersicFitResult(
        parameters=parameters,
        log_likelihood=float(fitted_log_likelihood),
        model_image=fitted_model,
        success=bool(optimized.success),
        message=str(optimized.message),
        evaluations=int(optimized.nfev),
    )
