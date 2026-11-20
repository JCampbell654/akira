"""Bayesian optimization of quadratic source regularization strength."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy.optimize import minimize

from akira.imaging import ImagingObservation
from akira.pixelized import (
    PixelizedReconstruction,
    RegularizationKind,
    SourceGrid,
    reconstruct_from_operator,
    regularization_matrix,
    solve_source,
)


@dataclass(frozen=True, slots=True)
class RegularizationOptimization:
    """Evidence-optimal source regularization and Suyu diagnostics."""

    kind: str
    log_strength: float
    strength: float
    log_evidence: float
    effective_source_parameters: float
    stationarity_residual: float
    reconstruction: PixelizedReconstruction
    success: bool
    message: str
    evaluations: int


def gaussian_source_log_evidence(
    data: Array,
    noise: Array,
    operator: Array,
    regularization: Array,
    log_strength: Array | float,
) -> tuple[Array, Array, Array, Array]:
    """Return log evidence, source, system, and data curvature at log lambda."""

    strength = jnp.exp(jnp.asarray(log_strength, dtype=jnp.float64))
    source, data_curvature, system, _ = solve_source(
        operator, data, noise, regularization, strength
    )
    residual = data - operator @ source
    chi_squared = jnp.sum((residual / noise) ** 2)
    penalty = strength * source @ regularization @ source
    sign_h, logdet_h = jnp.linalg.slogdet(strength * regularization)
    sign_a, logdet_a = jnp.linalg.slogdet(system)
    logdet_noise = jnp.sum(jnp.log(noise**2))
    evidence = (
        -0.5 * (chi_squared + penalty)
        + 0.5 * logdet_h
        - 0.5 * logdet_a
        - 0.5 * logdet_noise
        - 0.5 * data.size * jnp.log(2.0 * jnp.pi)
    )
    evidence = jnp.where((sign_h > 0) & (sign_a > 0), evidence, -jnp.inf)
    return evidence, source, system, data_curvature


def optimize_regularization(
    observation: ImagingObservation,
    source_grid: SourceGrid,
    operator: Array,
    kind: RegularizationKind,
    initial_log_strength: float = 0.0,
    log_strength_bounds: tuple[float, float] = (-20.0, 30.0),
) -> RegularizationOptimization:
    """Maximize Gaussian source evidence in ``log(lambda)`` for a fixed lens."""

    lower, upper = log_strength_bounds
    if not all(math.isfinite(value) for value in (lower, upper, initial_log_strength)):
        raise ValueError("Log-strength values must be finite.")
    if not lower < initial_log_strength < upper:
        raise ValueError("initial_log_strength must lie strictly within its bounds.")
    expected_shape = (observation.data.size, source_grid.size)
    if operator.shape != expected_shape:
        raise ValueError(f"operator must have shape {expected_shape}.")
    regularization = regularization_matrix(source_grid, kind)
    eigenvalues = np.linalg.eigvalsh(np.asarray(regularization))
    tolerance = (
        np.max(np.abs(eigenvalues))
        * regularization.shape[0]
        * np.finfo(np.float64).eps
    )
    if eigenvalues.min() <= tolerance:
        raise ValueError("Evidence optimization requires positive-definite regularization.")
    active_indices = jnp.asarray(
        np.flatnonzero(~np.asarray(observation.mask).reshape(-1)), dtype=jnp.int32
    )
    active_operator = jnp.asarray(operator, dtype=jnp.float64)[active_indices]
    data = observation.data.reshape(-1)[active_indices]
    noise = observation.noise_map.reshape(-1)[active_indices]

    def negative_evidence(log_strength: Array) -> Array:
        evidence, _, _, _ = gaussian_source_log_evidence(
            data, noise, active_operator, regularization, log_strength[0]
        )
        return -evidence

    value_and_gradient = jax.jit(jax.value_and_grad(negative_evidence))

    def scipy_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_gradient(jnp.asarray(values, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=np.float64)

    optimized = minimize(
        scipy_objective,
        x0=np.asarray([initial_log_strength], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[log_strength_bounds],
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 200},
    )
    log_strength = float(optimized.x[0])
    strength = math.exp(log_strength)
    evidence, source, system, _ = gaussian_source_log_evidence(
        data, noise, active_operator, regularization, log_strength
    )
    inverse_system = jnp.linalg.inv(system)
    effective_parameters = source_grid.size - strength * jnp.trace(
        inverse_system @ regularization
    )
    stationarity_residual = (
        strength * source @ regularization @ source - effective_parameters
    )
    reconstruction = reconstruct_from_operator(
        observation, source_grid, operator, kind, strength
    )
    return RegularizationOptimization(
        kind=kind,
        log_strength=log_strength,
        strength=strength,
        log_evidence=float(evidence),
        effective_source_parameters=float(effective_parameters),
        stationarity_residual=float(stationarity_residual),
        reconstruction=reconstruction,
        success=bool(optimized.success),
        message=str(optimized.message),
        evaluations=int(optimized.nfev),
    )
