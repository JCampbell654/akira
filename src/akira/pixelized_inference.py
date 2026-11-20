"""Nonlinear lens inference with an analytically solved pixelized source."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy.optimize import minimize

from akira.coordinates import CartesianGrid
from akira.imaging import ImagingObservation
from akira.inference import FreeParameter, ProfileModel
from akira.model import LensPlane
from akira.pixelized import (
    PixelizedReconstruction,
    RegularizationKind,
    SourceGrid,
    masked_forward_operator,
    reconstruct_from_masked_operator,
    regularization_matrix,
    solve_source,
)


PixelizedObjective = Literal["evidence", "penalized", "profile_likelihood"]


@dataclass(frozen=True, slots=True)
class OptimizationProgress:
    """A completed nonlinear objective evaluation during optimization."""

    evaluation: int
    objective_value: float
    best_objective_value: float
    gradient_norm: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PixelizedLensModel:
    """Free/fixed nonlinear lens parameters plus a linear source model."""

    lens: tuple[ProfileModel, ...]
    source_grid: SourceGrid
    regularization_kind: RegularizationKind = "zeroth"
    regularization_strength: float | FreeParameter = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lens", tuple(self.lens))
        if not self.lens:
            raise ValueError("PixelizedLensModel requires at least one lens profile.")
        regularization_matrix(self.source_grid, self.regularization_kind)
        strength = self.regularization_strength
        initial = strength.initial if isinstance(strength, FreeParameter) else strength
        if not math.isfinite(initial) or initial <= 0:
            raise ValueError("regularization_strength must be positive and finite.")

    @property
    def parameter_items(self) -> tuple[tuple[str, FreeParameter], ...]:
        output: list[tuple[str, FreeParameter]] = []
        for index, profile in enumerate(self.lens):
            profile.collect(f"lens.{index}.{profile.profile_type.__name__}", output)
        if isinstance(self.regularization_strength, FreeParameter):
            output.append(("regularization.strength", self.regularization_strength))
        return tuple(output)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.parameter_items)

    @property
    def initial_vector(self) -> Array:
        return jnp.asarray(
            [parameter.initial for _, parameter in self.parameter_items],
            dtype=jnp.float64,
        )

    def log_prior(self, vector: Array) -> Array:
        return sum(
            (
                parameter.prior.log_prob(value)
                for value, (_, parameter) in zip(vector, self.parameter_items)
            ),
            start=jnp.asarray(0.0, dtype=jnp.float64),
        )

    def build(self, vector: Array) -> tuple[LensPlane, Array]:
        cursor = [0]
        lens = LensPlane(tuple(profile.build(vector, cursor) for profile in self.lens))
        if isinstance(self.regularization_strength, FreeParameter):
            strength = vector[cursor[0]]
            cursor[0] += 1
        else:
            strength = jnp.asarray(self.regularization_strength, dtype=jnp.float64)
        return lens, strength


@dataclass(frozen=True, slots=True)
class PixelizedInferenceResult:
    """Best nonlinear lens and corresponding linear source reconstruction."""

    parameter_names: tuple[str, ...]
    parameter_vector: Array
    parameters: dict[str, float]
    lens: LensPlane
    reconstruction: PixelizedReconstruction
    log_likelihood: float
    log_prior: float
    log_source_evidence: float | None
    objective_value: float
    objective: str
    success: bool
    message: str
    evaluations: int
    covariance: Array | None
    regularization_strength: float
    effective_source_parameters: float | None
    stationarity_residual: float | None


def _log_source_evidence(
    data: Array,
    noise: Array,
    operator: Array,
    source: Array,
    system: Array,
    regularization: Array,
    strength: Array,
) -> Array:
    residual = data - operator @ source
    chi_squared = jnp.sum((residual / noise) ** 2)
    penalty = strength * source @ regularization @ source
    _, logdet_prior = jnp.linalg.slogdet(strength * regularization)
    _, logdet_system = jnp.linalg.slogdet(system)
    logdet_noise = jnp.sum(jnp.log(noise**2))
    return (
        -0.5 * (chi_squared + penalty)
        + 0.5 * logdet_prior
        - 0.5 * logdet_system
        - 0.5 * logdet_noise
        - 0.5 * data.size * jnp.log(2.0 * jnp.pi)
    )


def fit_pixelized_model(
    model: PixelizedLensModel,
    observation: ImagingObservation,
    image_grid: CartesianGrid,
    objective: PixelizedObjective = "evidence",
    optimize_regularization_strength: bool = False,
    log_strength_bounds: tuple[float, float] = (-20.0, 30.0),
    progress_callback: Callable[[OptimizationProgress], None] | None = None,
    max_iterations: int = 500,
    max_line_search_steps: int = 20,
) -> PixelizedInferenceResult:
    """Optimize nonlinear lens parameters while solving source pixels linearly.

    Nonlinear parameter priors are included for all three objectives, so the
    returned optimum is a MAP estimate. ``evidence`` additionally marginalizes
    the Gaussian linear source exactly.
    """

    if observation.data.shape != image_grid.shape:
        raise ValueError("Observation and image-grid shapes must match.")
    if max_iterations < 1 or max_line_search_steps < 1:
        raise ValueError("Optimizer iteration and line-search limits must be positive.")
    if objective not in ("evidence", "penalized", "profile_likelihood"):
        raise ValueError(
            "objective must be 'evidence', 'penalized', or 'profile_likelihood'."
        )
    parameter_items = model.parameter_items
    if not parameter_items and not optimize_regularization_strength:
        raise ValueError("PixelizedLensModel contains no free parameters.")
    if optimize_regularization_strength:
        if objective != "evidence":
            raise ValueError("Regularization optimization requires objective='evidence'.")
        if isinstance(model.regularization_strength, FreeParameter):
            raise ValueError(
                "Use a fixed positive initial regularization_strength when "
                "optimizing it in log space."
            )
        lower_log, upper_log = log_strength_bounds
        initial_log = math.log(model.regularization_strength)
        if not lower_log < initial_log < upper_log:
            raise ValueError("Initial log regularization strength lies outside bounds.")
    regularization = regularization_matrix(
        model.source_grid, model.regularization_kind
    )
    if objective == "evidence":
        eigenvalues = np.linalg.eigvalsh(np.asarray(regularization))
        tolerance = (
            np.max(np.abs(eigenvalues))
            * regularization.shape[0]
            * np.finfo(np.float64).eps
        )
        if eigenvalues.min() <= tolerance:
            raise ValueError(
                "Evidence requires positive-definite regularization; use zeroth "
                "order or choose a non-evidence objective."
            )
    active = ~observation.mask.reshape(-1)
    data = observation.data.reshape(-1)[active]
    noise = observation.noise_map.reshape(-1)[active]

    def components(values: Array):
        model_values = values[:-1] if optimize_regularization_strength else values
        lens, strength = model.build(model_values)
        if optimize_regularization_strength:
            strength = jnp.exp(values[-1])
        operator = masked_forward_operator(
            image_grid,
            model.source_grid,
            lens,
            observation.psf.kernel,
            observation.mask,
        )
        source, _, system, _ = solve_source(
            operator, data, noise, regularization, strength
        )
        residual = data - operator @ source
        log_likelihood = -0.5 * jnp.sum(
            (residual / noise) ** 2 + jnp.log(2.0 * jnp.pi * noise**2)
        )
        penalty = strength * source @ regularization @ source
        evidence = _log_source_evidence(
            data,
            noise,
            operator,
            source,
            system,
            regularization,
            strength,
        )
        log_prior = model.log_prior(model_values)
        return operator, source, log_likelihood, penalty, evidence, log_prior

    def negative_target(values: Array) -> Array:
        _, _, log_likelihood, penalty, evidence, log_prior = components(values)
        if objective == "evidence":
            target = evidence
        elif objective == "penalized":
            target = log_likelihood - 0.5 * penalty
        else:
            target = log_likelihood
        return -(target + log_prior)

    value_and_gradient = jax.jit(jax.value_and_grad(negative_target))

    evaluation_count = 0
    best_objective = math.inf
    start_time = time.monotonic()

    def scipy_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluation_count, best_objective
        value, gradient = value_and_gradient(jnp.asarray(values, dtype=jnp.float64))
        float_value = float(value)
        numpy_gradient = np.asarray(gradient, dtype=np.float64)
        evaluation_count += 1
        best_objective = min(best_objective, float_value)
        if progress_callback is not None:
            progress_callback(
                OptimizationProgress(
                    evaluation=evaluation_count,
                    objective_value=float_value,
                    best_objective_value=best_objective,
                    gradient_norm=float(np.linalg.norm(numpy_gradient)),
                    elapsed_seconds=time.monotonic() - start_time,
                )
            )
        return float_value, numpy_gradient

    initial_values = np.asarray(model.initial_vector, dtype=np.float64)
    optimization_bounds = [
        parameter.prior.bounds for _, parameter in parameter_items
    ]
    optimization_names = model.parameter_names
    if optimize_regularization_strength:
        initial_values = np.concatenate((initial_values, [initial_log]))
        optimization_bounds.append(log_strength_bounds)
        optimization_names = optimization_names + ("regularization.log_strength",)
    optimized = minimize(
        scipy_objective,
        x0=initial_values,
        jac=True,
        method="L-BFGS-B",
        bounds=optimization_bounds,
        options={
            "ftol": 1e-11,
            "gtol": 1e-7,
            "maxiter": max_iterations,
            "maxls": max_line_search_steps,
        },
    )
    fitted = jnp.asarray(optimized.x, dtype=jnp.float64)
    model_fitted = fitted[:-1] if optimize_regularization_strength else fitted
    lens, strength = model.build(model_fitted)
    if optimize_regularization_strength:
        strength = jnp.exp(fitted[-1])
    operator, _, log_likelihood, _, evidence, log_prior = components(fitted)
    reconstruction = reconstruct_from_masked_operator(
        observation,
        model.source_grid,
        operator,
        model.regularization_kind,
        float(strength),
    )
    covariance = None
    if hasattr(optimized.hess_inv, "todense"):
        covariance = jnp.asarray(optimized.hess_inv.todense(), dtype=jnp.float64)
    if objective == "evidence":
        source_objective = evidence
    elif objective == "penalized":
        source_objective = log_likelihood - 0.5 * (
            strength
            * reconstruction.source.reshape(-1)
            @ regularization
            @ reconstruction.source.reshape(-1)
        )
    else:
        source_objective = log_likelihood
    effective_parameters = None
    stationarity_residual = None
    if objective == "evidence":
        inverse_system = reconstruction.posterior_covariance
        source_vector = reconstruction.source.reshape(-1)
        effective_parameters_array = model.source_grid.size - strength * jnp.trace(
            inverse_system @ regularization
        )
        stationarity_array = (
            strength * source_vector @ regularization @ source_vector
            - effective_parameters_array
        )
        effective_parameters = float(effective_parameters_array)
        stationarity_residual = float(stationarity_array)
    parameter_values = dict(zip(optimization_names, map(float, optimized.x)))
    parameter_values["regularization.strength"] = float(strength)
    return PixelizedInferenceResult(
        parameter_names=optimization_names,
        parameter_vector=fitted,
        parameters=parameter_values,
        lens=lens,
        reconstruction=reconstruction,
        log_likelihood=float(log_likelihood),
        log_prior=float(log_prior),
        log_source_evidence=float(evidence) if objective == "evidence" else None,
        objective_value=float(source_objective + log_prior),
        objective=objective,
        success=bool(optimized.success),
        message=str(optimized.message),
        evaluations=int(optimized.nfev),
        covariance=covariance,
        regularization_strength=float(strength),
        effective_source_parameters=effective_parameters,
        stationarity_residual=stationarity_residual,
    )
