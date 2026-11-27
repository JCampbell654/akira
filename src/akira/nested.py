"""Nautilus nested sampling for pixelized strong-lens models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from akira.coordinates import CartesianGrid
from akira.imaging import ImagingObservation
from akira.inference import FreeParameter
from akira.pixelized import (
    PixelizedReconstruction,
    masked_forward_operator,
    reconstruct_from_masked_operator,
    regularization_matrix,
    solve_source,
)
from akira.pixelized_inference import (
    PixelizedInferenceResult,
    PixelizedLensModel,
    _log_source_evidence,
)


@dataclass(frozen=True, slots=True)
class NautilusSamplingResult:
    """Weighted posterior and representative reconstruction from Nautilus."""

    parameter_names: tuple[str, ...]
    points: np.ndarray
    log_weights: np.ndarray
    log_likelihood: np.ndarray
    log_evidence: float
    effective_sample_size: float
    evaluations: int
    success: bool
    maximum_posterior_index: int
    parameter_vector: np.ndarray
    parameters: dict[str, float]
    regularization_strength: float
    lens: object
    reconstruction: PixelizedReconstruction
    seed: int | None
    n_live: int
    f_live: float
    n_eff_target: float
    checkpoint_path: str | None
    point_estimate: PixelizedInferenceResult

    @property
    def weights(self) -> np.ndarray:
        """Normalized posterior weights."""

        weights = np.exp(self.log_weights - np.max(self.log_weights))
        return weights / weights.sum()

    @property
    def maximum_likelihood_index(self) -> int:
        return int(np.argmax(self.log_likelihood))


@dataclass(frozen=True, slots=True)
class _NautilusProblem:
    parameter_names: tuple[str, ...]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    log_likelihood: Callable[[np.ndarray], float]
    log_prior: Callable[[np.ndarray], float]
    components: Callable[[np.ndarray], tuple[object, float, Array, Array]]


def _make_nautilus_problem(
    model: PixelizedLensModel,
    observation: ImagingObservation,
    image_grid: CartesianGrid,
    sample_regularization_strength: bool,
    log_strength_bounds: tuple[float, float],
) -> _NautilusProblem:
    """Build unit-cube and marginalized-likelihood adapters for Nautilus."""

    if observation.data.shape != image_grid.shape:
        raise ValueError("Observation and image-grid shapes must match.")
    parameter_items = model.parameter_items
    if sample_regularization_strength and isinstance(
        model.regularization_strength, FreeParameter
    ):
        raise ValueError(
            "regularization_strength is already a free model parameter; do not "
            "also request log-strength sampling."
        )
    dimensions = len(parameter_items) + int(sample_regularization_strength)
    if dimensions == 0:
        raise ValueError("Nested sampling requires at least one free parameter.")
    if dimensions < 2:
        raise ValueError("Nautilus requires at least two free parameters.")
    lower_log, upper_log = map(float, log_strength_bounds)
    if sample_regularization_strength and not (
        math.isfinite(lower_log)
        and math.isfinite(upper_log)
        and lower_log < upper_log
    ):
        raise ValueError("log_strength_bounds must be finite and ordered.")

    regularization = regularization_matrix(
        model.source_grid, model.regularization_kind
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(regularization))
    tolerance = (
        np.max(np.abs(eigenvalues))
        * regularization.shape[0]
        * np.finfo(np.float64).eps
    )
    if eigenvalues.min() <= tolerance:
        raise ValueError(
            "Nested source evidence requires positive-definite regularization; "
            "use zeroth-order regularization."
        )

    active = ~np.asarray(observation.mask).reshape(-1)
    data = jnp.asarray(observation.data.reshape(-1)[active], dtype=jnp.float64)
    noise = jnp.asarray(observation.noise_map.reshape(-1)[active], dtype=jnp.float64)

    def evaluate(values: Array):
        model_values = values[:-1] if sample_regularization_strength else values
        lens, strength = model.build(model_values)
        if sample_regularization_strength:
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
        evidence = _log_source_evidence(
            data, noise, operator, source, system, regularization, strength
        )
        return evidence, operator, strength

    evaluate_jit = jax.jit(evaluate)

    def prior_transform(unit_values: np.ndarray) -> np.ndarray:
        units = np.asarray(unit_values, dtype=np.float64)
        if units.shape != (dimensions,):
            raise ValueError(f"Unit-cube point must have shape ({dimensions},).")
        # Avoid infinities for unbounded inverse-CDF priors at exact endpoints.
        units = np.clip(
            units,
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
        values = [
            float(parameter.prior.from_unit(unit))
            for unit, (_, parameter) in zip(units, parameter_items)
        ]
        if sample_regularization_strength:
            values.append(lower_log + units[-1] * (upper_log - lower_log))
        return np.asarray(values, dtype=np.float64)

    def log_likelihood(values: np.ndarray) -> float:
        evidence, _, _ = evaluate_jit(
            jnp.asarray(values, dtype=jnp.float64)
        )
        return float(evidence)

    def log_prior(values: np.ndarray) -> float:
        model_values = values[:-1] if sample_regularization_strength else values
        value = float(model.log_prior(jnp.asarray(model_values, dtype=jnp.float64)))
        if sample_regularization_strength:
            log_strength = float(values[-1])
            if not lower_log <= log_strength <= upper_log:
                return -math.inf
            value -= math.log(upper_log - lower_log)
        return value

    def components(values: np.ndarray):
        array = jnp.asarray(values, dtype=jnp.float64)
        model_values = array[:-1] if sample_regularization_strength else array
        lens, _ = model.build(model_values)
        evidence, operator, strength = evaluate_jit(array)
        return lens, float(strength), evidence, operator

    names = model.parameter_names
    if sample_regularization_strength:
        names += ("regularization.log_strength",)
    return _NautilusProblem(
        parameter_names=names,
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        components=components,
    )


def sample_pixelized_model_nautilus(
    model: PixelizedLensModel,
    observation: ImagingObservation,
    image_grid: CartesianGrid,
    *,
    sample_regularization_strength: bool = False,
    log_strength_bounds: tuple[float, float] = (-20.0, 30.0),
    n_live: int = 1000,
    n_update: int | None = None,
    n_networks: int = 4,
    n_batch: int | None = None,
    f_live: float = 0.01,
    n_eff: float = 10000,
    n_like_max: int | None = None,
    timeout: float | None = None,
    discard_exploration: bool = True,
    seed: int | None = 0,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    verbose: bool = False,
) -> NautilusSamplingResult:
    """Run Nautilus on the source-marginalized nonlinear likelihood.

    Nonlinear priors are applied exclusively through the unit-cube transform.
    The returned ``log_evidence`` therefore marginalizes both the linear source
    and all sampled nonlinear parameters.
    """

    try:
        from nautilus import Sampler
    except ImportError as error:
        raise ImportError(
            "Nautilus sampling requires the 'nested' extra: "
            "pip install -e '.[nested]'"
        ) from error

    if n_live < 1 or n_networks < 1:
        raise ValueError("n_live and n_networks must be positive.")
    if not 0.0 < f_live < 1.0:
        raise ValueError("f_live must lie strictly between zero and one.")
    if n_eff <= 0:
        raise ValueError("n_eff must be positive.")
    if n_like_max is not None and n_like_max < 1:
        raise ValueError("n_like_max must be positive when provided.")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive when provided.")

    problem = _make_nautilus_problem(
        model,
        observation,
        image_grid,
        sample_regularization_strength,
        log_strength_bounds,
    )
    evaluations = 0

    def counted_likelihood(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return problem.log_likelihood(values)

    checkpoint = None if checkpoint_path is None else str(Path(checkpoint_path))
    sampler = Sampler(
        problem.prior_transform,
        counted_likelihood,
        n_dim=len(problem.parameter_names),
        n_live=n_live,
        n_update=n_update,
        n_networks=n_networks,
        n_batch=n_batch,
        pass_dict=False,
        seed=seed,
        filepath=checkpoint,
        resume=resume,
    )
    run_options = {
        "f_live": f_live,
        "n_eff": n_eff,
        "discard_exploration": discard_exploration,
        "verbose": verbose,
    }
    if n_like_max is not None:
        run_options["n_like_max"] = n_like_max
    if timeout is not None:
        run_options["timeout"] = timeout
    success = bool(sampler.run(**run_options))
    total_evaluations = int(getattr(sampler, "n_like", evaluations))
    points, log_weights, log_likelihood = sampler.posterior(
        return_as_dict=False, equal_weight=False
    )
    points = np.asarray(points, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(problem.parameter_names):
        raise RuntimeError("Nautilus returned posterior points with an invalid shape.")
    if points.shape[0] == 0:
        raise RuntimeError("Nautilus returned an empty posterior.")

    log_posterior = log_likelihood + np.asarray(
        [problem.log_prior(point) for point in points]
    )
    map_index = int(np.argmax(log_posterior))
    parameter_vector = points[map_index].copy()
    lens, strength, _, operator = problem.components(parameter_vector)
    reconstruction = reconstruct_from_masked_operator(
        observation,
        model.source_grid,
        operator,
        model.regularization_kind,
        strength,
    )
    parameters = dict(zip(problem.parameter_names, map(float, parameter_vector)))
    parameters["regularization.strength"] = strength
    point_log_prior = problem.log_prior(parameter_vector)
    active = ~np.asarray(observation.mask).reshape(-1)
    active_noise = np.asarray(observation.noise_map).reshape(-1)[active]
    image_log_likelihood = -0.5 * (
        reconstruction.chi_squared
        + np.sum(np.log(2.0 * np.pi * active_noise**2))
    )
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    covariance = np.cov(points, rowvar=False, aweights=weights)
    covariance = np.atleast_2d(covariance)
    point_estimate = PixelizedInferenceResult(
        parameter_names=problem.parameter_names,
        parameter_vector=jnp.asarray(parameter_vector, dtype=jnp.float64),
        parameters=parameters,
        lens=lens,
        reconstruction=reconstruction,
        log_likelihood=float(image_log_likelihood),
        log_prior=point_log_prior,
        log_source_evidence=float(log_likelihood[map_index]),
        objective_value=float(log_posterior[map_index]),
        objective="evidence",
        success=success,
        message=(
            "Nautilus converged."
            if success
            else "Nautilus stopped before satisfying its convergence targets."
        ),
        evaluations=total_evaluations,
        covariance=jnp.asarray(covariance, dtype=jnp.float64),
        regularization_strength=strength,
        effective_source_parameters=None,
        stationarity_residual=None,
        optimized_regularization_strength=sample_regularization_strength,
        log_strength_bounds=(
            tuple(map(float, log_strength_bounds))
            if sample_regularization_strength
            else None
        ),
    )
    return NautilusSamplingResult(
        parameter_names=problem.parameter_names,
        points=points,
        log_weights=log_weights,
        log_likelihood=log_likelihood,
        log_evidence=float(sampler.log_z),
        effective_sample_size=float(sampler.n_eff),
        evaluations=total_evaluations,
        success=success,
        maximum_posterior_index=map_index,
        parameter_vector=parameter_vector,
        parameters=parameters,
        regularization_strength=strength,
        lens=lens,
        reconstruction=reconstruction,
        seed=seed,
        n_live=n_live,
        f_live=f_live,
        n_eff_target=n_eff,
        checkpoint_path=checkpoint,
        point_estimate=point_estimate,
    )
