"""Posterior sampling for pixelized strong-lens models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from akira.coordinates import CartesianGrid
from akira.imaging import ImagingObservation
from akira.pixelized import masked_forward_operator, regularization_matrix, solve_source
from akira.pixelized_inference import PixelizedLensModel, _log_source_evidence


@dataclass(frozen=True, slots=True)
class SamplingProgress:
    """Progress reported after a completed ensemble step."""

    step: int
    total_steps: int
    mean_acceptance_fraction: float


@dataclass(frozen=True, slots=True)
class PixelizedSamplingResult:
    """Saved affine-invariant ensemble samples and diagnostics."""

    parameter_names: tuple[str, ...]
    chain: np.ndarray
    log_probability: np.ndarray
    acceptance_fraction: np.ndarray
    burn_in: int
    thin: int
    seed: int

    @property
    def samples(self) -> np.ndarray:
        """Return the saved chain flattened over steps and walkers."""

        return self.chain.reshape((-1, self.chain.shape[-1]))


def _initial_walkers(
    centre: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    n_walkers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    walkers = np.empty((n_walkers, centre.size), dtype=np.float64)
    for dimension, (value, (lower, upper)) in enumerate(zip(centre, bounds)):
        if lower is not None and upper is not None:
            scale = max((upper - lower) * 0.01, np.finfo(float).eps)
        else:
            scale = max(abs(value) * 0.01, 1.0e-3)
        proposed = value + rng.normal(0.0, scale, n_walkers)
        if lower is not None:
            proposed = np.maximum(proposed, np.nextafter(lower, math.inf))
        if upper is not None:
            proposed = np.minimum(proposed, np.nextafter(upper, -math.inf))
        walkers[:, dimension] = proposed
    # A boundary MAP can otherwise collapse multiple clipped walkers together.
    for index in range(1, n_walkers):
        if np.array_equal(walkers[index], walkers[0]):
            walkers[index] = _initial_walkers(centre, bounds, 1, rng)[0]
    return walkers


def sample_pixelized_model(
    model: PixelizedLensModel,
    observation: ImagingObservation,
    image_grid: CartesianGrid,
    *,
    initial_vector: Array | np.ndarray | None = None,
    sample_regularization_strength: bool = False,
    log_strength_bounds: tuple[float, float] = (-20.0, 30.0),
    n_walkers: int = 32,
    n_steps: int = 1000,
    burn_in: int = 200,
    thin: int = 1,
    stretch_scale: float = 2.0,
    seed: int = 0,
    progress_callback: Callable[[SamplingProgress], None] | None = None,
) -> PixelizedSamplingResult:
    """Sample the source-marginalized posterior with an ensemble sampler.

    The linear source pixels are integrated out with the same Gaussian
    evidence used by :func:`fit_pixelized_model`. When requested, the natural
    logarithm of the regularization strength is sampled with a uniform prior
    over ``log_strength_bounds``.
    """

    if observation.data.shape != image_grid.shape:
        raise ValueError("Observation and image-grid shapes must match.")
    parameter_items = model.parameter_items
    dimensions = len(parameter_items) + int(sample_regularization_strength)
    if dimensions == 0:
        raise ValueError("Sampling requires at least one free parameter.")
    if n_walkers < 2 * dimensions or n_walkers % 2:
        raise ValueError("n_walkers must be even and at least twice the dimension.")
    if n_steps < 1 or burn_in < 0 or burn_in >= n_steps:
        raise ValueError("Require n_steps > burn_in >= 0.")
    if thin < 1 or stretch_scale <= 1.0:
        raise ValueError("thin must be positive and stretch_scale must exceed one.")
    lower_log, upper_log = map(float, log_strength_bounds)
    if sample_regularization_strength and not lower_log < upper_log:
        raise ValueError("log_strength_bounds must be finite and ordered.")

    regularization = regularization_matrix(model.source_grid, model.regularization_kind)
    active = ~np.asarray(observation.mask).reshape(-1)
    data = jnp.asarray(observation.data.reshape(-1)[active], dtype=jnp.float64)
    noise = jnp.asarray(observation.noise_map.reshape(-1)[active], dtype=jnp.float64)

    def log_probability(values: Array) -> Array:
        model_values = values[:-1] if sample_regularization_strength else values
        lens, strength = model.build(model_values)
        if sample_regularization_strength:
            strength = jnp.exp(values[-1])
        operator = masked_forward_operator(
            image_grid, model.source_grid, lens, observation.psf.kernel, observation.mask
        )
        source, _, system, _ = solve_source(
            operator, data, noise, regularization, strength
        )
        evidence = _log_source_evidence(
            data, noise, operator, source, system, regularization, strength
        )
        log_prior = model.log_prior(model_values)
        if sample_regularization_strength:
            inside = (values[-1] >= lower_log) & (values[-1] <= upper_log)
            log_prior = jnp.where(inside, log_prior, -jnp.inf)
        return evidence + log_prior

    evaluate = jax.jit(log_probability)
    names = model.parameter_names
    bounds: list[tuple[float | None, float | None]] = [
        item.prior.bounds for _, item in parameter_items
    ]
    default_centre = np.asarray(model.initial_vector, dtype=np.float64)
    if sample_regularization_strength:
        names += ("regularization.log_strength",)
        bounds.append((lower_log, upper_log))
        default_centre = np.concatenate(
            (default_centre, [math.log(float(model.regularization_strength))])
        )
    centre = default_centre if initial_vector is None else np.asarray(initial_vector, float)
    if centre.shape != (dimensions,):
        raise ValueError(f"initial_vector must have shape ({dimensions},).")

    rng = np.random.default_rng(seed)
    walkers = _initial_walkers(centre, bounds, n_walkers, rng)
    log_prob = np.asarray([float(evaluate(jnp.asarray(walker))) for walker in walkers])
    if not np.all(np.isfinite(log_prob)):
        raise ValueError("Initial walkers include non-finite posterior values.")

    accepted = np.zeros(n_walkers, dtype=np.int64)
    saved_chain: list[np.ndarray] = []
    saved_log_prob: list[np.ndarray] = []
    groups = (np.arange(0, n_walkers, 2), np.arange(1, n_walkers, 2))
    for step in range(n_steps):
        for group, complement in ((groups[0], groups[1]), (groups[1], groups[0])):
            for index in group:
                partner = walkers[rng.choice(complement)]
                unit = rng.random()
                stretch = ((stretch_scale - 1.0) * unit + 1.0) ** 2 / stretch_scale
                proposal = partner + stretch * (walkers[index] - partner)
                proposal_log_prob = float(evaluate(jnp.asarray(proposal)))
                log_acceptance = (
                    (dimensions - 1) * math.log(stretch)
                    + proposal_log_prob
                    - log_prob[index]
                )
                if math.log(rng.random()) < log_acceptance:
                    walkers[index] = proposal
                    log_prob[index] = proposal_log_prob
                    accepted[index] += 1
        if step >= burn_in and (step - burn_in) % thin == 0:
            saved_chain.append(walkers.copy())
            saved_log_prob.append(log_prob.copy())
        if progress_callback is not None:
            progress_callback(
                SamplingProgress(
                    step=step + 1,
                    total_steps=n_steps,
                    mean_acceptance_fraction=float(np.mean(accepted / (step + 1))),
                )
            )

    return PixelizedSamplingResult(
        parameter_names=names,
        chain=np.asarray(saved_chain),
        log_probability=np.asarray(saved_log_prob),
        acceptance_fraction=accepted / n_steps,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
    )
