"""General parameter, prior, and gradient-based inference framework."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
import math
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import ndtr, ndtri
import numpy as np
from scipy.optimize import minimize

from akira.coordinates import CartesianGrid
from akira.imaging import ImagingObservation, convolve_psf
from akira.likelihood import gaussian_log_likelihood
from akira.model import CompositeLight, LensPlane
from akira.render import render_from_coordinates


LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class UniformPrior:
    """Uniform probability density between finite lower and upper limits."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("Uniform bounds must be finite.")
        if self.lower >= self.upper:
            raise ValueError("Uniform lower bound must be below its upper bound.")

    @property
    def bounds(self) -> tuple[float, float]:
        return self.lower, self.upper

    def log_prob(self, value: Array | float) -> Array:
        value = jnp.asarray(value, dtype=jnp.float64)
        density = -jnp.log(self.upper - self.lower)
        return jnp.where((value >= self.lower) & (value <= self.upper), density, -jnp.inf)

    def from_unit(self, unit_value: Array | float) -> Array:
        unit_value = jnp.asarray(unit_value, dtype=jnp.float64)
        return self.lower + unit_value * (self.upper - self.lower)


@dataclass(frozen=True, slots=True)
class GaussianPrior:
    """Unbounded Gaussian prior parameterized by mean and standard deviation."""

    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("Gaussian mean must be finite.")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Gaussian sigma must be positive and finite.")

    @property
    def bounds(self) -> tuple[None, None]:
        return None, None

    def log_prob(self, value: Array | float) -> Array:
        standardized = (jnp.asarray(value, dtype=jnp.float64) - self.mean) / self.sigma
        return -0.5 * standardized**2 - jnp.log(self.sigma) - 0.5 * LOG_TWO_PI

    def from_unit(self, unit_value: Array | float) -> Array:
        return self.mean + self.sigma * ndtri(jnp.asarray(unit_value, dtype=jnp.float64))


@dataclass(frozen=True, slots=True)
class LogUniformPrior:
    """Prior uniform in the natural logarithm of a positive value."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or self.lower <= 0:
            raise ValueError("Log-uniform lower bound must be positive and finite.")
        if not math.isfinite(self.upper) or self.lower >= self.upper:
            raise ValueError("Log-uniform upper bound must exceed its lower bound.")

    @property
    def bounds(self) -> tuple[float, float]:
        return self.lower, self.upper

    def log_prob(self, value: Array | float) -> Array:
        value = jnp.asarray(value, dtype=jnp.float64)
        normalization = jnp.log(jnp.log(self.upper / self.lower))
        density = -jnp.log(value) - normalization
        return jnp.where((value >= self.lower) & (value <= self.upper), density, -jnp.inf)

    def from_unit(self, unit_value: Array | float) -> Array:
        unit_value = jnp.asarray(unit_value, dtype=jnp.float64)
        return self.lower * (self.upper / self.lower) ** unit_value


@dataclass(frozen=True, slots=True)
class TruncatedGaussianPrior:
    """Normalized Gaussian probability density restricted to finite bounds."""

    mean: float
    sigma: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.mean, self.sigma, self.lower, self.upper)):
            raise ValueError("Truncated-Gaussian parameters must be finite.")
        if self.sigma <= 0:
            raise ValueError("Truncated-Gaussian sigma must be positive.")
        if self.lower >= self.upper:
            raise ValueError("Lower bound must be below the upper bound.")

    @property
    def bounds(self) -> tuple[float, float]:
        return self.lower, self.upper

    def log_prob(self, value: Array | float) -> Array:
        value = jnp.asarray(value, dtype=jnp.float64)
        standardized = (value - self.mean) / self.sigma
        low = (self.lower - self.mean) / self.sigma
        high = (self.upper - self.mean) / self.sigma
        log_normalization = jnp.log(ndtr(high) - ndtr(low))
        density = (
            -0.5 * standardized**2
            - jnp.log(self.sigma)
            - 0.5 * LOG_TWO_PI
            - log_normalization
        )
        return jnp.where((value >= self.lower) & (value <= self.upper), density, -jnp.inf)

    def from_unit(self, unit_value: Array | float) -> Array:
        low = (self.lower - self.mean) / self.sigma
        high = (self.upper - self.mean) / self.sigma
        probability = ndtr(low) + jnp.asarray(unit_value, dtype=jnp.float64) * (
            ndtr(high) - ndtr(low)
        )
        return self.mean + self.sigma * ndtri(probability)


Prior = UniformPrior | GaussianPrior | LogUniformPrior | TruncatedGaussianPrior


@dataclass(frozen=True, slots=True)
class FreeParameter:
    """A free scalar parameter with an initial value and probability prior."""

    initial: float
    prior: Prior

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial):
            raise ValueError("A free parameter's initial value must be finite.")
        lower, upper = self.prior.bounds
        if lower is not None and self.initial < lower:
            raise ValueError("Initial value lies below its prior support.")
        if upper is not None and self.initial > upper:
            raise ValueError("Initial value lies above its prior support.")


@dataclass(frozen=True, slots=True)
class Fixed:
    """Explicitly mark a model value as fixed rather than inferred."""

    value: Any


def _initial_value(value: Any) -> Any:
    if isinstance(value, FreeParameter):
        return value.initial
    if isinstance(value, Fixed):
        return value.value
    if isinstance(value, tuple):
        return tuple(_initial_value(item) for item in value)
    return value


def _resolve_value(value: Any, vector: Array, cursor: list[int]) -> Any:
    if isinstance(value, FreeParameter):
        resolved = vector[cursor[0]]
        cursor[0] += 1
        return resolved
    if isinstance(value, Fixed):
        return value.value
    if isinstance(value, tuple):
        return tuple(_resolve_value(item, vector, cursor) for item in value)
    return value


def _collect_parameters(value: Any, prefix: str, output: list[tuple[str, FreeParameter]]) -> None:
    if isinstance(value, FreeParameter):
        output.append((prefix, value))
    elif isinstance(value, tuple):
        labels = ("x", "y") if len(value) == 2 and prefix.endswith("centre") else None
        for index, item in enumerate(value):
            suffix = labels[index] if labels is not None else str(index)
            _collect_parameters(item, f"{prefix}.{suffix}", output)


@dataclass(frozen=True, slots=True, init=False)
class ProfileModel:
    """Declarative fixed/free parameter specification for an analytic profile."""

    profile_type: type
    parameters: dict[str, Any]

    def __init__(self, profile_type: type, **parameters: Any) -> None:
        if not hasattr(profile_type, "tree_unflatten"):
            raise TypeError("profile_type must be a supported JAX profile class.")
        complete: dict[str, Any] = {}
        valid_fields = fields(profile_type)
        valid_names = {field.name for field in valid_fields}
        unknown = set(parameters) - valid_names
        if unknown:
            raise TypeError(f"Unknown {profile_type.__name__} parameters: {sorted(unknown)}")
        for field in valid_fields:
            if field.name in parameters:
                complete[field.name] = parameters[field.name]
            elif field.default is not MISSING:
                complete[field.name] = field.default
            elif field.default_factory is not MISSING:
                complete[field.name] = field.default_factory()
            else:
                raise TypeError(f"Missing required parameter: {field.name}")
        profile_type(**{name: _initial_value(value) for name, value in complete.items()})
        object.__setattr__(self, "profile_type", profile_type)
        object.__setattr__(self, "parameters", complete)

    def build(self, vector: Array, cursor: list[int]) -> object:
        children = tuple(
            _resolve_value(value, vector, cursor) for value in self.parameters.values()
        )
        return self.profile_type.tree_unflatten(None, children)

    def collect(self, prefix: str, output: list[tuple[str, FreeParameter]]) -> None:
        for name, value in self.parameters.items():
            _collect_parameters(value, f"{prefix}.{name}", output)


@dataclass(frozen=True, slots=True)
class LensModel:
    """A general parameterized single-plane lens and source-light model."""

    lens: tuple[ProfileModel, ...]
    source: tuple[ProfileModel, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lens", tuple(self.lens))
        object.__setattr__(self, "source", tuple(self.source))
        if not self.lens or not self.source:
            raise ValueError("LensModel requires lens and source profiles.")

    @property
    def parameter_items(self) -> tuple[tuple[str, FreeParameter], ...]:
        output: list[tuple[str, FreeParameter]] = []
        for index, profile in enumerate(self.lens):
            profile.collect(f"lens.{index}.{profile.profile_type.__name__}", output)
        for index, profile in enumerate(self.source):
            profile.collect(f"source.{index}.{profile.profile_type.__name__}", output)
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

    def from_unit(self, unit_vector: Array) -> Array:
        unit_vector = jnp.asarray(unit_vector, dtype=jnp.float64)
        if unit_vector.shape != (len(self.parameter_items),):
            raise ValueError("unit_vector has the wrong number of parameters.")
        return jnp.stack(
            [parameter.prior.from_unit(value) for value, (_, parameter) in zip(unit_vector, self.parameter_items)]
        )

    def log_prior(self, vector: Array) -> Array:
        return sum(
            (
                parameter.prior.log_prob(value)
                for value, (_, parameter) in zip(vector, self.parameter_items)
            ),
            start=jnp.asarray(0.0, dtype=jnp.float64),
        )

    def build(self, vector: Array) -> tuple[LensPlane, CompositeLight]:
        cursor = [0]
        lens = LensPlane(tuple(profile.build(vector, cursor) for profile in self.lens))
        source = CompositeLight(tuple(profile.build(vector, cursor) for profile in self.source))
        return lens, source


@dataclass(frozen=True, slots=True)
class InferenceProblem:
    """A differentiable analytic-lens imaging inference problem.

    This is the common boundary between a scientific model and an inference
    algorithm. Optimizers and posterior samplers can use the same parameter
    ordering, model construction, prediction, likelihood, and prior methods.
    """

    model: LensModel
    observation: ImagingObservation
    grid: CartesianGrid
    _coordinates: Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.observation.data.shape != self.grid.shape:
            raise ValueError("Observation and grid shapes must match.")
        object.__setattr__(self, "_coordinates", self.grid.subpixel_coordinates())

    @property
    def parameter_items(self) -> tuple[tuple[str, FreeParameter], ...]:
        """Free parameters in the stable order used by every vector method."""

        return self.model.parameter_items

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names corresponding to entries in a parameter vector."""

        return self.model.parameter_names

    @property
    def initial_vector(self) -> Array:
        """Configured starting point in physical parameter space."""

        return self.model.initial_vector

    @property
    def bounds(self) -> tuple[tuple[float | None, float | None], ...]:
        """Physical bounds supplied by each parameter's prior."""

        return tuple(parameter.prior.bounds for _, parameter in self.parameter_items)

    def from_unit(self, unit_vector: Array) -> Array:
        """Transform a unit-cube vector through the declared priors."""

        return self.model.from_unit(unit_vector)

    def build_model(self, vector: Array) -> tuple[LensPlane, CompositeLight]:
        """Construct the physical lens and source for a parameter vector."""

        return self.model.build(vector)

    def predict(self, vector: Array) -> Array:
        """Render and PSF-convolve the detector image for a parameter vector."""

        lens, source = self.build_model(vector)
        unconvolved = render_from_coordinates(self._coordinates, lens, source)
        return convolve_psf(unconvolved, self.observation.psf.kernel)

    def log_likelihood(self, vector: Array) -> Array:
        """Return the normalized masked Gaussian image log likelihood."""

        image = self.predict(vector)
        return gaussian_log_likelihood(
            self.observation.data,
            image,
            self.observation.noise_map,
            self.observation.mask,
        )

    def log_prior(self, vector: Array) -> Array:
        """Return the sum of all declared normalized parameter log priors."""

        return self.model.log_prior(vector)

    def log_posterior(self, vector: Array) -> Array:
        """Return log likelihood plus log prior, up to no omitted constants."""

        return self.log_likelihood(vector) + self.log_prior(vector)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Best-fitting model and diagnostics from general ML or MAP optimization."""

    parameter_names: tuple[str, ...]
    parameter_vector: Array
    parameters: dict[str, float]
    lens: LensPlane
    source: CompositeLight
    model_image: Array
    log_likelihood: float
    log_prior: float
    log_posterior: float
    mode: str
    success: bool
    message: str
    evaluations: int
    covariance: Array | None


def fit_model(
    model: LensModel,
    observation: ImagingObservation,
    grid: CartesianGrid,
    mode: Literal["maximum_likelihood", "map"] = "map",
) -> InferenceResult:
    """Fit every free model parameter using JAX gradients and L-BFGS-B."""

    if mode not in ("maximum_likelihood", "map"):
        raise ValueError("mode must be 'maximum_likelihood' or 'map'.")
    problem = InferenceProblem(model, observation, grid)
    if not problem.parameter_items:
        raise ValueError("LensModel contains no free parameters.")

    def objective(values: Array) -> Array:
        if mode == "maximum_likelihood":
            return -problem.log_likelihood(values)
        return -problem.log_posterior(values)

    value_and_gradient = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_gradient(jnp.asarray(values, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=np.float64)

    optimized = minimize(
        scipy_objective,
        x0=np.asarray(problem.initial_vector, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=problem.bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
    )
    fitted = jnp.asarray(optimized.x, dtype=jnp.float64)
    fitted_lens, fitted_source = problem.build_model(fitted)
    fitted_image = problem.predict(fitted)
    log_likelihood = problem.log_likelihood(fitted)
    log_prior = problem.log_prior(fitted)
    covariance = None
    if hasattr(optimized.hess_inv, "todense"):
        covariance = jnp.asarray(optimized.hess_inv.todense(), dtype=jnp.float64)
    values_by_name = dict(zip(problem.parameter_names, map(float, optimized.x)))
    return InferenceResult(
        parameter_names=problem.parameter_names,
        parameter_vector=fitted,
        parameters=values_by_name,
        lens=fitted_lens,
        source=fitted_source,
        model_image=fitted_image,
        log_likelihood=float(log_likelihood),
        log_prior=float(log_prior),
        log_posterior=float(log_likelihood + log_prior),
        mode=mode,
        success=bool(optimized.success),
        message=str(optimized.message),
        evaluations=int(optimized.nfev),
        covariance=covariance,
    )
