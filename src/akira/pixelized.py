"""Dense reference implementation of regular pixelized source inversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from akira.coordinates import CartesianGrid
from akira.imaging import ImagingObservation, convolve_psf
from akira.lensing import ray_trace
from akira.model import LensPlane


RegularizationKind = Literal["zeroth", "gradient", "curvature"]


@dataclass(frozen=True, slots=True)
class SourceGrid:
    """A regular Cartesian source plane whose pixels store surface brightness."""

    shape: tuple[int, int]
    pixel_scale: float
    centre: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(size <= 1 for size in self.shape):
            raise ValueError("Source-grid dimensions must each exceed one pixel.")
        if not math.isfinite(self.pixel_scale) or self.pixel_scale <= 0:
            raise ValueError("pixel_scale must be positive and finite.")
        if len(self.centre) != 2 or not all(math.isfinite(v) for v in self.centre):
            raise ValueError("centre must contain finite (x, y) coordinates.")

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]

    def coordinates(self) -> Array:
        """Return source-pixel centres with shape ``(ny, nx, 2)``."""

        ny, nx = self.shape
        x = (jnp.arange(nx, dtype=jnp.float64) - (nx - 1) / 2) * self.pixel_scale
        y = (jnp.arange(ny, dtype=jnp.float64) - (ny - 1) / 2) * self.pixel_scale
        yy, xx = jnp.meshgrid(y + self.centre[1], x + self.centre[0], indexing="ij")
        return jnp.stack((xx, yy), axis=-1)


def bilinear_interpolation_matrix(coordinates: Array, source_grid: SourceGrid) -> Array:
    """Map flattened source pixels to arbitrary source-plane coordinates.

    Rows for coordinates outside the source grid are zero. At the outermost
    source-pixel centres, interpolation remains valid and uses that edge pixel.
    """

    coordinates = jnp.asarray(coordinates, dtype=jnp.float64)
    original_shape = coordinates.shape[:-1]
    xy = coordinates.reshape((-1, 2))
    ny, nx = source_grid.shape
    x_min = source_grid.centre[0] - (nx - 1) * source_grid.pixel_scale / 2
    y_min = source_grid.centre[1] - (ny - 1) * source_grid.pixel_scale / 2
    fractional_x = (xy[:, 0] - x_min) / source_grid.pixel_scale
    fractional_y = (xy[:, 1] - y_min) / source_grid.pixel_scale
    tolerance = 32.0 * jnp.finfo(jnp.float64).eps
    inside = (
        (fractional_x >= -tolerance)
        & (fractional_x <= nx - 1 + tolerance)
        & (fractional_y >= -tolerance)
        & (fractional_y <= ny - 1 + tolerance)
    )
    fractional_x = jnp.clip(fractional_x, 0.0, nx - 1)
    fractional_y = jnp.clip(fractional_y, 0.0, ny - 1)
    lower_x = jnp.clip(jnp.floor(fractional_x).astype(jnp.int32), 0, nx - 2)
    lower_y = jnp.clip(jnp.floor(fractional_y).astype(jnp.int32), 0, ny - 2)
    dx = fractional_x - lower_x
    dy = fractional_y - lower_y
    indices = jnp.stack(
        (
            lower_y * nx + lower_x,
            lower_y * nx + lower_x + 1,
            (lower_y + 1) * nx + lower_x,
            (lower_y + 1) * nx + lower_x + 1,
        ),
        axis=1,
    )
    weights = jnp.stack(
        ((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy),
        axis=1,
    )
    weights = jnp.where(inside[:, None], weights, 0.0)
    matrix = jnp.zeros((xy.shape[0], source_grid.size), dtype=jnp.float64)
    matrix = matrix.at[jnp.arange(xy.shape[0])[:, None], indices].add(weights)
    return matrix.reshape(original_shape + (source_grid.size,))


def lensing_operator(
    image_grid: CartesianGrid,
    source_grid: SourceGrid,
    lens: LensPlane,
) -> Array:
    """Return the dense image-pixel by source-pixel lensing operator."""

    image_coordinates = image_grid.subpixel_coordinates()
    source_coordinates = ray_trace(
        image_coordinates, lens.deflection(image_coordinates)
    )
    interpolation = bilinear_interpolation_matrix(source_coordinates, source_grid)
    return interpolation.mean(axis=2).reshape((-1, source_grid.size))


def apply_psf_to_operator(
    operator: Array,
    image_shape: tuple[int, int],
    psf_kernel: Array,
) -> Array:
    """Convolve every source pixel's unit image with the detector PSF."""

    source_first = operator.T.reshape((operator.shape[1],) + image_shape)
    convolved = jax.vmap(lambda image: convolve_psf(image, psf_kernel))(source_first)
    return convolved.reshape((operator.shape[1], -1)).T


def forward_operator(
    image_grid: CartesianGrid,
    source_grid: SourceGrid,
    lens: LensPlane,
    psf_kernel: Array,
) -> Array:
    """Return the dense lensing-plus-PSF operator."""

    return apply_psf_to_operator(
        lensing_operator(image_grid, source_grid, lens), image_grid.shape, psf_kernel
    )


def regularization_matrix(
    source_grid: SourceGrid,
    kind: RegularizationKind,
) -> Array:
    """Return Warren--Dye zeroth, gradient, or curvature regularization H."""

    ny, nx = source_grid.shape
    size = source_grid.size
    if kind == "zeroth":
        return jnp.eye(size, dtype=jnp.float64)
    rows: list[np.ndarray] = []
    if kind == "gradient":
        for y in range(ny):
            for x in range(nx - 1):
                row = np.zeros(size)
                row[y * nx + x] = 1.0
                row[y * nx + x + 1] = -1.0
                rows.append(row)
        for y in range(ny - 1):
            for x in range(nx):
                row = np.zeros(size)
                row[y * nx + x] = 1.0
                row[(y + 1) * nx + x] = -1.0
                rows.append(row)
    elif kind == "curvature":
        for y in range(ny):
            for x in range(1, nx - 1):
                row = np.zeros(size)
                row[y * nx + x] = 1.0
                row[y * nx + x - 1] = -0.5
                row[y * nx + x + 1] = -0.5
                rows.append(row)
        for y in range(1, ny - 1):
            for x in range(nx):
                row = np.zeros(size)
                row[y * nx + x] = 1.0
                row[(y - 1) * nx + x] = -0.5
                row[(y + 1) * nx + x] = -0.5
                rows.append(row)
    else:
        raise ValueError("kind must be 'zeroth', 'gradient', or 'curvature'.")
    difference = jnp.asarray(np.stack(rows), dtype=jnp.float64)
    return difference.T @ difference


@dataclass(frozen=True, slots=True)
class PixelizedReconstruction:
    """Source solution and diagnostics for a fixed lens mass distribution."""

    source: Array
    model_image: Array
    residuals: Array
    operator: Array
    data_curvature: Array
    regularization: Array
    system_matrix: Array
    source_covariance: Array
    chi_squared: float
    regularization_penalty: float
    log_evidence: float | None
    regularization_kind: str
    regularization_strength: float


def solve_source(
    operator: Array,
    data: Array,
    noise: Array,
    regularization: Array,
    regularization_strength: float | Array,
) -> tuple[Array, Array, Array, Array]:
    """Solve the differentiable regularized normal equations.

    Returns the source vector, data curvature, full system matrix, and source
    covariance. Inputs must already contain only the desired unmasked pixels.
    """

    weighted_operator = operator / noise[:, None]
    data_curvature = weighted_operator.T @ weighted_operator
    data_vector = operator.T @ (data / noise**2)
    system = data_curvature + regularization_strength * regularization
    source = jnp.linalg.solve(system, data_vector)
    inverse_system = jnp.linalg.inv(system)
    covariance = inverse_system @ data_curvature @ inverse_system
    return source, data_curvature, system, covariance


def _proper_gaussian_log_evidence(
    data: Array,
    noise: Array,
    operator: Array,
    source: Array,
    system: Array,
    regularization: Array,
    strength: float,
) -> float | None:
    """Return Gaussian marginal likelihood when H defines a proper prior."""

    eigenvalues = jnp.linalg.eigvalsh(regularization)
    rank_tolerance = (
        jnp.max(jnp.abs(eigenvalues))
        * regularization.shape[0]
        * jnp.finfo(jnp.float64).eps
    )
    if float(jnp.min(eigenvalues)) <= float(rank_tolerance):
        return None
    sign_h, logdet_h = jnp.linalg.slogdet(strength * regularization)
    sign_a, logdet_a = jnp.linalg.slogdet(system)
    if float(sign_h) <= 0.0 or float(sign_a) <= 0.0:
        return None
    residual = data - operator @ source
    chi_squared = jnp.sum((residual / noise) ** 2)
    penalty = strength * source @ regularization @ source
    logdet_noise = jnp.sum(jnp.log(noise**2))
    value = (
        -0.5 * (chi_squared + penalty)
        + 0.5 * logdet_h
        - 0.5 * logdet_a
        - 0.5 * logdet_noise
        - 0.5 * data.size * jnp.log(2.0 * jnp.pi)
    )
    return float(value)


def reconstruct_source(
    observation: ImagingObservation,
    image_grid: CartesianGrid,
    source_grid: SourceGrid,
    lens: LensPlane,
    regularization_kind: RegularizationKind = "gradient",
    regularization_strength: float = 1.0,
) -> PixelizedReconstruction:
    """Solve the Warren--Dye regularized linear source problem for a fixed lens."""

    if observation.data.shape != image_grid.shape:
        raise ValueError("Observation and image-grid shapes must match.")
    if not math.isfinite(regularization_strength) or regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive and finite.")
    operator = forward_operator(
        image_grid, source_grid, lens, observation.psf.kernel
    )
    return reconstruct_from_operator(
        observation,
        source_grid,
        operator,
        regularization_kind,
        regularization_strength,
    )


def reconstruct_from_operator(
    observation: ImagingObservation,
    source_grid: SourceGrid,
    operator: Array,
    regularization_kind: RegularizationKind = "gradient",
    regularization_strength: float = 1.0,
) -> PixelizedReconstruction:
    """Reconstruct a source from a precomputed lensing-plus-PSF operator."""

    expected_shape = (observation.data.size, source_grid.size)
    if operator.shape != expected_shape:
        raise ValueError(f"operator must have shape {expected_shape}.")
    if not math.isfinite(regularization_strength) or regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive and finite.")
    data = observation.data.reshape(-1)
    noise = observation.noise_map.reshape(-1)
    active = ~observation.mask.reshape(-1)
    active_operator = operator[active]
    active_data = data[active]
    active_noise = noise[active]
    regularization = regularization_matrix(source_grid, regularization_kind)
    source, data_curvature, system, covariance = solve_source(
        active_operator,
        active_data,
        active_noise,
        regularization,
        regularization_strength,
    )
    flat_model = operator @ source
    model_image = flat_model.reshape(observation.data.shape)
    residuals = (observation.data - model_image) / observation.noise_map
    active_residuals = (active_data - active_operator @ source) / active_noise
    chi_squared = jnp.sum(active_residuals**2)
    penalty = source @ regularization @ source
    log_evidence = _proper_gaussian_log_evidence(
        active_data,
        active_noise,
        active_operator,
        source,
        system,
        regularization,
        regularization_strength,
    )
    return PixelizedReconstruction(
        source=source.reshape(source_grid.shape),
        model_image=model_image,
        residuals=residuals,
        operator=operator,
        data_curvature=data_curvature,
        regularization=regularization,
        system_matrix=system,
        source_covariance=covariance,
        chi_squared=float(chi_squared),
        regularization_penalty=float(penalty),
        log_evidence=log_evidence,
        regularization_kind=regularization_kind,
        regularization_strength=regularization_strength,
    )
