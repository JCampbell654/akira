"""Analytic mass and surface-brightness profiles."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp
from jax import Array


def _as_centre(centre: tuple[float, float] | Array) -> Array:
    return jnp.asarray(centre, dtype=jnp.float64)


def sis_deflection(
    coordinates: Array,
    einstein_radius: float | Array,
    centre: tuple[float, float] | Array = (0.0, 0.0),
) -> Array:
    """Return SIS deflections in arcseconds.

    The deflection points radially away from the lens centre and has magnitude
    equal to ``einstein_radius``. At the singular origin the numerical
    convention is a zero deflection.
    """

    offset = jnp.asarray(coordinates, dtype=jnp.float64) - _as_centre(centre)
    radius = jnp.linalg.vector_norm(offset, axis=-1, keepdims=True)
    safe_radius = jnp.where(radius > 0.0, radius, 1.0)
    direction = offset / safe_radius
    direction = jnp.where(radius > 0.0, direction, 0.0)
    return jnp.asarray(einstein_radius, dtype=jnp.float64) * direction


def gaussian_brightness(
    coordinates: Array,
    intensity: float | Array,
    sigma: float | Array,
    centre: tuple[float, float] | Array = (0.0, 0.0),
    q: float | Array = 1.0,
    phi: float | Array = 0.0,
) -> Array:
    """Evaluate an elliptical Gaussian surface-brightness profile.

    ``sigma`` is the circularized scale in arcseconds. ``q=b/a`` and ``phi``
    is the anticlockwise major-axis angle from positive x, in radians.
    """

    xy = jnp.asarray(coordinates, dtype=jnp.float64) - _as_centre(centre)
    cosine = jnp.cos(phi)
    sine = jnp.sin(phi)
    x_major = xy[..., 0] * cosine + xy[..., 1] * sine
    y_minor = -xy[..., 0] * sine + xy[..., 1] * cosine
    axis_ratio = jnp.asarray(q, dtype=jnp.float64)
    radius_squared = axis_ratio * x_major**2 + y_minor**2 / axis_ratio
    scale = jnp.asarray(sigma, dtype=jnp.float64)
    return jnp.asarray(intensity, dtype=jnp.float64) * jnp.exp(
        -0.5 * radius_squared / scale**2
    )


@dataclass(frozen=True, slots=True)
class SIS:
    """Singular isothermal sphere mass profile."""

    einstein_radius: float
    centre: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.einstein_radius) or self.einstein_radius <= 0:
            raise ValueError("einstein_radius must be a positive finite value.")
        if len(self.centre) != 2 or not all(math.isfinite(v) for v in self.centre):
            raise ValueError("centre must contain two finite values in (x, y) order.")

    def deflection(self, coordinates: Array) -> Array:
        return sis_deflection(coordinates, self.einstein_radius, self.centre)


@dataclass(frozen=True, slots=True)
class Gaussian:
    """Elliptical Gaussian surface-brightness profile."""

    intensity: float
    sigma: float
    centre: tuple[float, float] = (0.0, 0.0)
    q: float = 1.0
    phi: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.intensity) or self.intensity < 0:
            raise ValueError("intensity must be a non-negative finite value.")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be a positive finite value.")
        if not math.isfinite(self.q) or not 0 < self.q <= 1:
            raise ValueError("q must satisfy 0 < q <= 1.")
        if not math.isfinite(self.phi):
            raise ValueError("phi must be finite and expressed in radians.")
        if len(self.centre) != 2 or not all(math.isfinite(v) for v in self.centre):
            raise ValueError("centre must contain two finite values in (x, y) order.")

    def brightness(self, coordinates: Array) -> Array:
        return gaussian_brightness(
            coordinates,
            self.intensity,
            self.sigma,
            self.centre,
            self.q,
            self.phi,
        )

