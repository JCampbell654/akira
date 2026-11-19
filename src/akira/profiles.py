"""Analytic mass and surface-brightness profiles."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from jax import Array
from jax import lax
from jax.scipy.special import gammainc, gammaln


EPL_SERIES_TERMS = 256
"""Fixed Fourier terms for accurate EPL deflections down to q=0.05."""


def _unflatten_without_validation(cls, field_names, children):
    """Rebuild a frozen profile from JAX tracers without Python validation."""

    instance = object.__new__(cls)
    for name, value in zip(field_names, children, strict=True):
        object.__setattr__(instance, name, value)
    return instance


def _as_centre(centre: tuple[float, float] | Array) -> Array:
    return jnp.asarray(centre, dtype=jnp.float64)


def _aligned_coordinates(
    coordinates: Array,
    centre: tuple[float, float] | Array,
    phi: float | Array,
) -> tuple[Array, Array]:
    """Return internal major/minor-axis coordinates for a profile."""

    xy = jnp.asarray(coordinates, dtype=jnp.float64) - _as_centre(centre)
    cosine = jnp.cos(phi)
    sine = jnp.sin(phi)
    x_major = xy[..., 0] * cosine + xy[..., 1] * sine
    y_minor = -xy[..., 0] * sine + xy[..., 1] * cosine
    return x_major, y_minor


def _rotate_from_profile_frame(
    x_component: Array, y_component: Array, phi: float | Array
) -> Array:
    """Rotate vector components from a profile frame into public x/y axes."""

    cosine = jnp.cos(phi)
    sine = jnp.sin(phi)
    x = x_component * cosine - y_component * sine
    y = x_component * sine + y_component * cosine
    return jnp.stack((x, y), axis=-1)


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

    x_major, y_minor = _aligned_coordinates(coordinates, centre, phi)
    axis_ratio = jnp.asarray(q, dtype=jnp.float64)
    radius_squared = axis_ratio * x_major**2 + y_minor**2 / axis_ratio
    scale = jnp.asarray(sigma, dtype=jnp.float64)
    return jnp.asarray(intensity, dtype=jnp.float64) * jnp.exp(
        -0.5 * radius_squared / scale**2
    )


def epl_deflection(
    coordinates: Array,
    einstein_radius: float | Array,
    slope: float | Array,
    centre: tuple[float, float] | Array = (0.0, 0.0),
    q: float | Array = 1.0,
    phi: float | Array = 0.0,
) -> Array:
    """Return elliptical power-law deflections in arcseconds.

    ``slope`` is the three-dimensional density slope ``gamma`` in
    ``rho(r) ∝ r**(-gamma)``. The Tessore--Metcalf projected slope is therefore
    ``t = gamma - 1``. ``einstein_radius`` is the circularized scale, so their
    semi-minor-axis scale is ``b = einstein_radius * sqrt(q)``.

    The hypergeometric angular factor is evaluated with the convergent Fourier
    recurrence of Tessore & Metcalf (2015), equations 28--33.
    """

    x_major, y_minor = _aligned_coordinates(coordinates, centre, phi)
    axis_ratio = jnp.asarray(q, dtype=jnp.float64)
    projected_slope = jnp.asarray(slope, dtype=jnp.float64) - 1.0
    scale = jnp.asarray(einstein_radius, dtype=jnp.float64) * jnp.sqrt(axis_ratio)

    elliptical_radius = jnp.sqrt((axis_ratio * x_major) ** 2 + y_minor**2)
    safe_radius = jnp.where(elliptical_radius > 0.0, elliptical_radius, 1.0)
    elliptical_angle = jnp.arctan2(y_minor, axis_ratio * x_major)
    flattening = (1.0 - axis_ratio) / (1.0 + axis_ratio)

    initial_term = jnp.exp(1j * elliptical_angle).astype(jnp.complex128)
    rotation = jnp.exp(2j * elliptical_angle).astype(jnp.complex128)

    def add_term(index: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        total, previous = state
        n = jnp.asarray(index, dtype=jnp.float64)
        ratio = -flattening * (
            (2.0 * n - (2.0 - projected_slope))
            / (2.0 * n + (2.0 - projected_slope))
        )
        current = ratio * rotation * previous
        return total + current, current

    angular, _ = lax.fori_loop(
        1, EPL_SERIES_TERMS, add_term, (initial_term, initial_term)
    )
    radial = (
        2.0
        * scale
        / (1.0 + axis_ratio)
        * (scale / safe_radius) ** (projected_slope - 1.0)
    )
    alpha_complex = radial * angular
    alpha_major = jnp.real(alpha_complex)
    alpha_minor = jnp.imag(alpha_complex)
    alpha_major = jnp.where(elliptical_radius > 0.0, alpha_major, 0.0)
    alpha_minor = jnp.where(elliptical_radius > 0.0, alpha_minor, 0.0)
    return _rotate_from_profile_frame(alpha_major, alpha_minor, phi)


def external_shear_deflection(
    coordinates: Array,
    strength: float | Array,
    phi: float | Array = 0.0,
    centre: tuple[float, float] | Array = (0.0, 0.0),
) -> Array:
    """Return external-shear deflections.

    The Cartesian components are ``gamma1 = strength*cos(2*phi)`` and
    ``gamma2 = strength*sin(2*phi)``. At ``phi=0``, the deflection is
    ``(strength*x, -strength*y)`` relative to ``centre``.
    """

    xy = jnp.asarray(coordinates, dtype=jnp.float64) - _as_centre(centre)
    gamma1 = jnp.asarray(strength, dtype=jnp.float64) * jnp.cos(2.0 * phi)
    gamma2 = jnp.asarray(strength, dtype=jnp.float64) * jnp.sin(2.0 * phi)
    alpha_x = gamma1 * xy[..., 0] + gamma2 * xy[..., 1]
    alpha_y = gamma2 * xy[..., 0] - gamma1 * xy[..., 1]
    return jnp.stack((alpha_x, alpha_y), axis=-1)


def sersic_bn(sersic_index: float | Array) -> Array:
    """Solve ``P(2n, b_n)=0.5`` with differentiable Newton iterations."""

    n = jnp.asarray(sersic_index, dtype=jnp.float64)
    initial = jnp.maximum(2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n), 1e-6)

    def newton_step(_: int, value: Array) -> Array:
        residual = gammainc(2.0 * n, value) - 0.5
        derivative = jnp.exp((2.0 * n - 1.0) * jnp.log(value) - value - gammaln(2.0 * n))
        return jnp.maximum(value - residual / derivative, 1e-12)

    return lax.fori_loop(0, 8, newton_step, initial)


def sersic_brightness(
    coordinates: Array,
    intensity: float | Array,
    effective_radius: float | Array,
    sersic_index: float | Array,
    centre: tuple[float, float] | Array = (0.0, 0.0),
    q: float | Array = 1.0,
    phi: float | Array = 0.0,
) -> Array:
    """Evaluate an elliptical Sérsic surface-brightness profile.

    ``intensity`` is the surface brightness at the circularized half-light
    radius ``effective_radius``. ``sersic_index`` is conventionally denoted n.
    """

    x_major, y_minor = _aligned_coordinates(coordinates, centre, phi)
    axis_ratio = jnp.asarray(q, dtype=jnp.float64)
    radius = jnp.sqrt(axis_ratio * x_major**2 + y_minor**2 / axis_ratio)
    n = jnp.asarray(sersic_index, dtype=jnp.float64)
    bn = sersic_bn(n)
    scaled_radius = radius / jnp.asarray(effective_radius, dtype=jnp.float64)
    return jnp.asarray(intensity, dtype=jnp.float64) * jnp.exp(
        -bn * (scaled_radius ** (1.0 / n) - 1.0)
    )


@jax.tree_util.register_pytree_node_class
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

    def tree_flatten(self):
        return (self.einstein_radius, self.centre), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return _unflatten_without_validation(
            cls, ("einstein_radius", "centre"), children
        )


@jax.tree_util.register_pytree_node_class
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

    def tree_flatten(self):
        return (self.intensity, self.sigma, self.centre, self.q, self.phi), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return _unflatten_without_validation(
            cls, ("intensity", "sigma", "centre", "q", "phi"), children
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class EllipticalPowerLaw:
    """Elliptical power-law mass profile with a 3D density slope."""

    einstein_radius: float
    slope: float
    centre: tuple[float, float] = (0.0, 0.0)
    q: float = 1.0
    phi: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.einstein_radius) or self.einstein_radius <= 0:
            raise ValueError("einstein_radius must be a positive finite value.")
        if not math.isfinite(self.slope) or not 1.0 < self.slope < 3.0:
            raise ValueError("3D slope must satisfy 1 < slope < 3.")
        if not math.isfinite(self.q) or not 0.05 <= self.q <= 1.0:
            raise ValueError("q must satisfy 0.05 <= q <= 1.")
        if not math.isfinite(self.phi):
            raise ValueError("phi must be finite and expressed in radians.")
        if len(self.centre) != 2 or not all(math.isfinite(v) for v in self.centre):
            raise ValueError("centre must contain two finite values in (x, y) order.")

    def deflection(self, coordinates: Array) -> Array:
        return epl_deflection(
            coordinates,
            self.einstein_radius,
            self.slope,
            self.centre,
            self.q,
            self.phi,
        )

    def tree_flatten(self):
        return (self.einstein_radius, self.slope, self.centre, self.q, self.phi), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return _unflatten_without_validation(
            cls,
            ("einstein_radius", "slope", "centre", "q", "phi"),
            children,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class ExternalShear:
    """Constant external shear with angle anticlockwise from positive x."""

    strength: float
    phi: float = 0.0
    centre: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength) or not 0.0 <= self.strength < 1.0:
            raise ValueError("strength must satisfy 0 <= strength < 1.")
        if not math.isfinite(self.phi):
            raise ValueError("phi must be finite and expressed in radians.")
        if len(self.centre) != 2 or not all(math.isfinite(v) for v in self.centre):
            raise ValueError("centre must contain two finite values in (x, y) order.")

    def deflection(self, coordinates: Array) -> Array:
        return external_shear_deflection(
            coordinates, self.strength, self.phi, self.centre
        )

    def tree_flatten(self):
        return (self.strength, self.phi, self.centre), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return _unflatten_without_validation(
            cls, ("strength", "phi", "centre"), children
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Sersic:
    """Elliptical Sérsic light profile with circularized effective radius."""

    intensity: float
    effective_radius: float
    sersic_index: float
    centre: tuple[float, float] = (0.0, 0.0)
    q: float = 1.0
    phi: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.intensity) or self.intensity < 0:
            raise ValueError("intensity must be a non-negative finite value.")
        if not math.isfinite(self.effective_radius) or self.effective_radius <= 0:
            raise ValueError("effective_radius must be a positive finite value.")
        if not math.isfinite(self.sersic_index) or not 0.3 <= self.sersic_index <= 10:
            raise ValueError("sersic_index must satisfy 0.3 <= n <= 10.")
        if not math.isfinite(self.q) or not 0 < self.q <= 1:
            raise ValueError("q must satisfy 0 < q <= 1.")
        if not math.isfinite(self.phi):
            raise ValueError("phi must be finite and expressed in radians.")
        if len(self.centre) != 2 or not all(math.isfinite(v) for v in self.centre):
            raise ValueError("centre must contain two finite values in (x, y) order.")

    def brightness(self, coordinates: Array) -> Array:
        return sersic_brightness(
            coordinates,
            self.intensity,
            self.effective_radius,
            self.sersic_index,
            self.centre,
            self.q,
            self.phi,
        )

    def tree_flatten(self):
        return (
            self.intensity,
            self.effective_radius,
            self.sersic_index,
            self.centre,
            self.q,
            self.phi,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return _unflatten_without_validation(
            cls,
            (
                "intensity",
                "effective_radius",
                "sersic_index",
                "centre",
                "q",
                "phi",
            ),
            children,
        )
