"""Critical curves and source-plane caustics for single-plane lenses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from akira.coordinates import CartesianGrid
from akira.model import LensPlane

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True, slots=True)
class LensingCurves:
    """Matched image-plane critical curves and source-plane caustics.

    Each element of ``caustics`` is the ray-traced counterpart of the element
    at the same position in ``critical_curves``. Coordinates use ``(x, y)``
    order and arcseconds.
    """

    critical_curves: tuple[np.ndarray, ...]
    caustics: tuple[np.ndarray, ...]


def lens_mapping_determinant(lens: LensPlane, coordinates: Array) -> Array:
    """Return ``det(d beta / d theta)`` at image-plane coordinates.

    Critical curves are the zero contour of this determinant. JAX automatic
    differentiation is used, so this works for any differentiable mass model
    that implements the standard ``deflection`` interface.
    """

    coordinates = jnp.asarray(coordinates, dtype=jnp.float64)
    if coordinates.shape[-1:] != (2,):
        raise ValueError("coordinates must have a final axis of length two.")

    def mapping(coordinate: Array) -> Array:
        return coordinate - lens.deflection(coordinate)

    flat = coordinates.reshape((-1, 2))
    jacobians = jax.vmap(jax.jacfwd(mapping))(flat)
    determinants = jnp.linalg.det(jacobians)
    return determinants.reshape(coordinates.shape[:-1])


def find_lensing_curves(
    lens: LensPlane,
    image_grid: CartesianGrid,
) -> LensingCurves:
    """Find critical curves and ray-trace them into source-plane caustics.

    ``image_grid`` controls the search area and numerical resolution. It
    should cover every expected critical curve and is normally much finer
    than the observed image grid. Matplotlib's lightweight contour engine is
    supplied by the optional ``notebook`` dependencies.
    """

    try:
        import contourpy
    except ImportError as error:  # pragma: no cover - depends on installation
        raise ImportError(
            "Finding lensing curves requires the plotting dependencies; "
            "install project-akira[notebook]."
        ) from error

    coordinates = image_grid.coordinates()
    determinant = np.asarray(lens_mapping_determinant(lens, coordinates))
    determinant = np.ma.masked_invalid(determinant)
    x = np.asarray(coordinates[0, :, 0])
    y = np.asarray(coordinates[:, 0, 1])
    generator = contourpy.contour_generator(x=x, y=y, z=determinant)
    # A singular profile may use a finite convention at its exact centre.
    # Autodifferentiating that single grid point can create a sub-pixel zero
    # contour. Such a feature is unresolved rather than a physical curve.
    minimum_span = 2.0 * image_grid.pixel_scale
    critical_curves = tuple(
        line
        for raw_line in generator.lines(0.0)
        if len(raw_line) >= 2
        for line in (np.asarray(raw_line),)
        if max(np.ptp(line[:, 0]), np.ptp(line[:, 1])) >= minimum_span
    )
    caustics = tuple(
        np.asarray(curve - lens.deflection(jnp.asarray(curve)))
        for curve in critical_curves
    )
    return LensingCurves(critical_curves, caustics)


def plot_caustics(
    axis: Axes,
    curves: LensingCurves,
    **plot_kwargs: object,
) -> list[object]:
    """Plot source-plane caustics on a Matplotlib axis.

    Keyword arguments are passed to ``axis.plot``. The returned artists can
    be used to customize the lines or construct a legend.
    """

    defaults: dict[str, object] = {
        "color": "cyan",
        "linewidth": 1.5,
        "label": "caustic",
    }
    defaults.update(plot_kwargs)
    artists: list[object] = []
    for index, caustic in enumerate(curves.caustics):
        kwargs = defaults if index == 0 else {**defaults, "label": "_nolegend_"}
        artists.extend(axis.plot(caustic[:, 0], caustic[:, 1], **kwargs))
    return artists
