"""Angular coordinate grids and angle conversions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp
from jax import Array


def degrees_to_radians(angle_degrees: float | Array) -> Array:
    """Convert an angle in degrees to radians."""

    return jnp.asarray(angle_degrees, dtype=jnp.float64) * (jnp.pi / 180.0)


def radians_to_degrees(angle_radians: float | Array) -> Array:
    """Convert an angle in radians to degrees."""

    return jnp.asarray(angle_radians, dtype=jnp.float64) * (180.0 / jnp.pi)


@dataclass(frozen=True, slots=True)
class CartesianGrid:
    """A regular image-plane grid in arcseconds.

    Scientific coordinates are returned in ``(x, y)`` order. Native image
    arrays retain the usual ``[y, x]`` indexing. ``centre`` is also ``(x, y)``.
    Pixel coordinates refer to pixel centres.
    """

    shape: tuple[int, int]
    pixel_scale: float
    centre: tuple[float, float] = (0.0, 0.0)
    supersampling: int = 1

    def __post_init__(self) -> None:
        ny, nx = self.shape
        if ny <= 0 or nx <= 0:
            raise ValueError("Grid dimensions must be positive.")
        if not math.isfinite(self.pixel_scale) or self.pixel_scale <= 0:
            raise ValueError("pixel_scale must be a positive finite value.")
        if self.supersampling < 1:
            raise ValueError("supersampling must be at least 1.")
        if len(self.centre) != 2 or not all(math.isfinite(v) for v in self.centre):
            raise ValueError("centre must contain two finite values in (x, y) order.")

    @property
    def pixel_area(self) -> float:
        """Area of one image pixel in square arcseconds."""

        return self.pixel_scale**2

    def coordinates(self) -> Array:
        """Return pixel-centre coordinates with shape ``(ny, nx, 2)``."""

        return self.subpixel_coordinates().mean(axis=2)

    def subpixel_coordinates(self) -> Array:
        """Return sub-pixel centres with shape ``(ny, nx, s**2, 2)``."""

        ny, nx = self.shape
        scale = jnp.asarray(self.pixel_scale, dtype=jnp.float64)
        centre_x, centre_y = self.centre

        x_centres = (jnp.arange(nx, dtype=jnp.float64) - (nx - 1) / 2) * scale
        y_centres = (jnp.arange(ny, dtype=jnp.float64) - (ny - 1) / 2) * scale
        x_centres = x_centres + centre_x
        y_centres = y_centres + centre_y

        sub = self.supersampling
        offsets = ((jnp.arange(sub, dtype=jnp.float64) + 0.5) / sub - 0.5) * scale
        offset_y, offset_x = jnp.meshgrid(offsets, offsets, indexing="ij")
        offsets_xy = jnp.stack((offset_x.reshape(-1), offset_y.reshape(-1)), axis=-1)

        image_y, image_x = jnp.meshgrid(y_centres, x_centres, indexing="ij")
        pixel_xy = jnp.stack((image_x, image_y), axis=-1)
        return pixel_xy[:, :, None, :] + offsets_xy[None, None, :, :]

