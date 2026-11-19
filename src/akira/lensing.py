"""Single-plane gravitational ray tracing."""

import jax.numpy as jnp
from jax import Array


def ray_trace(image_coordinates: Array, deflections: Array) -> Array:
    """Map image-plane coordinates to the source plane.

    Both inputs and the returned coordinates use ``(..., 2)`` arrays in
    ``(x, y)`` order and arcseconds.
    """

    image_coordinates = jnp.asarray(image_coordinates, dtype=jnp.float64)
    deflections = jnp.asarray(deflections, dtype=jnp.float64)
    if image_coordinates.shape != deflections.shape:
        raise ValueError("image_coordinates and deflections must have matching shapes.")
    if image_coordinates.shape[-1] != 2:
        raise ValueError("Coordinate arrays must have a final dimension of size 2.")
    return image_coordinates - deflections

