"""Numerical precision policy for Project Akira."""

import jax


def enable_float64() -> None:
    """Enable JAX 64-bit arrays for scientific calculations.

    This must run before callers create JAX arrays. Project Akira imports it at
    package initialization so float64 is the default throughout the package.
    """

    jax.config.update("jax_enable_x64", True)

