"""Composable mass and light model containers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class LensPlane:
    """A single lens plane whose component deflections add linearly."""

    mass_profiles: tuple[object, ...]

    def __post_init__(self) -> None:
        profiles = tuple(self.mass_profiles)
        if not profiles:
            raise ValueError("LensPlane requires at least one mass profile.")
        if not all(hasattr(profile, "deflection") for profile in profiles):
            raise TypeError("Every mass profile must provide deflection(coordinates).")
        object.__setattr__(self, "mass_profiles", profiles)

    def deflection(self, coordinates: Array) -> Array:
        total = jnp.zeros_like(coordinates, dtype=jnp.float64)
        for profile in self.mass_profiles:
            total = total + profile.deflection(coordinates)
        return total

    def tree_flatten(self):
        return self.mass_profiles, None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        instance = object.__new__(cls)
        object.__setattr__(instance, "mass_profiles", tuple(children))
        return instance


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class CompositeLight:
    """A surface-brightness model formed by adding light profiles."""

    profiles: tuple[object, ...]

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if not profiles:
            raise ValueError("CompositeLight requires at least one light profile.")
        if not all(hasattr(profile, "brightness") for profile in profiles):
            raise TypeError("Every light profile must provide brightness(coordinates).")
        object.__setattr__(self, "profiles", profiles)

    def brightness(self, coordinates: Array) -> Array:
        total = jnp.zeros(coordinates.shape[:-1], dtype=jnp.float64)
        for profile in self.profiles:
            total = total + profile.brightness(coordinates)
        return total

    def tree_flatten(self):
        return self.profiles, None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        instance = object.__new__(cls)
        object.__setattr__(instance, "profiles", tuple(children))
        return instance
