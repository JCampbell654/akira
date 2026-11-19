"""Project Akira: differentiable strong gravitational lens modelling."""

from akira._precision import enable_float64 as _enable_float64

_enable_float64()

from akira.coordinates import CartesianGrid, degrees_to_radians, radians_to_degrees
from akira.lensing import ray_trace
from akira.profiles import Gaussian, SIS
from akira.render import render_sis_gaussian

__all__ = [
    "CartesianGrid",
    "Gaussian",
    "SIS",
    "degrees_to_radians",
    "radians_to_degrees",
    "ray_trace",
    "render_sis_gaussian",
]

__version__ = "0.1.0.dev0"

