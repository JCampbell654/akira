"""Project Akira: differentiable strong gravitational lens modelling."""

from akira._precision import enable_float64 as _enable_float64

_enable_float64()

from akira.coordinates import CartesianGrid, degrees_to_radians, radians_to_degrees
from akira.fit import FitResult, fit_sis_einstein_radius
from akira.imaging import ImagingObservation, PSF, gaussian_psf
from akira.lensing import ray_trace
from akira.likelihood import gaussian_log_likelihood
from akira.profiles import Gaussian, SIS
from akira.render import render_sis_gaussian
from akira.simulate import simulate_imaging

__all__ = [
    "CartesianGrid",
    "Gaussian",
    "ImagingObservation",
    "FitResult",
    "PSF",
    "SIS",
    "degrees_to_radians",
    "fit_sis_einstein_radius",
    "gaussian_log_likelihood",
    "gaussian_psf",
    "radians_to_degrees",
    "ray_trace",
    "render_sis_gaussian",
    "simulate_imaging",
]

__version__ = "0.1.0.dev0"
