"""Project Akira: differentiable strong gravitational lens modelling."""

from akira._precision import enable_float64 as _enable_float64

_enable_float64()

from akira.coordinates import CartesianGrid, degrees_to_radians, radians_to_degrees
from akira.fit import (
    EPLSersicFitResult,
    EPLSersicParameters,
    FitResult,
    fit_epl_sersic_parameters,
    fit_sis_einstein_radius,
)
from akira.imaging import ImagingObservation, PSF, gaussian_psf
from akira.inference import (
    Fixed,
    FreeParameter,
    GaussianPrior,
    InferenceResult,
    LensModel,
    LogUniformPrior,
    ProfileModel,
    TruncatedGaussianPrior,
    UniformPrior,
    fit_model,
)
from akira.lensing import ray_trace
from akira.likelihood import gaussian_log_likelihood
from akira.model import CompositeLight, LensPlane
from akira.profiles import (
    EllipticalPowerLaw,
    ExternalShear,
    Gaussian,
    SIS,
    Sersic,
)
from akira.render import render, render_sis_gaussian
from akira.simulate import simulate_imaging

__all__ = [
    "CartesianGrid",
    "CompositeLight",
    "EllipticalPowerLaw",
    "EPLSersicFitResult",
    "EPLSersicParameters",
    "ExternalShear",
    "Fixed",
    "FreeParameter",
    "Gaussian",
    "GaussianPrior",
    "ImagingObservation",
    "LensPlane",
    "LensModel",
    "LogUniformPrior",
    "InferenceResult",
    "FitResult",
    "PSF",
    "ProfileModel",
    "SIS",
    "Sersic",
    "TruncatedGaussianPrior",
    "UniformPrior",
    "degrees_to_radians",
    "fit_sis_einstein_radius",
    "fit_model",
    "fit_epl_sersic_parameters",
    "gaussian_log_likelihood",
    "gaussian_psf",
    "radians_to_degrees",
    "ray_trace",
    "render",
    "render_sis_gaussian",
    "simulate_imaging",
]

__version__ = "0.1.0.dev0"
