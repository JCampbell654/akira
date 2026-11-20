"""Project Akira: differentiable strong gravitational lens modelling."""

from akira._precision import enable_float64 as _enable_float64

_enable_float64()

from akira.coordinates import CartesianGrid, degrees_to_radians, radians_to_degrees
from akira.caustics import (
    LensingCurves,
    find_lensing_curves,
    lens_mapping_determinant,
    plot_caustics,
)
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
from akira.pixelized import (
    PixelizedReconstruction,
    SourceGrid,
    apply_psf_to_operator,
    bilinear_interpolation_matrix,
    forward_operator,
    lensing_operator,
    masked_forward_operator,
    reconstruct_source,
    reconstruct_from_operator,
    reconstruct_from_masked_operator,
    regularization_matrix,
    solve_source,
)
from akira.pixelized_inference import (
    OptimizationProgress,
    PixelizedInferenceResult,
    PixelizedLensModel,
    fit_pixelized_model,
)
from akira.profiles import (
    EllipticalPowerLaw,
    ExternalShear,
    Gaussian,
    SIS,
    Sersic,
)
from akira.render import render, render_sis_gaussian
from akira.regularization import (
    RegularizationOptimization,
    gaussian_source_log_evidence,
    optimize_regularization,
)
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
    "LensingCurves",
    "LensModel",
    "LogUniformPrior",
    "InferenceResult",
    "FitResult",
    "PSF",
    "OptimizationProgress",
    "PixelizedReconstruction",
    "PixelizedInferenceResult",
    "PixelizedLensModel",
    "ProfileModel",
    "RegularizationOptimization",
    "SIS",
    "Sersic",
    "SourceGrid",
    "TruncatedGaussianPrior",
    "UniformPrior",
    "degrees_to_radians",
    "apply_psf_to_operator",
    "bilinear_interpolation_matrix",
    "fit_sis_einstein_radius",
    "find_lensing_curves",
    "fit_model",
    "fit_pixelized_model",
    "forward_operator",
    "fit_epl_sersic_parameters",
    "gaussian_log_likelihood",
    "gaussian_source_log_evidence",
    "gaussian_psf",
    "lensing_operator",
    "masked_forward_operator",
    "lens_mapping_determinant",
    "optimize_regularization",
    "plot_caustics",
    "radians_to_degrees",
    "ray_trace",
    "reconstruct_source",
    "reconstruct_from_operator",
    "reconstruct_from_masked_operator",
    "regularization_matrix",
    "solve_source",
    "render",
    "render_sis_gaussian",
    "simulate_imaging",
]

__version__ = "0.1.0.dev0"
