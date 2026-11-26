"""Portable storage and resumption for pixelized-source inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from akira.coordinates import CartesianGrid
from akira.imaging import ImagingObservation, PSF
from akira.inference import (
    Fixed,
    FreeParameter,
    GaussianPrior,
    LogUniformPrior,
    ProfileModel,
    TruncatedGaussianPrior,
    UniformPrior,
)
from akira.pixelized import SourceGrid
from akira.pixelized_inference import (
    PixelizedInferenceProblem,
    PixelizedInferenceResult,
    PixelizedLensModel,
)
from akira.profiles import EllipticalPowerLaw, ExternalShear, Gaussian, SIS, Sersic
from akira.results import ARRAYS_FILENAME, SUMMARY_FILENAME, _akira_version


PIXELIZED_RESULT_FORMAT = "akira-pixelized-inference-result"
PIXELIZED_RESULT_FORMAT_VERSION = 1


_PROFILE_TYPES = {
    profile.__name__: profile
    for profile in (SIS, Gaussian, EllipticalPowerLaw, ExternalShear, Sersic)
}


def _prior_to_dict(prior: object) -> dict[str, Any]:
    if isinstance(prior, UniformPrior):
        return {"type": "uniform", "lower": prior.lower, "upper": prior.upper}
    if isinstance(prior, GaussianPrior):
        return {"type": "gaussian", "mean": prior.mean, "sigma": prior.sigma}
    if isinstance(prior, LogUniformPrior):
        return {"type": "log_uniform", "lower": prior.lower, "upper": prior.upper}
    if isinstance(prior, TruncatedGaussianPrior):
        return {
            "type": "truncated_gaussian",
            "mean": prior.mean,
            "sigma": prior.sigma,
            "lower": prior.lower,
            "upper": prior.upper,
        }
    raise TypeError(f"Unsupported prior type: {type(prior).__name__}.")


def _prior_from_dict(specification: dict[str, Any]):
    kind = specification.get("type")
    if kind == "uniform":
        return UniformPrior(specification["lower"], specification["upper"])
    if kind == "gaussian":
        return GaussianPrior(specification["mean"], specification["sigma"])
    if kind == "log_uniform":
        return LogUniformPrior(specification["lower"], specification["upper"])
    if kind == "truncated_gaussian":
        return TruncatedGaussianPrior(
            specification["mean"],
            specification["sigma"],
            specification["lower"],
            specification["upper"],
        )
    raise ValueError(f"Unsupported stored prior type: {kind!r}.")


def _value_to_json(value: Any) -> Any:
    if isinstance(value, FreeParameter):
        return {
            "status": "free",
            "initial": value.initial,
            "prior": _prior_to_dict(value.prior),
        }
    if isinstance(value, Fixed):
        return {"status": "fixed", "value": _value_to_json(value.value)}
    if isinstance(value, tuple):
        return [_value_to_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    raise TypeError(f"Unsupported model value type: {type(value).__name__}.")


def _value_from_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_value_from_json(item) for item in value)
    if isinstance(value, dict):
        status = value.get("status")
        if status == "free":
            return FreeParameter(value["initial"], _prior_from_dict(value["prior"]))
        if status == "fixed":
            return Fixed(_value_from_json(value["value"]))
        raise ValueError(f"Invalid stored model value: {value!r}.")
    return value


def _profile_to_dict(profile: ProfileModel) -> dict[str, Any]:
    return {
        "type": profile.profile_type.__name__,
        "parameters": {
            name: _value_to_json(value) for name, value in profile.parameters.items()
        },
    }


def _profile_from_dict(specification: dict[str, Any]) -> ProfileModel:
    name = specification.get("type")
    try:
        profile_type = _PROFILE_TYPES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported stored profile type: {name!r}.") from error
    return ProfileModel(
        profile_type,
        **{
            key: _value_from_json(value)
            for key, value in specification["parameters"].items()
        },
    )


def _array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "file": ARRAYS_FILENAME,
            "key": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        for name, array in arrays.items()
    }


@dataclass(frozen=True, slots=True)
class StoredPixelizedInferenceResult:
    """A loaded fit with its complete problem and a sampling convenience API."""

    problem: PixelizedInferenceProblem
    parameter_names: tuple[str, ...]
    parameter_vector: np.ndarray
    regularization_strength: float
    optimized_regularization_strength: bool
    log_strength_bounds: tuple[float, float] | None
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def sample(self, **kwargs: Any):
        """Resume posterior sampling from the stored point estimate."""

        from akira.sampling import sample_pixelized_model

        options = dict(kwargs)
        options.setdefault("initial_vector", self.parameter_vector)
        options.setdefault(
            "sample_regularization_strength",
            self.optimized_regularization_strength,
        )
        if self.optimized_regularization_strength:
            options.setdefault("log_strength_bounds", self.log_strength_bounds)
        return sample_pixelized_model(
            self.problem.model,
            self.problem.observation,
            self.problem.image_grid,
            **options,
        )


def save_pixelized_inference_result(
    directory: str | Path,
    problem: PixelizedInferenceProblem,
    result: PixelizedInferenceResult,
) -> tuple[Path, Path]:
    """Save a fit and every scientific input required to resume sampling."""

    expected_names = problem.parameter_names
    if result.optimized_regularization_strength:
        expected_names += ("regularization.log_strength",)
    if result.parameter_names != expected_names:
        raise ValueError("Result parameter names do not match the inference problem.")

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    reconstruction = result.reconstruction
    arrays = {
        "parameter_vector": np.asarray(result.parameter_vector),
        "observation_data": np.asarray(problem.observation.data),
        "noise_map": np.asarray(problem.observation.noise_map),
        "mask": np.asarray(problem.observation.mask),
        "psf_kernel": np.asarray(problem.observation.psf.kernel),
        "model_image": np.asarray(reconstruction.model_image),
        "residual_map": np.asarray(reconstruction.residuals),
        "source": np.asarray(reconstruction.source),
    }
    if result.covariance is not None:
        arrays["covariance"] = np.asarray(result.covariance)

    source_grid = problem.model.source_grid
    grid = problem.image_grid
    summary = {
        "format": PIXELIZED_RESULT_FORMAT,
        "format_version": PIXELIZED_RESULT_FORMAT_VERSION,
        "provenance": {
            "akira_version": _akira_version(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_backend": "jax",
            "jax_version": jax.__version__,
            "inference_method": "scipy-lbfgsb",
            "objective": result.objective,
        },
        "problem": {
            "image_grid": {
                "shape": list(grid.shape),
                "pixel_scale": grid.pixel_scale,
                "centre": list(grid.centre),
                "supersampling": grid.supersampling,
            },
            "observation": {
                "shape": list(problem.observation.data.shape),
                "pixel_scale": problem.observation.pixel_scale,
            },
            "model": {
                "lens": [_profile_to_dict(profile) for profile in problem.model.lens],
                "source_grid": {
                    "shape": list(source_grid.shape),
                    "pixel_scale": source_grid.pixel_scale,
                    "centre": list(source_grid.centre),
                },
                "regularization_kind": problem.model.regularization_kind,
                "regularization_strength": _value_to_json(
                    problem.model.regularization_strength
                ),
            },
        },
        "point_estimate": {
            "parameter_names": list(result.parameter_names),
            "values": [float(value) for value in np.asarray(result.parameter_vector)],
            "parameters": result.parameters,
            "regularization_strength": result.regularization_strength,
            "log_likelihood": result.log_likelihood,
            "log_prior": result.log_prior,
            "log_source_evidence": result.log_source_evidence,
            "objective_value": result.objective_value,
        },
        "optimization": {
            "success": result.success,
            "message": result.message,
            "evaluations": result.evaluations,
            "optimized_regularization_strength": (
                result.optimized_regularization_strength
            ),
            "log_strength_bounds": (
                list(result.log_strength_bounds)
                if result.log_strength_bounds is not None
                else None
            ),
        },
        "sampling": None,
        "arrays": _array_manifest(arrays),
    }

    arrays_path = output / ARRAYS_FILENAME
    summary_path = output / SUMMARY_FILENAME
    np.savez_compressed(arrays_path, **arrays)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary_path, arrays_path


def load_pixelized_inference_result(
    directory: str | Path,
) -> StoredPixelizedInferenceResult:
    """Load, validate, and reconstruct a saved pixelized inference problem."""

    location = Path(directory)
    summary = json.loads((location / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    if summary.get("format") != PIXELIZED_RESULT_FORMAT:
        raise ValueError(f"Not an {PIXELIZED_RESULT_FORMAT!r} result.")
    if summary.get("format_version") != PIXELIZED_RESULT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported pixelized result format version "
            f"{summary.get('format_version')!r}."
        )

    manifest = summary.get("arrays")
    if not isinstance(manifest, dict):
        raise ValueError("Result array manifest is missing or invalid.")
    arrays: dict[str, np.ndarray] = {}
    archives: dict[str, Any] = {}
    try:
        for name, specification in manifest.items():
            filename = specification.get("file")
            key = specification.get("key")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(f"Invalid array filename for {name!r}.")
            if not isinstance(key, str):
                raise ValueError(f"Invalid array key for {name!r}.")
            if filename not in archives:
                archives[filename] = np.load(location / filename, allow_pickle=False)
            archive = archives[filename]
            if key not in archive:
                raise ValueError(f"Array key {key!r} is missing from {filename}.")
            array = np.asarray(archive[key])
            if list(array.shape) != specification.get("shape"):
                raise ValueError(f"Stored shape for array {name!r} does not match.")
            if str(array.dtype) != specification.get("dtype"):
                raise ValueError(f"Stored dtype for array {name!r} does not match.")
            arrays[name] = array
    finally:
        for archive in archives.values():
            archive.close()

    problem_specification = summary["problem"]
    model_specification = problem_specification["model"]
    source_specification = model_specification["source_grid"]
    model = PixelizedLensModel(
        lens=tuple(
            _profile_from_dict(profile) for profile in model_specification["lens"]
        ),
        source_grid=SourceGrid(
            shape=tuple(source_specification["shape"]),
            pixel_scale=source_specification["pixel_scale"],
            centre=tuple(source_specification["centre"]),
        ),
        regularization_kind=model_specification["regularization_kind"],
        regularization_strength=_value_from_json(
            model_specification["regularization_strength"]
        ),
    )
    observation_specification = problem_specification["observation"]
    observation = ImagingObservation(
        arrays["observation_data"],
        arrays["noise_map"],
        PSF(arrays["psf_kernel"]),
        observation_specification["pixel_scale"],
        arrays["mask"],
    )
    grid_specification = problem_specification["image_grid"]
    image_grid = CartesianGrid(
        shape=tuple(grid_specification["shape"]),
        pixel_scale=grid_specification["pixel_scale"],
        centre=tuple(grid_specification["centre"]),
        supersampling=grid_specification["supersampling"],
    )
    problem = PixelizedInferenceProblem(model, observation, image_grid)
    optimization = summary["optimization"]
    bounds = optimization["log_strength_bounds"]
    return StoredPixelizedInferenceResult(
        problem=problem,
        parameter_names=tuple(summary["point_estimate"]["parameter_names"]),
        parameter_vector=arrays["parameter_vector"],
        regularization_strength=summary["point_estimate"][
            "regularization_strength"
        ],
        optimized_regularization_strength=optimization[
            "optimized_regularization_strength"
        ],
        log_strength_bounds=tuple(bounds) if bounds is not None else None,
        summary=summary,
        arrays=arrays,
    )
