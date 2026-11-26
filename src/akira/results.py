"""Versioned, portable storage for Akira inference results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from akira.inference import (
    Fixed,
    FreeParameter,
    GaussianPrior,
    InferenceProblem,
    InferenceResult,
    LogUniformPrior,
    ProfileModel,
    TruncatedGaussianPrior,
    UniformPrior,
)


RESULT_FORMAT = "akira-inference-result"
RESULT_FORMAT_VERSION = 1
SUMMARY_FILENAME = "result_summary.json"
ARRAYS_FILENAME = "result_arrays.npz"


@dataclass(frozen=True, slots=True)
class StoredInferenceResult:
    """A validated result loaded from Akira's portable on-disk format."""

    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _akira_version() -> str:
    try:
        return version("project-akira")
    except PackageNotFoundError:
        return "unknown"


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
    raise TypeError(f"Unsupported model value type: {type(value).__name__}.")


def _profile_to_dict(profile: ProfileModel) -> dict[str, Any]:
    return {
        "type": profile.profile_type.__name__,
        "parameters": {
            name: _value_to_json(value) for name, value in profile.parameters.items()
        },
    }


def _parameter_definitions(problem: InferenceProblem) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "initial": parameter.initial,
            "prior": _prior_to_dict(parameter.prior),
        }
        for name, parameter in problem.parameter_items
    ]


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


def save_inference_result(
    directory: str | Path,
    problem: InferenceProblem,
    result: InferenceResult,
) -> tuple[Path, Path]:
    """Save an analytic optimization result as versioned JSON and NPZ files."""

    if result.parameter_names != problem.parameter_names:
        raise ValueError("Result parameter names do not match the inference problem.")

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "parameter_vector": np.asarray(result.parameter_vector),
        "model_image": np.asarray(result.model_image),
        "residual_map": np.asarray(problem.observation.data - result.model_image),
        "observation_data": np.asarray(problem.observation.data),
        "noise_map": np.asarray(problem.observation.noise_map),
        "mask": np.asarray(problem.observation.mask),
        "psf_kernel": np.asarray(problem.observation.psf.kernel),
    }
    if result.covariance is not None:
        arrays["covariance"] = np.asarray(result.covariance)

    summary = {
        "format": RESULT_FORMAT,
        "format_version": RESULT_FORMAT_VERSION,
        "provenance": {
            "akira_version": _akira_version(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_backend": "jax",
            "jax_version": jax.__version__,
            "inference_method": "scipy-lbfgsb",
            "objective": result.mode,
            "random_seed": None,
        },
        "problem": {
            "grid": {
                "shape": list(problem.grid.shape),
                "pixel_scale": problem.grid.pixel_scale,
                "supersampling": problem.grid.supersampling,
            },
            "observation": {
                "shape": list(problem.observation.data.shape),
                "pixel_scale": problem.observation.pixel_scale,
            },
            "model": {
                "lens": [_profile_to_dict(profile) for profile in problem.model.lens],
                "source": [
                    _profile_to_dict(profile) for profile in problem.model.source
                ],
            },
            "parameters": _parameter_definitions(problem),
        },
        "point_estimate": {
            "parameter_names": list(result.parameter_names),
            "values": [float(value) for value in np.asarray(result.parameter_vector)],
            "parameters": result.parameters,
            "log_likelihood": result.log_likelihood,
            "log_prior": result.log_prior,
            "log_posterior": result.log_posterior,
        },
        "optimization": {
            "success": result.success,
            "message": result.message,
            "evaluations": result.evaluations,
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


def load_inference_result(directory: str | Path) -> StoredInferenceResult:
    """Load and validate a version 1 Akira inference result."""

    location = Path(directory)
    summary_path = location / SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("format") != RESULT_FORMAT:
        raise ValueError(f"Not an {RESULT_FORMAT!r} result.")
    format_version = summary.get("format_version")
    if format_version != RESULT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported result format version {format_version!r}; "
            f"this Akira version supports {RESULT_FORMAT_VERSION}."
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
    return StoredInferenceResult(summary=summary, arrays=arrays)
