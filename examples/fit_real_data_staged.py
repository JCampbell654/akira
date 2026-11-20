#!/usr/bin/env python3
"""Fit real imaging by progressively releasing lens-model parameters."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from akira import (
    CartesianGrid,
    EllipticalPowerLaw,
    ExternalShear,
    FreeParameter,
    ImagingObservation,
    PSF,
    PixelizedLensModel,
    ProfileModel,
    SourceGrid,
    fit_pixelized_model,
    source_grid_from_lens,
)
try:
    from examples.fit_real_data import (
        _ProgressDisplay,
        _load_array,
        _make_model,
        _prior,
        _report_memory_scale,
        _save_results,
    )
except ModuleNotFoundError:  # Direct execution adds examples/, not its parent.
    from fit_real_data import (
        _ProgressDisplay,
        _load_array,
        _make_model,
        _prior,
        _report_memory_scale,
        _save_results,
    )


STAGES = (
    ("01_regularization", frozenset()),
    ("02_radial_mass", frozenset({"einstein_radius", "slope"})),
    (
        "03_elliptical_mass",
        frozenset({"einstein_radius", "slope", "centre_x", "centre_y", "q", "phi"}),
    ),
    (
        "04_mass_and_shear",
        frozenset(
            {
                "einstein_radius",
                "slope",
                "centre_x",
                "centre_y",
                "q",
                "phi",
                "shear_strength",
                "shear_phi",
            }
        ),
    ),
)


def _initial(spec: float | dict[str, Any]) -> float:
    return float(spec["initial"] if isinstance(spec, dict) else spec)


def _stage_parameter(
    spec: float | dict[str, Any],
    value: float,
    free: bool,
) -> float | FreeParameter:
    if free and isinstance(spec, dict):
        return FreeParameter(value, _prior(spec["prior"]))
    return value


def _starting_values(config: dict[str, Any]) -> dict[str, float]:
    power_law = config["lens"]["power_law"]
    shear = config["lens"]["external_shear"]
    return {
        "einstein_radius": _initial(power_law["einstein_radius"]),
        "slope": _initial(power_law["slope"]),
        "centre_x": _initial(power_law.get("centre_x", 0.0)),
        "centre_y": _initial(power_law.get("centre_y", 0.0)),
        "q": _initial(power_law.get("q", 1.0)),
        "phi": _initial(power_law.get("phi", 0.0)),
        "shear_strength": _initial(shear.get("strength", 0.0)),
        "shear_phi": _initial(shear.get("phi", 0.0)),
    }


def _build_stage_model(
    config: dict[str, Any],
    source_grid: SourceGrid,
    values: dict[str, float],
    free: frozenset[str],
    regularization_strength: float,
) -> PixelizedLensModel:
    power_law = config["lens"]["power_law"]
    shear = config["lens"]["external_shear"]
    return PixelizedLensModel(
        lens=(
            ProfileModel(
                EllipticalPowerLaw,
                einstein_radius=_stage_parameter(
                    power_law["einstein_radius"], values["einstein_radius"],
                    "einstein_radius" in free,
                ),
                slope=_stage_parameter(
                    power_law["slope"], values["slope"], "slope" in free
                ),
                centre=(
                    _stage_parameter(
                        power_law.get("centre_x", 0.0), values["centre_x"],
                        "centre_x" in free,
                    ),
                    _stage_parameter(
                        power_law.get("centre_y", 0.0), values["centre_y"],
                        "centre_y" in free,
                    ),
                ),
                q=_stage_parameter(power_law.get("q", 1.0), values["q"], "q" in free),
                phi=_stage_parameter(
                    power_law.get("phi", 0.0), values["phi"], "phi" in free
                ),
            ),
            ProfileModel(
                ExternalShear,
                strength=_stage_parameter(
                    shear.get("strength", 0.0), values["shear_strength"],
                    "shear_strength" in free,
                ),
                phi=_stage_parameter(
                    shear.get("phi", 0.0), values["shear_phi"],
                    "shear_phi" in free,
                ),
                centre=tuple(shear.get("centre", (0.0, 0.0))),
            ),
        ),
        source_grid=source_grid,
        regularization_kind=config["regularization"]["kind"],
        regularization_strength=regularization_strength,
    )


def _update_values(values: dict[str, float], result) -> None:
    names = {
        "lens.0.EllipticalPowerLaw.einstein_radius": "einstein_radius",
        "lens.0.EllipticalPowerLaw.slope": "slope",
        "lens.0.EllipticalPowerLaw.centre.x": "centre_x",
        "lens.0.EllipticalPowerLaw.centre.y": "centre_y",
        "lens.0.EllipticalPowerLaw.q": "q",
        "lens.0.EllipticalPowerLaw.phi": "phi",
        "lens.1.ExternalShear.strength": "shear_strength",
        "lens.1.ExternalShear.phi": "shear_phi",
    }
    for result_name, local_name in names.items():
        if result_name in result.parameters:
            values[local_name] = result.parameters[result_name]


def run(config_path: Path, dry_run: bool = False) -> int:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    base = config_path.parent
    data = _load_array(config["data"], base).astype(np.float64)
    noise = _load_array(config["noise_map"], base).astype(np.float64)
    psf_array = _load_array(config["psf"], base).astype(np.float64)
    psf = PSF(psf_array / psf_array.sum())
    mask = None
    if config.get("mask") is not None:
        mask = _load_array(config["mask"], base).astype(bool)
    pixel_scale = float(config["pixel_scale"])
    observation = ImagingObservation(data, noise, psf, pixel_scale, mask)
    image_grid = CartesianGrid(
        data.shape, pixel_scale, supersampling=int(config.get("supersampling", 1))
    )
    source_config = config["source_grid"]
    source_shape = tuple(source_config["shape"])
    if source_config.get("auto_extent", False):
        temporary_grid = SourceGrid(source_shape, 1.0)
        reference_model = _make_model(config, temporary_grid)
        reference_lens, _ = reference_model.build(reference_model.initial_vector)
        source_grid = source_grid_from_lens(
            image_grid,
            reference_lens,
            source_shape,
            observation.mask,
            float(source_config.get("padding_fraction", 0.05)),
            float(source_config.get("exclude_ray_fraction", 0.0)),
        )
        print(
            f"Automatic source grid: scale={source_grid.pixel_scale:.6g} arcsec, "
            f"centre=({source_grid.centre[0]:+.6g}, {source_grid.centre[1]:+.6g})",
            flush=True,
        )
    else:
        source_grid = SourceGrid(
            source_shape,
            float(source_config["pixel_scale"]),
            tuple(source_config.get("centre", (0.0, 0.0))),
        )
    _report_memory_scale(observation, source_grid, image_grid.supersampling)
    if dry_run:
        print("Dry run complete; no fitting stages were started.")
        return 0

    strength_bounds = config["regularization"]["strength_bounds"]
    log_bounds = tuple(math.log(value) for value in strength_bounds)
    values = _starting_values(config)
    strength = float(config["regularization"]["initial_strength"])
    staged_output = config.get("staged_output_directory")
    if staged_output is None:
        ordinary_output = Path(config.get("output_directory", "fit_output"))
        staged_output = ordinary_output.with_name(ordinary_output.name + "_staged")
    output = Path(staged_output)
    if not output.is_absolute():
        output = base / output
    max_iterations = int(config.get("max_iterations", 500))
    max_line_search = int(config.get("max_line_search_steps", 100))
    final_result = None
    for stage_number, (stage_name, free) in enumerate(STAGES, start=1):
        print(f"\n=== Stage {stage_number}/{len(STAGES)}: {stage_name} ===", flush=True)
        print("Free lens parameters: " + (", ".join(sorted(free)) or "none"), flush=True)
        model = _build_stage_model(config, source_grid, values, free, strength)
        progress = _ProgressDisplay(report_every=int(config.get("progress_every", 5)))
        progress.begin()
        try:
            result = fit_pixelized_model(
                model,
                observation,
                image_grid,
                objective="evidence",
                optimize_regularization_strength=True,
                log_strength_bounds=log_bounds,
                progress_callback=progress.update,
                max_iterations=max_iterations,
                max_line_search_steps=max_line_search,
            )
        finally:
            progress.finish()
        _save_results(
            output / stage_name,
            config,
            observation,
            source_grid,
            image_grid,
            result,
        )
        print(f"Stage result: success={result.success}; {result.message}", flush=True)
        _update_values(values, result)
        strength = result.regularization_strength
        final_result = result
        if not result.success:
            print(
                "Stopping before releasing additional parameters. Inspect this "
                "stage's diagnostics and summary.",
                flush=True,
            )
            return 2

    assert final_result is not None
    _save_results(output, config, observation, source_grid, image_grid, final_result)
    print(f"\nAll stages succeeded. Final results written to: {output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return run(arguments.config, arguments.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
