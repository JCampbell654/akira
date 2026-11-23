#!/usr/bin/env python3
"""Reconstruct a pixelized source for one completely fixed lens model.

No nonlinear parameters and no regularization strength are optimized. Run as

    python examples/reconstruct_fixed_lens.py --config fixed_lens_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from akira import (
    CartesianGrid,
    EllipticalPowerLaw,
    ExternalShear,
    ImagingObservation,
    LensPlane,
    PSF,
    SourceGrid,
    find_lensing_curves,
    masked_forward_operator,
    plot_caustics,
    ray_trace,
    reconstruct_from_masked_operator,
    source_grid_from_lens,
)

try:
    from examples.fit_real_data import _extent, _load_array, _report_memory_scale
except ModuleNotFoundError:  # Direct execution adds examples/, not its parent.
    from fit_real_data import _extent, _load_array, _report_memory_scale


def _fixed_value(spec: Any, name: str, default: float | None = None) -> float:
    if spec is None:
        if default is None:
            raise ValueError(f"Missing required fixed parameter {name!r}.")
        return default
    if isinstance(spec, dict):
        raise ValueError(
            f"{name} must be a single number in a fixed-lens reconstruction; "
            "prior/initial dictionaries are not accepted."
        )
    value = float(spec)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _make_fixed_lens(config: dict[str, Any]) -> LensPlane:
    lens_config = config["lens"]
    power_law = lens_config["power_law"]
    mass_profiles: list[object] = [
        EllipticalPowerLaw(
            einstein_radius=_fixed_value(
                power_law.get("einstein_radius"), "einstein_radius"
            ),
            slope=_fixed_value(power_law.get("slope"), "slope"),
            centre=(
                _fixed_value(power_law.get("centre_x"), "centre_x", 0.0),
                _fixed_value(power_law.get("centre_y"), "centre_y", 0.0),
            ),
            q=_fixed_value(power_law.get("q"), "q", 1.0),
            phi=_fixed_value(power_law.get("phi"), "phi", 0.0),
        )
    ]
    shear = lens_config.get("external_shear")
    if shear is not None:
        centre = shear.get("centre", (0.0, 0.0))
        if len(centre) != 2 or any(isinstance(value, dict) for value in centre):
            raise ValueError("external_shear.centre must contain two fixed numbers.")
        mass_profiles.append(
            ExternalShear(
                strength=_fixed_value(shear.get("strength"), "shear strength", 0.0),
                phi=_fixed_value(shear.get("phi"), "shear phi", 0.0),
                centre=(float(centre[0]), float(centre[1])),
            )
        )
    return LensPlane(tuple(mass_profiles))


def _source_grid(
    config: dict[str, Any],
    image_grid: CartesianGrid,
    lens: LensPlane,
    mask: np.ndarray,
) -> SourceGrid:
    source_config = config["source_grid"]
    shape = tuple(source_config["shape"])
    if source_config.get("auto_extent", False):
        return source_grid_from_lens(
            image_grid,
            lens,
            shape,
            mask,
            padding_fraction=float(source_config.get("padding_fraction", 0.05)),
            exclude_ray_fraction=float(
                source_config.get("exclude_ray_fraction", 0.0)
            ),
        )
    return SourceGrid(
        shape,
        float(source_config["pixel_scale"]),
        tuple(source_config.get("centre", (0.0, 0.0))),
    )


def _coverage(
    image_grid: CartesianGrid,
    source_grid: SourceGrid,
    lens: LensPlane,
    mask: np.ndarray,
    data_curvature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    image_coordinates = np.asarray(image_grid.coordinates()).reshape((-1, 2))
    active = ~mask.reshape(-1)
    traced = np.asarray(
        ray_trace(image_coordinates[active], lens.deflection(image_coordinates[active]))
    )
    ny, nx = source_grid.shape
    half_x = (nx - 1) * source_grid.pixel_scale / 2
    half_y = (ny - 1) * source_grid.pixel_scale / 2
    inside = (
        (traced[:, 0] >= source_grid.centre[0] - half_x)
        & (traced[:, 0] <= source_grid.centre[0] + half_x)
        & (traced[:, 1] >= source_grid.centre[1] - half_y)
        & (traced[:, 1] <= source_grid.centre[1] + half_y)
    )
    diagonal = np.diag(data_curvature).reshape(source_grid.shape)
    maximum = float(np.max(diagonal)) if diagonal.size else 0.0
    threshold = maximum * 1e-12
    constrained = diagonal > threshold if maximum > 0 else np.zeros_like(diagonal, bool)
    return traced, constrained, float(np.mean(inside))


def run(config_path: Path, dry_run: bool = False) -> int:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    base = config_path.parent
    data = _load_array(config["data"], base).astype(np.float64)
    noise_map = _load_array(config["noise_map"], base).astype(np.float64)
    psf_array = _load_array(config["psf"], base).astype(np.float64)
    psf_sum = float(psf_array.sum())
    if not np.isfinite(psf_sum) or psf_sum <= 0:
        raise ValueError("PSF must have a positive finite sum.")
    mask = (
        np.zeros(data.shape, dtype=bool)
        if config.get("mask") is None
        else _load_array(config["mask"], base).astype(bool)
    )
    pixel_scale = float(config["pixel_scale"])
    observation = ImagingObservation(
        data, noise_map, PSF(psf_array / psf_sum), pixel_scale, mask
    )
    image_grid = CartesianGrid(
        data.shape,
        pixel_scale,
        supersampling=int(config.get("supersampling", 1)),
    )
    lens = _make_fixed_lens(config)
    source_grid = _source_grid(config, image_grid, lens, mask)
    regularization = config["regularization"]
    kind = regularization["kind"]
    strength = _fixed_value(
        regularization.get("strength", regularization.get("initial_strength")),
        "regularization strength",
    )

    print(
        "Fixed source grid: "
        f"shape={source_grid.shape}, pixel_scale={source_grid.pixel_scale:.6g} "
        f"arcsec, centre=({source_grid.centre[0]:+.6g}, "
        f"{source_grid.centre[1]:+.6g}) arcsec.",
        flush=True,
    )
    print(f"Fixed {kind} regularization strength: {strength:.6g}", flush=True)
    _report_memory_scale(observation, source_grid, image_grid.supersampling)
    if dry_run:
        print("Dry run complete; no operator was constructed.", flush=True)
        return 0

    started = time.monotonic()
    print("Constructing the fixed lensing and PSF operator...", flush=True)
    operator = masked_forward_operator(
        image_grid, source_grid, lens, observation.psf.kernel, observation.mask
    )
    print(
        f"Operator ready after {time.monotonic() - started:.1f} s; "
        "solving the linear source reconstruction...",
        flush=True,
    )
    reconstruction = reconstruct_from_masked_operator(
        observation, source_grid, operator, kind, strength
    )
    operator.block_until_ready()
    elapsed = time.monotonic() - started

    traced, constrained, inside_fraction = _coverage(
        image_grid,
        source_grid,
        lens,
        mask,
        np.asarray(reconstruction.data_curvature),
    )
    source_uncertainty = np.sqrt(
        np.diag(np.asarray(reconstruction.posterior_covariance))
    ).reshape(source_grid.shape)
    active_pixels = int(np.sum(~mask))
    constrained_pixels = int(np.sum(constrained))

    output_directory = Path(
        config.get("output_directory", "fixed_reconstruction_output")
    )
    if not output_directory.is_absolute():
        output_directory = base / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "reconstruction_arrays.npz",
        data=data,
        noise_map=noise_map,
        mask=mask,
        model_image=np.asarray(reconstruction.model_image),
        normalized_residuals=np.asarray(reconstruction.residuals),
        source=np.asarray(reconstruction.source),
        source_posterior_std=source_uncertainty,
        source_constrained=constrained,
        ray_traced_coordinates=traced,
    )
    summary = {
        "success": True,
        "message": "Fixed linear source reconstruction completed.",
        "elapsed_seconds": elapsed,
        "chi_squared": reconstruction.chi_squared,
        "reduced_chi_squared_per_unmasked_pixel": (
            reconstruction.chi_squared / active_pixels
        ),
        "log_source_evidence": reconstruction.log_evidence,
        "regularization_kind": kind,
        "regularization_strength": strength,
        "unmasked_pixels": active_pixels,
        "source_pixels": source_grid.size,
        "constrained_source_pixels": constrained_pixels,
        "ray_fraction_inside_source_grid": inside_fraction,
        "source_grid": {
            "shape": list(source_grid.shape),
            "pixel_scale": source_grid.pixel_scale,
            "centre": list(source_grid.centre),
        },
    }
    (output_directory / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    caustic_config = config.get("caustic_grid", {})
    caustic_grid = CartesianGrid(
        tuple(caustic_config.get("shape", (301, 301))),
        float(caustic_config.get("pixel_scale", pixel_scale / 5)),
        tuple(caustic_config.get("centre", image_grid.centre)),
    )
    curves = find_lensing_curves(lens, caustic_grid)
    image_extent = _extent(image_grid.shape, pixel_scale, image_grid.centre)
    source_extent = _extent(
        source_grid.shape, source_grid.pixel_scale, source_grid.centre
    )
    residuals = np.asarray(reconstruction.residuals).copy()
    residuals[mask] = np.nan
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    panels = (
        (data, "data", "jet", image_extent),
        (reconstruction.model_image, "fixed-lens model", "jet", image_extent),
        (residuals, "normalized residual", "coolwarm", image_extent),
        (reconstruction.source, "reconstructed source", "jet", source_extent),
        (source_uncertainty, "source posterior uncertainty", "jet", source_extent),
        (constrained, "source coverage", "gray_r", source_extent),
    )
    for axis, (values, title, colour_map, extent) in zip(axes.flat, panels):
        shown = axis.imshow(values, origin="lower", extent=extent, cmap=colour_map)
        axis.set_title(title)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    plot_caustics(axes[1, 0], curves, color="white", linewidth=1.5)

    # Restrict the plot to the image extent
    axes[1, 0].set_xlim(source_extent[0], source_extent[1])
    axes[1, 0].set_ylim(source_extent[2], source_extent[3])

    # Downsample only for plotting; every coordinate is retained in the NPZ.
    stride = max(1, traced.shape[0] // 10_000)
    axes[1, 2].scatter(
        traced[::stride, 0],
        traced[::stride, 1],
        s=1,
        c="tab:orange",
        alpha=0.12,
        linewidths=0,
        label="ray-traced mask",
    )
    axes[1, 2].legend(loc="best", fontsize="small")
    for axis in axes[0]:
        axis.set_xlabel("image x [arcsec]")
    axes[0, 0].set_ylabel("image y [arcsec]")
    for axis in axes[1]:
        axis.set_xlabel("source x [arcsec]")
    axes[1, 0].set_ylabel("source y [arcsec]")
    figure.savefig(output_directory / "reconstruction_diagnostics.png", dpi=180)
    plt.close(figure)

    print(f"Reconstruction completed in {elapsed:.1f} s.", flush=True)
    print(f"Chi-squared / unmasked pixel: {summary['reduced_chi_squared_per_unmasked_pixel']:.6g}")
    print(f"Rays inside source grid: {100 * inside_fraction:.4f}%")
    print(f"Constrained source pixels: {constrained_pixels}/{source_grid.size}")
    print(f"Results written to: {output_directory}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and report memory without reconstructing",
    )
    arguments = parser.parse_args()
    return run(arguments.config, dry_run=arguments.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
