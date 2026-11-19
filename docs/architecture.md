# Architecture and Decision Log

This document is the durable record of architectural decisions for the lens-modelling package. Update it when a decision is accepted; do not rely on chat history alone for important choices.

## Design principles

1. **Correctness first:** conventions, units, coordinates, and profile equations must be explicit and tested.
2. **Fast by construction:** core calculations should be vectorized, compilable where practical, and free of unnecessary allocations.
3. **Composable models:** mass, light, data, instrument, and inference components should be independently replaceable.
4. **Backend discipline:** scientific APIs should not be tightly coupled to one array or inference library without a measured reason.
5. **Reproducibility:** configurations, random seeds, dependencies, and fitted results should be serializable.
6. **Benchmark honestly:** compare both numerical accuracy and runtime on representative lensing problems.

## Coordinate, unit, and ellipticity conventions

The public scientific coordinate system is Cartesian `(x, y)` in angular units:

- positions, offsets, pixel scales, radii, and Einstein radii are measured in arcseconds;
- `x` increases along the positive horizontal coordinate and `y` along the positive vertical coordinate;
- scientific coordinate vectors are ordered `(x, y)`, even though image arrays are indexed `[row, column]` or `[y, x]`;
- axis ratio is `q = b / a`, with `0 < q <= 1`;
- `phi` is the major-axis position angle, increasing anticlockwise from the positive x-axis;
- because an ellipse is unchanged by a 180-degree rotation, `phi` is periodic modulo pi.

For coordinates relative to a profile centre, the major-axis-aligned coordinates are defined by

```text
x_major =  dx cos(phi) + dy sin(phi)
y_minor = -dx sin(phi) + dy cos(phi)
```

Therefore `phi = 0` places the major axis along positive x, and `phi = pi / 2` places it along positive y. The public and internal angular unit for `phi` is radians. Explicit conversion helpers will be provided for degrees.

`x_major` and `y_minor` are internal intermediate values only. Users always provide positions in the public `(x, y)` coordinate system together with a profile `centre`, `q`, and `phi`; they are never asked to construct or supply profile-aligned coordinates.

Numerical inputs use documented implicit units rather than runtime unit objects: a plain value of `1.2` for a position or radius means `1.2 arcsec`. Version 1 will provide explicit conversion helpers at the API boundary rather than passing objects such as `astropy.units.Quantity` through numerical kernels.

The default elliptical radius is the area-preserving (circularized) radius

```text
R_ell = sqrt(q * x_major**2 + y_minor**2 / q)
```

so an `R_ell = R` contour has semi-axes `R / sqrt(q)` and `R * sqrt(q)`, axis ratio `q`, and area `pi * R**2`. This calculation is internal. Public profile constructors expose physically named scale parameters such as `effective_radius` or `einstein_radius`, plus `centre`, `q`, and `phi`. Conversion helpers will be provided for catalogues that quote semi-major-axis radii. Profile-specific formula conventions must still be documented where the literature definition requires a different normalization.

## Proposed component boundaries

- `coordinates`: grids, geometry, transforms, and supersampling.
- `profiles.mass`: deflection, convergence, shear, and potential calculations.
- `profiles.light`: analytic surface-brightness profiles.
- `raytrace`: single- and multi-plane lens equations.
- `imaging`: detector sampling, PSF convolution, masks, and noise models.
- `model`: parameterized composition of lens systems and observations.
- `inference`: likelihoods and adapters for optimizers and samplers.
- `linear`: pixelized sources and other linear inversions.
- `validation`: reference solutions and cross-package comparisons.

These names are provisional and should be tested against real usage before the package structure is fixed.

## Version 1 scientific scope

- Working project designation: Project Akira.
- BSD-3-Clause open-source license.
- Python 3.12 or later.
- JAX-first numerical implementation with CPU and GPU execution.
- NumPy/SciPy reference calculations for validation, not a general interchangeable backend.
- Single-plane lensing with analytic mass and light profiles.
- Initial mass profiles: singular isothermal sphere, elliptical power law, NFW, truncated NFW, and external shear.
- Initial light profiles: Gaussian and elliptical Sersic. Multiple Gaussian components can be composed into an MGE-like model.
- Regular Cartesian pixelized sources with Warren-Dye semi-linear inversion.
- Zeroth-order, gradient, and curvature regularization.
- Adaptive Delaunay or Voronoi source grids are a future extension, not part of version 1.
- Optimistix for JAX-native optimization, with a SciPy optimization fallback.
- BlackJAX for gradient-based HMC/NUTS posterior sampling and an optional dynesty adapter for nested sampling.

For power-law mass profiles, `slope` (or `gamma`) denotes the three-dimensional density slope in `rho(r) proportional to r**(-gamma)`. The isothermal value is exactly `gamma = 2`. Any projected-slope representation must use a different explicit name and document its conversion.

The implemented elliptical power-law follows Tessore and Metcalf (2015), with
projected slope `t = gamma - 1`. Its public Einstein radius is circularized;
the paper's semi-minor-axis scale is `b = einstein_radius * sqrt(q)`. External
shear uses `gamma1 = strength*cos(2*phi)` and
`gamma2 = strength*sin(2*phi)`. Sérsic `intensity` is the brightness at the
circularized half-light radius `effective_radius`.

## Decisions still required

- Final package/distribution name. The working project designation is Project Akira.
- CPU/GPU benchmarking targets and acceptable numerical tolerances.
- Detailed future design for adaptive pixelized source grids.

## Decision log

### 2026-11-19 — Project purpose established

Build a fast strong-gravitational-lens modelling code in the spirit of PyAutoLens or HercuLens. Python is expected to be the public interface. Exact backend and package architecture remain open pending focused design and benchmarking.

### 2026-11-19 — Angular and ellipticity conventions established

Use arcseconds for positions, pixel scales, radii, and Einstein radii. Describe elliptical profiles using the minor-to-major axis ratio `q = b / a` and major-axis position angle `phi`, measured anticlockwise from positive x and periodic modulo pi. Scientific coordinates use `(x, y)` ordering; image-array indexing remains `[y, x]`.

### 2026-11-19 — Initial version 1 scope established

Use the working designation Project Akira and release the eventual package under an open-source license to be selected. Support Python 3.12 and later. Use radians for `phi` with explicit degree conversion helpers. Begin with singular isothermal sphere, elliptical power-law, NFW, truncated NFW, and external-shear mass profiles, plus Gaussian and elliptical Sersic light profiles. Version 1 pixelized reconstruction uses regular Cartesian source grids; adaptive Delaunay and Voronoi grids remain a future extension.

### 2026-11-19 — Core technology and integration choices established

Use a JAX-first production core with independent NumPy/SciPy reference calculations for validation rather than a general backend abstraction. License the project under BSD-3-Clause. Use the area-preserving elliptical-radius convention by default. Provide Optimistix and SciPy optimization integrations, BlackJAX HMC/NUTS integration, and an optional dynesty nested-sampling adapter. Treat profile-aligned coordinates and elliptical-radius calculations as internal implementation details rather than user inputs.

### 2026-11-19 — Power-law slope convention established

All power-law mass-profile slopes describe the three-dimensional density, `rho(r) proportional to r**(-gamma)`. An isothermal density profile has `gamma = 2`.

### 2026-11-19 — Initial imaging and inference conventions established

PSF kernels are explicitly normalized and odd-sized. Detector convolution is zero-padded linear convolution, not circular convolution. Image masks use `True` for excluded pixels and are applied to likelihood evaluation after the full convolved model is formed. Independent Gaussian image likelihoods include their normalization term. Simulated noise is generated only from an explicit random seed. The initial verified inference path fits an SIS Einstein radius with a fixed Gaussian source using JAX gradients and SciPy L-BFGS-B.

### 2026-11-19 — Elliptical analytic vertical slice implemented

Implement the Tessore--Metcalf elliptical power-law deflection using a fixed,
JAX-compatible Fourier recurrence, plus external shear and elliptical Sérsic
light. Compose profiles by summing mass deflections and light brightnesses.
Validate the first multi-parameter inference path by fitting Einstein radius,
3D density slope, and the two source-centre coordinates while holding the
remaining analytic shape parameters fixed.

### 2026-11-19 — General parameter and prior framework implemented

Represent inference models as declarative compositions of the existing
scientific profile classes. Plain values are fixed, while `FreeParameter`
associates an initial value with a normalized prior. Support uniform, Gaussian,
log-uniform, and truncated-Gaussian priors with both JAX-compatible log density
and unit-cube transformation. Automatically flatten free values into a stable
named vector and reconstruct `LensPlane` and `CompositeLight` objects inside
the differentiable rendering calculation. The initial generic inference path
supports maximum-likelihood and maximum-a-posteriori L-BFGS-B fitting; posterior
sampling remains a subsequent milestone.

### 2026-11-19 — Dense regular pixelized source inversion implemented

Implement the Warren--Dye semi-linear inversion on a regular Cartesian source
grid. Use bilinear interpolation from ray-traced image sub-pixels, average over
detector pixels, and propagate every source basis image through the PSF to form
a dense forward operator. Provide zeroth, nearest-neighbour gradient, and axial
curvature regularization matrices, the regularized linear solution, source
covariance, image diagnostics, and a differentiable lower-level solve. Treat
this as the inspectable correctness reference for a future sparse or
matrix-free production implementation.

Absolute Gaussian evidence is reported only when the regularization matrix is
positive definite. Zeroth-order regularization meets this condition. The
current gradient and curvature matrices contain constant or planar null modes,
so their priors are improper and their absolute evidence is deliberately left
undefined until explicit boundary or null-space conventions are adopted.

### 2026-11-19 — Nonlinear lens inference coupled to pixelized sources

Connect `ProfileModel` lens declarations and nonlinear priors to the dense
semi-linear source solver. At every nonlinear evaluation, rebuild the
lensing-plus-PSF operator, solve the regularized source, and differentiate the
complete calculation with JAX. Provide source-marginalized Gaussian evidence,
penalized likelihood, and profile likelihood objectives. All objectives add
the declared nonlinear priors, producing MAP lens estimates. Permit the
regularization strength to be fixed or declared as another free parameter.
Restrict the evidence objective to positive-definite regularization while
allowing singular gradient and curvature forms with the diagnostic objectives.
