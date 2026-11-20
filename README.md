# Strong Gravitational Lens Modelling Code

This project will build a fast, capable, open strong-gravitational-lens modelling package, inspired by PyAutoLens and HercuLens.

## Project goals

- Provide accurate forward modelling of strong gravitational lenses.
- Keep the public API clear enough for scientific use and extension.
- Achieve high performance through vectorized numerical code and optional accelerator support.
- Support reproducible optimisation and Bayesian inference workflows.
- Validate numerical results against analytic solutions and established lensing packages.

## Initial scope

The first useful version should include:

- image-plane and source-plane coordinate grids;
- analytic lens mass profiles and source/lens light profiles;
- ray tracing through one or more lens planes;
- image rendering, PSF convolution, masks, noise maps, and likelihoods;
- parameter definitions and interfaces to optimizers and posterior samplers;
- tests for physical correctness, numerical stability, and performance;
- examples that progress from simulation to inference.

## Technical direction

Python will be the user-facing language. The numerical backend has not yet been finalized; NumPy and JAX are leading candidates. Architecture decisions should preserve a clean separation between the scientific model, numerical backend, and inference machinery.

See [docs/architecture.md](docs/architecture.md) for the evolving design and decision log.

## Current status

Seven executable vertical slices are implemented. The first is a JAX-based,
float64, single-plane SIS lens with an elliptical Gaussian source, Cartesian
image grids, sub-pixel integration, ray tracing, and independent NumPy/analytic
verification. The second adds normalized PSFs, zero-padded convolution,
validated observations, masks, reproducible Gaussian noise, normalized image
likelihoods, and gradient-based recovery of an SIS Einstein radius. The third
adds an elliptical power-law mass profile, external shear, elliptical Sérsic
light, composable mass and light components, and a controlled four-parameter
fit of Einstein radius, 3D density slope, and source position.
The fourth introduces general fixed/free parameter declarations, uniform,
Gaussian, log-uniform, and truncated-Gaussian priors, automatic parameter
vector construction, and generic maximum-likelihood or MAP optimization for
any composition of the currently implemented analytic profiles.
The fifth adds a dense, differentiable reference implementation of regular
pixelized source reconstruction: bilinear ray-tracing, PSF propagation,
Warren--Dye semi-linear inversion, three regularization forms, source
covariance, and proper-Gaussian evidence where mathematically defined.
The sixth connects free nonlinear lens parameters and their priors to this
linear inversion, differentiating through operator construction, source
solution, and Gaussian source marginalization for evidence-based lens fitting.
The seventh implements Suyu-style Bayesian regularization selection: proper
boundary-treated gradient and curvature priors, direct optimization in
log-strength, effective source dimension, stationarity diagnostics, and
evidence-based comparison of regularization forms.
The lens-mapping Jacobian can also be evaluated on a fine image-plane grid to
find critical curves and ray-trace them into source-plane caustics. These can
be overlaid on any analytic or pixelized source plot:

```python
from akira import CartesianGrid, find_lensing_curves, plot_caustics

search_grid = CartesianGrid((301, 301), pixel_scale=0.01)
curves = find_lensing_curves(lens, search_grid)
plot_caustics(axis, curves, color="cyan")
```

The search grid is independent of the data grid: make it fine enough to trace
the curve smoothly and wide enough to contain all critical curves.

Run the tests from the repository root:

```bash
python -m pytest
```

Run the Einstein-ring example after installing the package, or directly from a
development checkout by placing `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src python examples/sis_gaussian.py
```

Run the noisy parameter-recovery example from the project environment:

```bash
.venv/bin/python examples/fit_sis_gaussian.py
```

Run the elliptical analytic model and four-parameter recovery example with:

```bash
.venv/bin/python examples/fit_epl_sersic.py
```

## Fit real imaging

Copy [`examples/real_data_config.json`](examples/real_data_config.json), point
it at a two-dimensional image, noise map, normalized-or-unnormalized PSF and
optional mask, then adjust the pixel scale, source grid, starting values and
priors. Inputs can be FITS, NPY, NPZ, CSV or whitespace-delimited text. For
FITS data, install the real-data extras and run:

```bash
.venv/bin/python -m pip install -e '.[real-data]'
.venv/bin/python examples/fit_real_data.py --config my_config.json
```

The fit jointly optimizes all parameters marked with a prior and the logarithm
of the regularization strength. It writes `fit_summary.json`, `fit_arrays.npz`,
and `fit_diagnostics.png` to the configured output directory. Mask values that
convert to `True` are excluded. Until simultaneous lens-light modelling is
implemented, the input image should have foreground lens light subtracted or
those pixels should be masked.

## Tutorials

The [`notebooks`](notebooks/) directory contains eight executable tutorials:

1. Build the SIS-plus-Gaussian forward model from its individual equations.
2. Add PSF convolution, masks, seeded noise, and image likelihoods.
3. Inspect the likelihood surface and recover an SIS Einstein radius.
4. Compose an elliptical power-law lens, shear, and Sérsic source, then recover
   its Einstein radius, 3D density slope, and source position.
5. Define priors and arbitrary free parameters, then run a generic
   six-parameter maximum-a-posteriori fit.
6. Reconstruct a non-parametric source and compare zeroth, gradient, and
   curvature regularization.
7. Infer free lens parameters while solving and marginalizing a pixelized
   source at every nonlinear evaluation.
8. Optimize regularization strength and evidence-rank zeroth, gradient, and
   curvature source priors.

Install the optional notebook environment with:

```bash
.venv/bin/python -m pip install -e '.[test,notebook]'
```

Then open the notebooks using the `.venv/bin/python` interpreter. Each notebook
is self-contained and has been verified from a clean kernel.

The distribution and import names are provisional during pre-1.0 development.
