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

Four executable vertical slices are implemented. The first is a JAX-based,
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

## Tutorials

The [`notebooks`](notebooks/) directory contains five executable tutorials:

1. Build the SIS-plus-Gaussian forward model from its individual equations.
2. Add PSF convolution, masks, seeded noise, and image likelihoods.
3. Inspect the likelihood surface and recover an SIS Einstein radius.
4. Compose an elliptical power-law lens, shear, and Sérsic source, then recover
   its Einstein radius, 3D density slope, and source position.
5. Define priors and arbitrary free parameters, then run a generic
   six-parameter maximum-a-posteriori fit.

Install the optional notebook environment with:

```bash
.venv/bin/python -m pip install -e '.[test,notebook]'
```

Then open the notebooks using the `.venv/bin/python` interpreter. Each notebook
is self-contained and has been verified from a clean kernel.

The distribution and import names are provisional during pre-1.0 development.
