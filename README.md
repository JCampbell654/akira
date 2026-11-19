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

The first executable vertical slice is implemented: a JAX-based, float64,
single-plane SIS lens with an elliptical Gaussian source, Cartesian image grids,
sub-pixel integration, ray tracing, and independent NumPy/analytic verification.

Run the tests from the repository root:

```bash
python -m pytest
```

Run the Einstein-ring example after installing the package, or directly from a
development checkout by placing `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src python examples/sis_gaussian.py
```

The distribution and import names are provisional during pre-1.0 development.
