# Project Akira tutorials

These notebooks are self-contained demonstrations of the verified features.
Run them in order for the most natural introduction:

1. `01_sis_gaussian_forward_model.ipynb` builds the forward model from its
   individual equations and compares it with the public renderer.
2. `02_imaging_psf_noise_likelihood.ipynb` adds detector convolution,
   reproducible noise, masks, and the normalized Gaussian likelihood.
3. `03_fit_einstein_radius.ipynb` scans the likelihood, inspects its gradient,
   and recovers an SIS Einstein radius from noisy imaging.
4. `04_elliptical_power_law_sersic.ipynb` composes an elliptical power-law
   lens, external shear, and Sérsic source, verifies the SIS limit, and fits
   the Einstein radius, 3D density slope, and source position.
5. `05_general_inference_and_priors.ipynb` declares arbitrary fixed and free
   profile parameters, applies several prior families, and performs a generic
   six-parameter maximum-a-posteriori fit.
6. `06_pixelized_source_reconstruction.ipynb` builds the dense lensing operator,
   reconstructs a regular pixelized source, and compares zeroth, gradient, and
   curvature regularization.
7. `07_free_lens_pixelized_inference.ipynb` optimizes nonlinear power-law lens
   parameters while solving and marginalizing a pixelized source at every
   evaluation.
8. `08_optimize_regularization.ipynb` optimizes log regularization strength,
   reports the effective source dimension and Suyu stationarity condition, and
   evidence-ranks proper zeroth, gradient, and curvature priors.
9. `09_construct_inference_problem.ipynb` constructs the common analytic
   inference definition, inspects its parameter contract, and evaluates its
   differentiable likelihood, prior, posterior, and prediction.
10. `10_result_format_and_handling.ipynb` saves a fitted problem in the
    versioned JSON-plus-NPZ result format, inspects its schema, reloads it with
    validation, and works with the portable arrays.
11. `11_build_a_modelling_script.ipynb` assembles an end-to-end pixelized-source
    modelling script from observation arrays, grids, priors, and profile models
    through optimization, posterior sampling, diagnostics, and production
    script design.

Install the notebook dependencies into the project environment with:

```bash
.venv/bin/python -m pip install -e '.[test,notebook]'
```

Select the Python interpreter at `.venv/bin/python` when opening a notebook.
All angular positions and radii are in arcseconds; `phi` is in radians.
