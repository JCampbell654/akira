# Project Akira tutorials

These notebooks are self-contained demonstrations of the verified features.
Run them in order for the most natural introduction:

1. `01_sis_gaussian_forward_model.ipynb` builds the forward model from its
   individual equations and compares it with the public renderer.
2. `02_imaging_psf_noise_likelihood.ipynb` adds detector convolution,
   reproducible noise, masks, and the normalized Gaussian likelihood.
3. `03_fit_einstein_radius.ipynb` scans the likelihood, inspects its gradient,
   and recovers an SIS Einstein radius from noisy imaging.

Install the notebook dependencies into the project environment with:

```bash
.venv/bin/python -m pip install -e '.[test,notebook]'
```

Select the Python interpreter at `.venv/bin/python` when opening a notebook.
All angular positions and radii are in arcseconds; `phi` is in radians.

