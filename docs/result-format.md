# Akira inference result format

Akira saves portable inference results as a directory containing:

- `result_summary.json`: human-readable metadata, model definition, parameter
  definitions, point estimate, and inference diagnostics;
- `result_arrays.npz`: numerical arrays referenced by the JSON manifest.

The initial format covers analytic-source optimization. Sampling information
is reserved as an optional section so posterior results can use the same
container later.

## Saving and loading

```python
from akira import (
    InferenceProblem,
    fit_model,
    load_inference_result,
    save_inference_result,
)

problem = InferenceProblem(model, observation, grid)
result = fit_model(model, observation, grid, mode="map")

save_inference_result("output/my_fit", problem, result)
stored = load_inference_result("output/my_fit")

print(stored.summary["point_estimate"]["parameters"])
model_image = stored.arrays["model_image"]
```

Loading validates the format identifier and version as well as every array's
key, shape, and data type. NPZ files are loaded with pickled objects disabled.

## Pixelized-source fits

Pixelized fits use a dedicated problem wrapper because their saved definition
also includes a source grid and regularization. The result contains everything
needed to reconstruct the model, observation, PSF, mask, image grid, source
grid, priors, fitted vector, and reconstruction products.

```python
problem = ak.PixelizedInferenceProblem(model, observation, image_grid)
fit = ak.fit_pixelized_model(model, observation, image_grid)
ak.save_pixelized_inference_result("output/pixelized_fit", problem, fit)
```

In a later process, load the complete problem and resume sampling from the
saved point estimate. The loader remembers whether log regularization strength
was optimized and configures sampling consistently.

```python
stored = ak.load_pixelized_inference_result("output/pixelized_fit")
sampling = stored.sample(
    n_walkers=40,
    n_steps=1000,
    burn_in=200,
    thin=5,
    seed=21,
)
```

## Version identification

Every summary begins with:

```json
{
  "format": "akira-inference-result",
  "format_version": 1
}
```

`format_version` describes this storage schema. It is independent of the Akira
package version recorded under `provenance.akira_version`. A future incompatible
schema change will increment `format_version`; readers must not silently
interpret an unsupported version.

## Version 1 summary structure

```text
result_summary.json
├── format
├── format_version
├── provenance
│   ├── akira_version
│   ├── created_at
│   ├── python_backend
│   ├── jax_version
│   ├── inference_method
│   ├── objective
│   └── random_seed
├── problem
│   ├── grid
│   ├── observation
│   ├── model
│   │   ├── lens
│   │   └── source
│   └── parameters
├── point_estimate
├── optimization
├── sampling
└── arrays
```

`problem.model` preserves both free and fixed profile values. Each free
parameter records its initial value and normalized prior. `problem.parameters`
also provides the flattened, stable parameter ordering used by inference
vectors.

`point_estimate` contains parameter names and values together with the log
likelihood, log prior, and log posterior. `optimization` records success,
message, and evaluation count. `sampling` is `null` in optimization-only
results.

## Version 1 arrays

The initial writer stores:

- `parameter_vector`;
- `model_image`;
- `residual_map`;
- `observation_data`;
- `noise_map`;
- `mask`;
- `psf_kernel`;
- `covariance`, when supplied by the optimizer.

The JSON `arrays` object is the authoritative manifest. Each entry records its
NPZ filename, key, shape, and data type. Consumers should use the manifest
rather than assume that every optional array exists.

## Scope and compatibility

Version 1 is an archival scientific result, not yet a complete restart file.
It preserves the assumptions and outputs needed to inspect an analysis, but it
does not reconstruct arbitrary Python objects or resume an interrupted
optimizer. Pixelized-source and posterior-sampling extensions should retain
this top-level structure and add method-specific diagnostics and arrays.
