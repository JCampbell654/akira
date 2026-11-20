from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from akira import (
    CartesianGrid,
    EllipticalPowerLaw,
    ImagingObservation,
    LensPlane,
    PSF,
    SourceGrid,
    apply_psf_to_operator,
    bilinear_interpolation_matrix,
    gaussian_psf,
    lensing_operator,
    reconstruct_source,
    regularization_matrix,
    solve_source,
)


@dataclass(frozen=True)
class ZeroDeflection:
    def deflection(self, coordinates):
        return jnp.zeros_like(coordinates)


def test_source_grid_coordinates_follow_xy_and_array_yx_conventions():
    grid = SourceGrid(shape=(3, 4), pixel_scale=0.2, centre=(0.1, -0.2))
    coordinates = np.asarray(grid.coordinates())
    assert coordinates.shape == (3, 4, 2)
    assert coordinates[0, 0] == pytest.approx([-0.2, -0.4])
    assert coordinates[-1, -1] == pytest.approx([0.4, 0.0])


def test_bilinear_interpolation_has_expected_weights_and_zero_outside():
    source_grid = SourceGrid((3, 3), 1.0)
    coordinates = jnp.asarray([[0.25, 0.5], [3.0, 0.0]])
    matrix = np.asarray(bilinear_interpolation_matrix(coordinates, source_grid))
    expected = np.zeros(9)
    expected[4] = 0.375
    expected[5] = 0.125
    expected[7] = 0.375
    expected[8] = 0.125
    assert matrix[0] == pytest.approx(expected)
    assert matrix[0].sum() == pytest.approx(1.0)
    assert matrix[1] == pytest.approx(np.zeros(9))


def test_identity_lensing_operator_on_matching_grids():
    image_grid = CartesianGrid((4, 5), pixel_scale=0.2)
    source_grid = SourceGrid((4, 5), pixel_scale=0.2)
    operator = lensing_operator(
        image_grid, source_grid, LensPlane((ZeroDeflection(),))
    )
    assert np.asarray(operator) == pytest.approx(np.eye(20), abs=1e-14)


def test_psf_operator_matches_column_by_column_convolution():
    image_shape = (5, 5)
    operator = jnp.eye(25)
    psf = gaussian_psf(0.7, size=3)
    convolved = np.asarray(apply_psf_to_operator(operator, image_shape, psf.kernel))
    central_column = convolved[:, 12].reshape(image_shape)
    expected = np.zeros(image_shape)
    expected[1:4, 1:4] = np.asarray(psf.kernel)
    assert central_column == pytest.approx(expected)


def test_regularization_matrices_equal_explicit_penalties():
    grid = SourceGrid((3, 4), 1.0)
    source = np.arange(12, dtype=float).reshape(grid.shape)
    flat = source.reshape(-1)
    zeroth = np.asarray(regularization_matrix(grid, "zeroth"))
    gradient = np.asarray(regularization_matrix(grid, "gradient"))
    curvature = np.asarray(regularization_matrix(grid, "curvature"))
    expected_gradient = (
        np.sum(np.diff(source, axis=0) ** 2)
        + np.sum(np.diff(source, axis=1) ** 2)
        + np.sum(source[:, -1] ** 2)
        + np.sum(source[-1, :] ** 2)
    )
    expected_curvature = np.sum(
        np.diff(source, n=2, axis=1) ** 2
    ) + np.sum(
        np.diff(source, n=2, axis=0) ** 2
    ) + np.sum(np.diff(source[:, -2:], axis=1) ** 2) + np.sum(
        np.diff(source[-2:, :], axis=0) ** 2
    ) + np.sum(source[:, -1] ** 2) + np.sum(source[-1, :] ** 2)
    assert flat @ zeroth @ flat == pytest.approx(np.sum(source**2))
    assert flat @ gradient @ flat == pytest.approx(expected_gradient)
    assert flat @ curvature @ flat == pytest.approx(expected_curvature)
    assert np.linalg.eigvalsh(gradient).min() > -1e-12
    assert np.linalg.eigvalsh(curvature).min() > -1e-12


@pytest.mark.parametrize("kind", ["zeroth", "gradient", "curvature"])
def test_reconstruction_solves_normal_equations(kind):
    image_grid = CartesianGrid((4, 4), 0.2)
    source_grid = SourceGrid((4, 4), 0.2)
    source = np.asarray(
        [[0.0, 0.1, 0.2, 0.0], [0.1, 0.8, 1.0, 0.2],
         [0.0, 0.5, 0.7, 0.1], [0.0, 0.0, 0.1, 0.0]]
    )
    observation = ImagingObservation(
        data=source,
        noise_map=np.full((4, 4), 0.01),
        psf=PSF.delta(),
        pixel_scale=0.2,
    )
    result = reconstruct_source(
        observation,
        image_grid,
        source_grid,
        LensPlane((ZeroDeflection(),)),
        regularization_kind=kind,
        regularization_strength=1e-6,
    )
    rhs = np.asarray(result.operator.T @ (observation.data.reshape(-1) / 0.01**2))
    assert np.asarray(result.system_matrix @ result.source.reshape(-1)) == pytest.approx(
        rhs, rel=1e-10, abs=1e-10
    )
    assert np.asarray(result.source) == pytest.approx(source, abs=1e-8)
    assert result.source_covariance.shape == (16, 16)
    assert result.log_evidence is not None
    assert result.posterior_covariance.shape == (16, 16)


def test_zeroth_order_log_evidence_matches_direct_gaussian_marginal():
    image_grid = CartesianGrid((2, 2), 0.2)
    source_grid = SourceGrid((2, 2), 0.2)
    data = np.asarray([[0.2, -0.1], [0.4, 0.05]])
    noise_sigma = 0.3
    strength = 2.5
    observation = ImagingObservation(
        data=data,
        noise_map=np.full((2, 2), noise_sigma),
        psf=PSF.delta(),
        pixel_scale=0.2,
    )
    result = reconstruct_source(
        observation,
        image_grid,
        source_grid,
        LensPlane((ZeroDeflection(),)),
        regularization_kind="zeroth",
        regularization_strength=strength,
    )
    covariance = (noise_sigma**2 + 1.0 / strength) * np.eye(4)
    expected = multivariate_normal.logpdf(data.reshape(-1), mean=np.zeros(4), cov=covariance)
    assert result.log_evidence == pytest.approx(expected, rel=1e-11)


def test_lensing_operator_is_differentiable_with_respect_to_einstein_radius():
    image_grid = CartesianGrid((5, 5), 0.15)
    source_grid = SourceGrid((7, 7), 0.12)
    weights = jnp.linspace(0.1, 1.0, image_grid.shape[0] * image_grid.shape[1])

    def summary(einstein_radius):
        profile = EllipticalPowerLaw.tree_unflatten(
            None, (einstein_radius, 2.0, (0.0, 0.0), 0.8, 0.2)
        )
        operator = lensing_operator(image_grid, source_grid, LensPlane((profile,)))
        return jnp.sum(operator @ jnp.linspace(0.0, 1.0, source_grid.size) * weights)

    value = 0.25
    automatic = float(jax.grad(summary)(value))
    step = 1e-5
    finite = float((summary(value + step) - summary(value - step)) / (2 * step))
    assert automatic == pytest.approx(finite, rel=2e-5, abs=2e-5)


def test_linear_solution_is_differentiable_with_respect_to_regularization():
    operator = jnp.asarray([[1.0, 0.2], [0.1, 0.8], [0.4, 0.3]])
    data = jnp.asarray([0.7, 0.2, 0.5])
    noise = jnp.asarray([0.1, 0.15, 0.12])
    regularization = jnp.eye(2)

    def source_sum(strength):
        source, _, _, _ = solve_source(
            operator, data, noise, regularization, strength
        )
        return source.sum()

    strength = 2.0
    automatic = float(jax.grad(source_sum)(strength))
    step = 1e-4
    finite = float(
        (source_sum(strength + step) - source_sum(strength - step)) / (2 * step)
    )
    assert automatic == pytest.approx(finite, rel=1e-7, abs=1e-9)


def test_invalid_source_grid_regularization_and_strength_are_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        SourceGrid((1, 3), 0.1)
    with pytest.raises(ValueError, match="kind"):
        regularization_matrix(SourceGrid((3, 3), 0.1), "invalid")
    observation = ImagingObservation(
        np.ones((3, 3)), np.ones((3, 3)), PSF.delta(), 0.1
    )
    with pytest.raises(ValueError, match="positive"):
        reconstruct_source(
            observation,
            CartesianGrid((3, 3), 0.1),
            SourceGrid((3, 3), 0.1),
            LensPlane((ZeroDeflection(),)),
            regularization_strength=0.0,
        )
